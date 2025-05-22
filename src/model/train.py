from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import Trainer
from datasets import Dataset
from transformers import default_data_collator
import pandas as pd
from collections import defaultdict, deque
from Agent.evaluation_agent.evaluate import Evaluate
import torch
import numpy as np
import json

eval = Evaluate()

class CustomSFTTrainer(Trainer):
    def __init__(self, *args, processing_class=None, **kwargs):
        # Pass tokenizer as processing_class per Transformers ≥5.x
        super().__init__(*args, processing_class=processing_class, **kwargs)
        # Backwards compatibility
        self.tokenizer = self.processing_class

        # Initialize feedback history and counters
        self._total_train_tokens = 0
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.performance_history = deque(maxlen=20)
        self.feedback_momentum = {'trend': 0.0, 'strength': 0.0}
        self.step_counter = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        mode = "eval" if self.control.should_evaluate else "train"

        # Normalize inputs if passed as list
        if isinstance(inputs, list):
            if inputs and isinstance(inputs[0], dict):
                inputs = {
                    k: torch.stack([item[k] for item in inputs])
                        if isinstance(inputs[0][k], torch.Tensor) else
                        [item[k] for item in inputs]
                    for k in inputs[0]
                }
            else:
                raise ValueError("Expected inputs to be a dictionary or list of dicts")

        # Get the base loss & model outputs (loss has grad)
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        # Detached copy for metrics & potential meta adjustments logging
        detached_loss = loss.detach().clone()

        if mode == "train":
            self.step_counter += 1
            # Count tokens
            if "attention_mask" in inputs:
                num_tokens = self.accelerator.gather_for_metrics(
                    inputs["attention_mask"].sum()
                ).sum().item()
            elif "position_ids" in inputs:
                local = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens = self.accelerator.gather_for_metrics(local).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100

            correct = (preds == shift_labels) & mask
            tot = mask.sum()
            corr = correct.sum()
            corr = self.accelerator.gather_for_metrics(corr)
            tot = self.accelerator.gather_for_metrics(tot)
            tot_sum = tot.sum()
            acc = (corr.sum() / tot_sum).item() if tot_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(acc)

            # META-LEARNING FEEDBACK
            feedback_interval = 50
            if mode == "eval" or (mode == "train" and self.step_counter % feedback_interval == 0):
                sample_size = (min(max(10, preds.size(0) // 10), 20)
                               if mode == "train" else preds.size(0))
                actual_n = min(sample_size, preds.size(0))
                pred_sample = preds[:actual_n].detach().cpu().tolist()
                lbl_sample = shift_labels[:actual_n].detach().cpu().tolist()
                # Pad -100 to pad_token_id for decoding
                lbl_dec = [[tok if tok != -100 else self.tokenizer.pad_token_id for tok in seq]
                           for seq in lbl_sample]
                gt_texts = self.tokenizer.batch_decode(lbl_dec, skip_special_tokens=True)
                pred_texts = self.tokenizer.batch_decode(pred_sample, skip_special_tokens=True)
                try:
                    if gt_texts and pred_texts:
                        feedback = eval.evaluate(gt_texts, pred_texts)
                        content = feedback["content"].replace('```json','').replace('```','')
                        data = json.loads(content)
                        sol_z = data.get("solution_score", 0.0)
                        rea_z = data.get("reasoning_score", 0.0)
                        sol = 1/(1+np.exp(-sol_z))
                        rea = 1/(1+np.exp(-rea_z))
                        perf = {
                            'is_correct': data.get("is_correct", False),
                            'reasoning_score': float(rea),
                            'solution_score': float(sol),
                            'token_accuracy': acc,
                            'step': self.step_counter,
                            'mode': mode
                        }
                        self.performance_history.append(perf)
                        if len(self.performance_history) >= 3:
                            self._update_feedback_momentum()
                    else:
                        # skip or append neutral
                        self.performance_history.append({
                            'is_correct': False,
                            'reasoning_score': 0.5,
                            'solution_score': 0.5,
                            'token_accuracy': acc,
                            'step': self.step_counter,
                            'mode': mode
                        })
                except Exception as e:
                    print(f"Warning: feedback failed at step {self.step_counter}: {e}")
                    self.performance_history.append({
                        'is_correct': False,
                        'reasoning_score': 0.5,
                        'solution_score': 0.5,
                        'token_accuracy': acc,
                        'step': self.step_counter,
                        'mode': mode
                    })

            # META-ADJUSTMENT OF LOSS
            if mode == "train":
                warmup = 50
                min_history = 3
                base_lr_mult = 0.01
                if len(self.performance_history) >= min_history and self.step_counter > warmup:
                    trend = self.feedback_momentum['trend']
                    strength = self.feedback_momentum['strength']
                    if abs(trend) > 0.05 and strength > 0.1:
                        # compute L2
                        l2_strength = 1e-5
                        l2 = l2_strength * torch.norm(shift_logits, p=2)
                        # log probs & entropy
                        logp = torch.log_softmax(shift_logits, dim=-1)
                        bidx, sidx = torch.where(mask)
                        if bidx.numel() > 0:
                            token_lp = logp[bidx, sidx, shift_labels[bidx, sidx]]
                            mean_lp = token_lp.mean()
                            probs = torch.softmax(shift_logits, dim=-1)
                            ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
                            mean_ent = ent[mask].mean()
                        else:
                            mean_lp = torch.tensor(0.0, device=loss.device)
                            mean_ent = torch.tensor(0.0, device=loss.device)
                        # build adjustment
                        if trend > 0:
                            adj = -base_lr_mult * trend * strength * (-mean_lp)
                        else:
                            adj = base_lr_mult * abs(trend) * strength * (1.0/(mean_ent+1e-9))
                        # apply on real loss
                        loss_l2 = loss + l2
                        max_adj = 0.1 * loss_l2.abs().item()
                        adj_clamped = torch.clamp(adj, -max_adj, max_adj)
                        loss = loss_l2 + adj_clamped
        # return
        return (loss, outputs) if return_outputs else loss

    def _update_feedback_momentum(self):
        history = list(self.performance_history)[-10:]
        if len(history) < 3:
            self.feedback_momentum['trend'] = 0.0
            self.feedback_momentum['strength'] = 0.0
            return
        combined = [ (p['reasoning_score'] + p['solution_score'])/2.0 for p in history ]
        alpha = 0.3
        ewma = [combined[0]]
        for val in combined[1:]:
            ewma.append(alpha*val + (1-alpha)*ewma[-1])
        trend = ewma[-1] - ewma[-2]
        var = np.var(combined)
        strength = 1.0/(1.0 + var + 1e-6)
        self.feedback_momentum['trend'] = float(trend)
        self.feedback_momentum['strength'] = float(strength)

class MODEL:
    def __init__(self, train_data, test_data):
        self.max_seq_length = 2048
        self.lora_rank = 32
        self.train_data = train_data
        self.test_data = test_data
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-4B-Base",
            max_seq_length=self.max_seq_length,
            load_in_4bit=False,
            fast_inference=True,
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=0.7,
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.lora_rank*2,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        self.reasoning_start = "<start_working_out>"
        self.reasoning_end   = "<end_working_out>"
        self.solution_start  = "<SOLUTION>"
        self.solution_end    = "</SOLUTION>"

        self.system_prompt = (
            f"You are given a problem.\n"
            f"Think about the problem and provide your working out.\n"
            f"Place it between {self.reasoning_start} and {self.reasoning_end}.\n"
            f"Then, provide your solution between {self.solution_start}{self.solution_end}"
        )

        chat_tpl = (
            "{% if messages[0]['role'] == 'system' %}"  \
            "{{ messages[0]['content'] + eos_token }}"    \
            "{% set loop_messages = messages[1:] %}"     \
            "{% else %}"                                \
            "{{ '{system_prompt}' + eos_token }}"        \
            "{% set loop_messages = messages %}"        \
            "{% endif %}"                               \
            "{% for message in loop_messages %}"         \
            "{% if message['role'] == 'user' %}"         \
            "{{ message['content'] }}"                   \
            "{% elif message['role'] == 'assistant' %}"   \
            "{{ message['content'] + eos_token }}"       \
            "{% endif %}"                               \
            "{% endfor %}"                              \
            "{% if add_generation_prompt %}{{ '{reasoning_start}' }}" \
            "{% endif %}"
        )
        chat_tpl = chat_tpl.replace("'{system_prompt}'", f"'{self.system_prompt}'")\
                        .replace("'{reasoning_start}'", f"'{self.reasoning_start}'")
        self.tokenizer.chat_template = chat_tpl

    def format_dataset(self, x):
        ans = str(x["answer"])
        prob = str(x["question"]).strip()
        thoughts = str(x.get("reasoning","")).strip()
        prompt = (
            self.reasoning_start + thoughts + self.reasoning_end +
            self.solution_start + ans + self.solution_end
        )
        return [
            {"role": "system",    "content": self.system_prompt},
            {"role": "user",      "content": prob},
            {"role": "assistant", "content": prompt},
        ]

    def data_collator(self, batch):
        texts = [ex["text"] for ex in batch]
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        toks["labels"] = toks["input_ids"].clone()
        return toks

    def _train(self):
        import pandas as pd
        self.train_data = pd.DataFrame(self.train_data)
        self.test_data  = pd.DataFrame(self.test_data)

        self.train_data["Messages"] = self.train_data.apply(self.format_dataset, axis=1)
        self.test_data["Messages"]  = self.test_data.apply(self.format_dataset, axis=1)

        self.train_data["text"] = self.tokenizer.apply_chat_template(
            self.train_data["Messages"].tolist(), tokenize=False
        )
        self.test_data["text"]  = self.tokenizer.apply_chat_template(
            self.test_data["Messages"].tolist(), tokenize=False
        )

        self.train_dataset = Dataset.from_pandas(self.train_data)
        self.test_dataset  = Dataset.from_pandas(self.test_data)

        self.trainer = CustomSFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=1,
                warmup_steps=5,
                do_eval=True,
                num_train_epochs=50,
                learning_rate=2e-4,
                logging_steps=200,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                eval_steps=200,
                eval_strategy="steps",
                seed=3407,
                report_to="none",
                remove_unused_columns=False
            ),
            data_collator=self.data_collator,
        )
        self.trainer.train()

    def _prediction(self):
        return self.trainer.performance_history
