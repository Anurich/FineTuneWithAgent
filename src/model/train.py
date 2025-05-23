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
        self.performance_history = deque(maxlen=50)  # Increased for stability
        self.feedback_momentum = {'trend': 0.0, 'strength': 0.0}
        self.step_counter = 0
        
        # Regularization parameters
        self.l2_strength = 1e-5
        self.label_smoothing = 0.1
        
        # Meta-learning parameters (more conservative)
        self.warmup_steps = 100  # Longer warmup
        self.feedback_interval = 100  # Less frequent feedback
        self.min_history_for_meta = 10  # More history required
        self.meta_lr_mult = 0.005  # Smaller adjustments

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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

        # Get the base loss & model outputs
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # CRITICAL FIX: Only modify loss during training, never during evaluation
        if mode == "eval":
            # Store metrics for evaluation but don't modify loss
            detached_loss = loss.detach().clone()
            self._store_eval_metrics(inputs, outputs, detached_loss)
            return (loss, outputs) if return_outputs else loss

        # Training mode: apply modifications
        self.step_counter += 1
        detached_loss = loss.detach().clone()
        
        # Count tokens for training
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

        # Apply consistent regularization to all training steps
        modified_loss = self._apply_regularization(loss, inputs, outputs)

        # Compute token accuracy and collect feedback (less frequently)
        if "labels" in inputs and not self.args.use_liger_kernel:
            token_accuracy = self._compute_token_accuracy(inputs, outputs)
            self._metrics[mode]["mean_token_accuracy"].append(token_accuracy)
            
            # Collect feedback less frequently and only during training
            if self.step_counter % self.feedback_interval == 0:
                self._collect_feedback(inputs, outputs, token_accuracy, mode)
            
            # Apply meta-learning adjustments more conservatively
            if self._should_apply_meta_adjustment():
                modified_loss = self._apply_meta_adjustment(modified_loss, inputs, outputs)

        return (modified_loss, outputs) if return_outputs else modified_loss

    def _store_eval_metrics(self, inputs, outputs, loss):
        """Store evaluation metrics without modifying loss"""
        if "labels" in inputs and not self.args.use_liger_kernel:
            token_accuracy = self._compute_token_accuracy(inputs, outputs)
            self._metrics["eval"]["mean_token_accuracy"].append(token_accuracy)
            
            # Collect feedback for monitoring (but don't use for loss modification)
            if len(self.performance_history) == 0 or self.step_counter % (self.feedback_interval * 2) == 0:
                self._collect_feedback(inputs, outputs, token_accuracy, "eval")

    def _apply_regularization(self, loss, inputs, outputs):
        """Apply consistent regularization to all training steps"""
        regularized_loss = loss
        
        # L2 regularization
        if self.l2_strength > 0:
            l2_penalty = sum(p.pow(2.0).sum() for p in self.model.parameters() if p.requires_grad)
            regularized_loss = regularized_loss + self.l2_strength * l2_penalty
        
        # Label smoothing (simple implementation)
        if self.label_smoothing > 0 and "labels" in inputs:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            vocab_size = shift_logits.size(-1)
            
            # Create smoothed targets
            mask = shift_labels != -100
            if mask.any():
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                smooth_loss = -log_probs.mean(dim=-1)
                
                # Apply label smoothing
                smoothed_loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
                smoothed_loss = smoothed_loss[mask].mean()
                
                # Blend with original loss
                regularized_loss = 0.8 * regularized_loss + 0.2 * smoothed_loss
        
        return regularized_loss

    def _compute_token_accuracy(self, inputs, outputs):
        """Compute token-level accuracy"""
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
        return (corr.sum() / tot_sum).item() if tot_sum > 0 else 0.0

    def _collect_feedback(self, inputs, outputs, token_accuracy, mode):
        """Collect feedback for meta-learning (with error handling)"""
        try:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)
            
            # Sample a smaller subset for efficiency
            sample_size = min(5, preds.size(0))  # Reduced sample size
            pred_sample = preds[:sample_size].detach().cpu().tolist()
            lbl_sample = shift_labels[:sample_size].detach().cpu().tolist()
            
            # Pad -100 to pad_token_id for decoding
            lbl_dec = [[tok if tok != -100 else self.tokenizer.pad_token_id for tok in seq]
                       for seq in lbl_sample]
            gt_texts = self.tokenizer.batch_decode(lbl_dec, skip_special_tokens=True)
            pred_texts = self.tokenizer.batch_decode(pred_sample, skip_special_tokens=True)
            
            if gt_texts and pred_texts:
                feedback = eval.evaluate(gt_texts, pred_texts)
                content = feedback["content"].replace('```json','').replace('```','')
                data = json.loads(content)
                
                # Apply sigmoid to z-scores more safely
                sol_z = np.clip(data.get("solution_score", 0.0), -10, 10)
                rea_z = np.clip(data.get("reasoning_score", 0.0), -10, 10)
                sol = 1/(1+np.exp(-sol_z))
                rea = 1/(1+np.exp(-rea_z))
                
                perf = {
                    'is_correct': data.get("is_correct", False),
                    'reasoning_score': float(rea),
                    'solution_score': float(sol),
                    'token_accuracy': token_accuracy,
                    'step': self.step_counter,
                    'mode': mode
                }
                self.performance_history.append(perf)
                
                # Update momentum only with sufficient history
                if len(self.performance_history) >= self.min_history_for_meta:
                    self._update_feedback_momentum()
            else:
                # Add neutral entry when no valid texts
                self.performance_history.append({
                    'is_correct': False,
                    'reasoning_score': 0.5,
                    'solution_score': 0.5,
                    'token_accuracy': token_accuracy,
                    'step': self.step_counter,
                    'mode': mode
                })
                
        except Exception as e:
            print(f"Warning: feedback collection failed at step {self.step_counter}: {e}")
            # Add neutral entry on failure
            self.performance_history.append({
                'is_correct': False,
                'reasoning_score': 0.5,
                'solution_score': 0.5,
                'token_accuracy': token_accuracy,
                'step': self.step_counter,
                'mode': mode
            })

    def _should_apply_meta_adjustment(self):
        """Determine if meta-learning adjustment should be applied"""
        return (len(self.performance_history) >= self.min_history_for_meta and 
                self.step_counter > self.warmup_steps and
                abs(self.feedback_momentum['trend']) > 0.02 and  # More conservative threshold
                self.feedback_momentum['strength'] > 0.2)

    def _apply_meta_adjustment(self, loss, inputs, outputs):
        """Apply meta-learning adjustment more conservatively"""
        try:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            mask = shift_labels != -100
            
            if not mask.any():
                return loss
            
            # Compute metrics more safely
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            probs = torch.softmax(shift_logits, dim=-1)
            
            # Get valid indices
            valid_indices = torch.where(mask)
            if len(valid_indices[0]) == 0:
                return loss
            
            # Token-level log probabilities and entropy
            token_log_probs = log_probs[valid_indices[0], valid_indices[1], shift_labels[valid_indices]]
            mean_log_prob = token_log_probs.mean()
            
            token_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            mean_entropy = token_entropy[mask].mean()
            
            # Simple, conservative adjustment
            trend = self.feedback_momentum['trend']
            strength = self.feedback_momentum['strength']
            
            if trend > 0:
                # Model is improving - slightly encourage diversity
                adjustment = -self.meta_lr_mult * trend * strength * mean_entropy
            else:
                # Model is declining - slightly encourage confidence
                adjustment = -self.meta_lr_mult * abs(trend) * strength * mean_log_prob
            
            # Clamp adjustment to prevent instability
            max_adjustment = 0.05 * loss.abs().item()
            adjustment = torch.clamp(adjustment, -max_adjustment, max_adjustment)
            
            return loss + adjustment
            
        except Exception as e:
            print(f"Warning: meta-adjustment failed at step {self.step_counter}: {e}")
            return loss

    def _update_feedback_momentum(self):
        """Update feedback momentum with more stable calculation"""
        # Use more history for stability
        history = list(self.performance_history)[-20:]  # Increased window
        if len(history) < self.min_history_for_meta:
            self.feedback_momentum['trend'] = 0.0
            self.feedback_momentum['strength'] = 0.0
            return
        
        # Compute combined score
        combined_scores = []
        for p in history:
            # Weight recent scores more heavily
            combined = (p['reasoning_score'] + p['solution_score']) / 2.0
            combined_scores.append(combined)
        
        # Use more stable trend calculation
        if len(combined_scores) >= 5:
            # Compare first and last thirds for trend
            first_third = np.mean(combined_scores[:len(combined_scores)//3])
            last_third = np.mean(combined_scores[-len(combined_scores)//3:])
            trend = last_third - first_third
        else:
            trend = 0.0
        
        # More stable strength calculation
        variance = np.var(combined_scores) if len(combined_scores) > 1 else 1.0
        strength = 1.0 / (1.0 + variance + 1e-3)  # Less sensitive to variance
        
        # Apply momentum smoothing
        alpha = 0.7  # Higher smoothing
        self.feedback_momentum['trend'] = alpha * self.feedback_momentum['trend'] + (1 - alpha) * trend
        self.feedback_momentum['strength'] = alpha * self.feedback_momentum['strength'] + (1 - alpha) * strength

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