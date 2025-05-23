from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import Trainer
from datasets import Dataset
from transformers import default_data_collator
import pandas as pd
from collections import defaultdict, deque
from Agent.evaluation_agent.evaluate import Evaluate
import torch
import torch.nn.functional as F
import numpy as np
import json
import math

eval = Evaluate()

class MathematicallyPrincipledSFTTrainer(Trainer):
    def __init__(self, *args, processing_class=None, **kwargs):
        super().__init__(*args, processing_class=processing_class, **kwargs)
        self.tokenizer = self.processing_class
        
        # Core tracking
        self._total_train_tokens = 0
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.step_counter = 0
        
        # Mathematical feedback integration parameters
        self.feedback_history = deque(maxlen=100)
        self.feedback_buffer = deque(maxlen=20)  # Recent feedback for current batch
        
        # Confidence-based adjustment parameters
        self.confidence_threshold = 0.7  # Only adjust when model is confident
        self.feedback_strength = 0.1     # Base feedback strength
        self.uncertainty_penalty = 0.05  # Penalty for high entropy when wrong
        
        # Momentum and stability
        self.feedback_momentum = 0.0
        self.momentum_decay = 0.9
        self.stability_threshold = 0.02  # Minimum change required for adjustment
        
        # Evaluation stability
        self.eval_mode = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Mathematically principled loss computation with Agent B feedback integration
        """
        # Detect evaluation mode
        self.eval_mode = not model.training
        
        # Normalize inputs
        if isinstance(inputs, list):
            if inputs and isinstance(inputs[0], dict):
                inputs = {
                    k: torch.stack([item[k] for item in inputs])
                        if isinstance(inputs[0][k], torch.Tensor) else
                        [item[k] for item in inputs]
                    for k in inputs[0]
                }
        
        # Get base loss and outputs
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        # CRITICAL: Only apply feedback adjustments during training
        if self.eval_mode:
            self._track_eval_metrics(inputs, outputs, loss.detach())
            return (loss, outputs) if return_outputs else loss
        
        # Training mode: apply mathematical feedback integration
        self.step_counter += 1
        modified_loss = self._apply_feedback_guided_loss(loss, inputs, outputs)
        
        # Track training metrics
        self._track_training_metrics(inputs, outputs, loss.detach())
        
        return (modified_loss, outputs) if return_outputs else modified_loss
    
    def _apply_feedback_guided_loss(self, base_loss, inputs, outputs):
        """
        Mathematical approach to integrate Agent B feedback:
        
        L_total = L_base + α * L_confidence + β * L_feedback + γ * L_consistency
        
        Where:
        - L_confidence: Penalizes high entropy when model should be confident
        - L_feedback: Adjusts based on Agent B's quality assessment
        - L_consistency: Encourages consistent reasoning patterns
        """
        
        if "labels" not in inputs:
            return base_loss
            
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        mask = shift_labels != -100
        
        if not mask.any():
            return base_loss
        
        # 1. Compute confidence metrics
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Entropy as uncertainty measure
        entropy = -(probs * log_probs).sum(dim=-1)
        mean_entropy = entropy[mask].mean()
        
        # Max probability as confidence measure
        max_probs = probs.max(dim=-1)[0]
        mean_confidence = max_probs[mask].mean()
        
        # 2. Collect Agent B feedback for current predictions
        current_feedback = self._collect_agent_feedback(inputs, outputs)
        
        # 3. Compute feedback-guided adjustments
        confidence_loss = self._compute_confidence_loss(mean_entropy, mean_confidence, current_feedback)
        feedback_loss = self._compute_feedback_loss(shift_logits, shift_labels, mask, current_feedback)
        consistency_loss = self._compute_consistency_loss(shift_logits, mask)
        
        # 4. Combine losses with adaptive weights
        alpha, beta, gamma = self._compute_adaptive_weights(current_feedback)
        
        total_loss = (base_loss + 
                     alpha * confidence_loss + 
                     beta * feedback_loss + 
                     gamma * consistency_loss)
        
        return total_loss
    
    def _compute_confidence_loss(self, entropy, confidence, feedback):
        """
        Penalize high uncertainty when the model should be confident
        
        L_confidence = H(p) * (1 - quality_score) * strength
        """
        if feedback is None:
            return torch.tensor(0.0, device=entropy.device)
        
        # Higher penalty when Agent B says quality is low but model is uncertain
        quality_score = (feedback.get('reasoning_score', 0.5) + 
                        feedback.get('solution_score', 0.5)) / 2.0
        
        # Penalty increases with entropy when quality should be high
        expected_confidence = max(0.3, quality_score)  # Minimum expected confidence
        confidence_gap = max(0.0, expected_confidence - confidence.item())
        
        confidence_penalty = entropy * confidence_gap * self.uncertainty_penalty
        
        return confidence_penalty
    
    def _compute_feedback_loss(self, logits, labels, mask, feedback):
        """
        Direct feedback integration using KL divergence adjustment
        
        L_feedback = KL(p_model || p_target) where p_target is adjusted based on feedback
        """
        if feedback is None:
            return torch.tensor(0.0, device=logits.device)
        
        # Get model predictions
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute per-token negative log likelihood
        nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        masked_nll = nll[mask]
        
        if len(masked_nll) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Adjust loss based on feedback quality
        quality_score = (feedback.get('reasoning_score', 0.5) + 
                        feedback.get('solution_score', 0.5)) / 2.0
        
        # If quality is low, increase the loss to push model away from this pattern
        # If quality is high, slightly decrease loss to reinforce good patterns
        quality_adjustment = 2.0 - 2.0 * quality_score  # Range: [0, 2]
        
        adjusted_nll = masked_nll * quality_adjustment
        feedback_loss = adjusted_nll.mean() * self.feedback_strength
        
        return feedback_loss
    
    def _compute_consistency_loss(self, logits, mask):
        """
        Encourage consistent reasoning patterns using temporal smoothness
        """
        if len(self.feedback_buffer) < 2:
            return torch.tensor(0.0, device=logits.device)
        
        # Simple consistency: penalize rapid changes in prediction confidence
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        
        if mask.size(1) < 2:
            return torch.tensor(0.0, device=logits.device)
        
        # Temporal smoothness: adjacent tokens should have similar confidence
        confidence_diff = torch.abs(max_probs[:, 1:] - max_probs[:, :-1])
        smoothness_penalty = confidence_diff[mask[:, 1:]].mean()
        
        return smoothness_penalty * 0.01  # Small weight for consistency
    
    def _compute_adaptive_weights(self, feedback):
        """
        Compute adaptive weights based on feedback quality and model confidence
        """
        base_alpha, base_beta, base_gamma = 0.1, 0.2, 0.05
        
        if feedback is None:
            return base_alpha, 0.0, base_gamma
        
        # Adjust based on feedback reliability
        quality_score = (feedback.get('reasoning_score', 0.5) + 
                        feedback.get('solution_score', 0.5)) / 2.0
        
        # Higher feedback weight when quality assessment is more extreme (confident)
        feedback_confidence = abs(quality_score - 0.5) * 2  # Range: [0, 1]
        
        alpha = base_alpha
        beta = base_beta * feedback_confidence  # Only apply when Agent B is confident
        gamma = base_gamma
        
        return alpha, beta, gamma
    
    def _collect_agent_feedback(self, inputs, outputs):
        """
        Collect Agent B feedback with error handling and caching
        """
        try:
            # Only collect feedback periodically to avoid overhead
            if self.step_counter % 50 != 0:  # Every 50 steps
                return self._get_recent_feedback()
            
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)
            
            # Sample one example for efficiency
            if preds.size(0) > 0:
                pred_sample = preds[0:1].detach().cpu().tolist()
                lbl_sample = shift_labels[0:1].detach().cpu().tolist()
                
                # Decode
                lbl_dec = [[tok if tok != -100 else self.tokenizer.pad_token_id for tok in seq]
                          for seq in lbl_sample]
                gt_texts = self.tokenizer.batch_decode(lbl_dec, skip_special_tokens=True)
                pred_texts = self.tokenizer.batch_decode(pred_sample, skip_special_tokens=True)
                
                if gt_texts and pred_texts and gt_texts[0].strip():
                    feedback_response = eval.evaluate(gt_texts, pred_texts)
                    content = feedback_response["content"].replace('```json','').replace('```','')
                    data = json.loads(content)
                    
                    # Safely extract scores
                    feedback_data = {
                        'is_correct': data.get("is_correct", False),
                        'reasoning_score': self._safe_sigmoid(data.get("reasoning_score", 0.0)),
                        'solution_score': self._safe_sigmoid(data.get("solution_score", 0.0)),
                        'step': self.step_counter
                    }
                    
                    self.feedback_buffer.append(feedback_data)
                    self.feedback_history.append(feedback_data)
                    
                    return feedback_data
                    
        except Exception as e:
            print(f"Warning: Agent B feedback failed at step {self.step_counter}: {e}")
        
        return self._get_recent_feedback()
    
    def _safe_sigmoid(self, x):
        """Safe sigmoid transformation"""
        x = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-x))
    
    def _get_recent_feedback(self):
        """Get most recent feedback or return neutral"""
        if self.feedback_buffer:
            return self.feedback_buffer[-1]
        return None
    
    def _track_training_metrics(self, inputs, outputs, loss):
        """Track training metrics without affecting loss computation"""
        if "attention_mask" in inputs:
            num_tokens = inputs["attention_mask"].sum().item()
            self._total_train_tokens += num_tokens
            self._metrics["train"]["num_tokens"] = [self._total_train_tokens]
        
        if "labels" in inputs:
            token_accuracy = self._compute_token_accuracy(inputs, outputs)
            self._metrics["train"]["mean_token_accuracy"].append(token_accuracy)
    
    def _track_eval_metrics(self, inputs, outputs, loss):
        """Track evaluation metrics without any modifications"""
        if "labels" in inputs:
            token_accuracy = self._compute_token_accuracy(inputs, outputs)
            self._metrics["eval"]["mean_token_accuracy"].append(token_accuracy)
    
    def _compute_token_accuracy(self, inputs, outputs):
        """Compute token-level accuracy"""
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        preds = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100

        correct = (preds == shift_labels) & mask
        tot = mask.sum()
        corr = correct.sum()
        
        if hasattr(self, 'accelerator'):
            corr = self.accelerator.gather_for_metrics(corr)
            tot = self.accelerator.gather_for_metrics(tot)
            tot_sum = tot.sum()
        else:
            tot_sum = tot
            
        return (corr.sum() / tot_sum).item() if tot_sum > 0 else 0.0

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

        self.trainer = MathematicallyPrincipledSFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=1,
                warmup_steps=10,
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
        return self.trainer.feedback_history