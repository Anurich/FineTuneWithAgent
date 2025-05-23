from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import Trainer
from datasets import Dataset
import pandas as pd
from collections import defaultdict, deque
from Agent.evaluation_agent.evaluate import Evaluate
import torch
import torch.nn.functional as F
import numpy as np
import json
import math

eval = Evaluate()

class FinalCorrectedSFTTrainer(Trainer):
    def __init__(self, *args, processing_class=None, **kwargs):
        super().__init__(*args, processing_class=processing_class, **kwargs)
        self.tokenizer = self.processing_class
        
        # Core tracking
        self._total_train_tokens = 0
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.step_counter = 0
        
        # Feedback system designed for Agent B's z-score normalization
        self.feedback_cache = {}
        self.quality_history = deque(maxlen=50)
        self.current_batch_feedback = None
        
        # Z-score aware parameters
        self.z_score_clip_range = 3.0      # Clip z-scores to [-3, 3] for stability
        self.temp_base = 1.0               # Base temperature
        self.temp_range = 0.4              # Temperature adjustment range
        self.confidence_penalty_strength = 0.08  # Reduced for z-score stability
        self.gradient_scale_range = 0.15   # Reduced gradient scaling
        
        # Quality interpretation thresholds (in z-score space)
        self.z_score_high_threshold = 1.0   # z > 1 is "high quality"
        self.z_score_low_threshold = -1.0   # z < -1 is "low quality"
        
        # Stability and smoothing
        self.min_quality_samples = 5
        self.smoothing_factor = 0.6        # Stronger smoothing for z-scores
        self.quality_ema = 0.0             # Exponential moving average
        self.ema_initialized = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Loss computation accounting for Agent B's z-score normalization
        """
        # Check evaluation mode
        is_eval_mode = not model.training
        
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
        
        # Only apply feedback during training
        if is_eval_mode:
            self._track_eval_metrics(inputs, outputs, loss.detach())
            return (loss, outputs) if return_outputs else loss
        
        # Training mode: apply z-score aware feedback integration
        self.step_counter += 1
        modified_loss = self._apply_z_score_aware_feedback(loss, inputs, outputs)
        
        # Track metrics
        self._track_training_metrics(inputs, outputs, loss.detach())
        
        return (modified_loss, outputs) if return_outputs else modified_loss
    
    def _apply_z_score_aware_feedback(self, base_loss, inputs, outputs):
        """
        Apply feedback adjustments accounting for Agent B's z-score normalization
        """
        if "labels" not in inputs:
            return base_loss
            
        # Get current batch quality (z-scores from Agent B)
        current_quality = self._get_current_batch_quality(inputs, outputs)
        
        if current_quality is None:
            return base_loss
        
        # Extract z-scores and clip for stability
        raw_reasoning_z = current_quality.get('reasoning_score', 0.0)
        raw_solution_z = current_quality.get('solution_score', 0.0)
        
        # Clip z-scores to prevent extreme values
        reasoning_z = np.clip(raw_reasoning_z, -self.z_score_clip_range, self.z_score_clip_range)
        solution_z = np.clip(raw_solution_z, -self.z_score_clip_range, self.z_score_clip_range)
        
        # Convert z-scores to quality estimates using tanh (better for z-scores than sigmoid)
        # tanh maps [-inf, inf] to [-1, 1], then we shift to [0, 1]
        reasoning_quality = (math.tanh(reasoning_z / 2.0) + 1.0) / 2.0  # Divide by 2 for gentler mapping
        solution_quality = (math.tanh(solution_z / 2.0) + 1.0) / 2.0
        overall_quality = (reasoning_quality + solution_quality) / 2.0
        
        # Update quality history with EMA for better stability
        if not self.ema_initialized:
            self.quality_ema = overall_quality
            self.ema_initialized = True
        else:
            self.quality_ema = self.smoothing_factor * self.quality_ema + (1 - self.smoothing_factor) * overall_quality
        
        self.quality_history.append(overall_quality)
        
        # Don't apply adjustments until we have enough samples
        if len(self.quality_history) < self.min_quality_samples:
            return base_loss
        
        # Use EMA quality for more stable adjustments
        stable_quality = self.quality_ema
        
        # Apply the corrected components with z-score awareness
        temperature_adjusted_loss = self._apply_z_score_temperature_adjustment(
            base_loss, inputs, outputs, reasoning_z, solution_z, stable_quality
        )
        
        confidence_penalty = self._compute_z_score_confidence_penalty(
            inputs, outputs, reasoning_z, solution_z, stable_quality
        )
        
        # Combine with conservative scaling
        total_loss = temperature_adjusted_loss + confidence_penalty
        
        # Apply gradient scaling based on z-score magnitude
        gradient_scale = self._compute_z_score_gradient_scale(reasoning_z, solution_z)
        if abs(gradient_scale - 1.0) > 0.01:  # Only apply if significant
            total_loss = total_loss * gradient_scale
        
        return total_loss
    
    def _apply_z_score_temperature_adjustment(self, base_loss, inputs, outputs, reasoning_z, solution_z, quality):
        """
        Temperature adjustment based on z-scores
        """
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        mask = shift_labels != -100
        
        if not mask.any():
            return base_loss
        
        # Compute temperature based on z-scores
        # High z-score -> lower temperature (more confident)
        # Low z-score -> higher temperature (less confident)
        avg_z_score = (reasoning_z + solution_z) / 2.0
        
        if avg_z_score > self.z_score_high_threshold:
            # High quality (positive z-score): lower temperature
            temperature = self.temp_base - self.temp_range * min(1.0, avg_z_score / 2.0)
        elif avg_z_score < self.z_score_low_threshold:
            # Low quality (negative z-score): higher temperature
            temperature = self.temp_base + self.temp_range * min(1.0, abs(avg_z_score) / 2.0)
        else:
            # Medium quality: normal temperature
            temperature = self.temp_base
        
        # Ensure temperature is positive and reasonable
        temperature = max(0.5, min(2.0, temperature))
        
        # Apply temperature scaling
        scaled_logits = shift_logits / temperature
        
        # Compute loss with temperature-adjusted logits
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        valid_loss = nll_loss[mask]
        if len(valid_loss) == 0:
            return base_loss
            
        return valid_loss.mean()
    
    def _compute_z_score_confidence_penalty(self, inputs, outputs, reasoning_z, solution_z, quality):
        """
        Confidence penalty aware of z-score distribution
        """
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        mask = shift_labels != -100
        
        if not mask.any():
            return torch.tensor(0.0, device=shift_logits.device)
        
        # Compute model confidence
        probs = F.softmax(shift_logits, dim=-1)
        max_probs = probs.max(dim=-1)[0]
        avg_confidence = max_probs[mask].mean()
        
        # Compute target confidence based on z-scores
        avg_z_score = (reasoning_z + solution_z) / 2.0
        
        # Map z-score to target confidence more conservatively
        # z-score of 0 -> confidence 0.6
        # z-score of +2 -> confidence 0.8
        # z-score of -2 -> confidence 0.4
        target_confidence = 0.6 + 0.1 * math.tanh(avg_z_score)
        target_confidence = max(0.3, min(0.9, target_confidence))
        
        confidence_diff = avg_confidence.item() - target_confidence
        penalty = torch.tensor(0.0, device=shift_logits.device)
        
        # Apply penalty with z-score consideration
        if confidence_diff > 0.15:  # Overconfident
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            avg_entropy = entropy[mask].mean()
            # Stronger penalty for overconfidence when z-score is negative (low quality)
            penalty_strength = self.confidence_penalty_strength * (1.0 + max(0, -avg_z_score) / 2.0)
            penalty = -penalty_strength * confidence_diff * avg_entropy
            
        elif confidence_diff < -0.15:  # Underconfident
            # Only penalize underconfidence if z-score suggests high quality
            if avg_z_score > 0:
                penalty_strength = self.confidence_penalty_strength * (avg_z_score / 2.0)
                penalty = penalty_strength * abs(confidence_diff) * (1 - avg_confidence)
        
        return penalty
    
    def _compute_z_score_gradient_scale(self, reasoning_z, solution_z):
        """
        Gradient scaling based on z-score magnitude
        """
        avg_z_score = (reasoning_z + solution_z) / 2.0
        
        if avg_z_score > 1.5:
            # Very high quality: gentle gradients
            return 1.0 - self.gradient_scale_range * 0.7
        elif avg_z_score < -1.5:
            # Very low quality: strong gradients
            return 1.0 + self.gradient_scale_range * 1.2
        elif avg_z_score < -0.5:
            # Low quality: moderate boost
            return 1.0 + self.gradient_scale_range * 0.6
        else:
            # Normal quality: standard gradients
            return 1.0
    
    def _get_current_batch_quality(self, inputs, outputs):
        """
        Get quality assessment for current batch with improved error handling
        """
        try:
            # Collect feedback every 20 steps to balance overhead and freshness
            if self.step_counter % 20 != 0:
                return self._get_cached_quality()
            
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)
            
            if preds.size(0) > 0:
                # Get a sample from current batch
                sample_idx = min(torch.randint(0, preds.size(0), (1,)).item(), preds.size(0) - 1)
                pred_sample = preds[sample_idx:sample_idx+1].detach().cpu().tolist()
                lbl_sample = shift_labels[sample_idx:sample_idx+1].detach().cpu().tolist()
                
                # Decode with better error handling
                lbl_dec = []
                for seq in lbl_sample:
                    decoded_seq = []
                    for tok in seq:
                        if tok != -100:
                            decoded_seq.append(tok)
                        else:
                            decoded_seq.append(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0)
                    lbl_dec.append(decoded_seq)
                
                try:
                    gt_texts = self.tokenizer.batch_decode(lbl_dec, skip_special_tokens=True)
                    pred_texts = self.tokenizer.batch_decode(pred_sample, skip_special_tokens=True)
                except Exception as e:
                    print(f"Warning: Decoding failed at step {self.step_counter}: {e}")
                    return self._get_cached_quality()
                
                if gt_texts and pred_texts and gt_texts[0].strip():
                    # Get Agent B feedback (returns z-scores)
                    feedback_response = eval.evaluate(gt_texts, pred_texts)
                    content = feedback_response["content"].replace('```json','').replace('```','')
                    data = json.loads(content)
                    
                    # Extract z-scores directly (no sigmoid needed)
                    quality_data = {
                        'is_correct': data.get("is_correct", False),
                        'reasoning_score': float(data.get("reasoning_score", 0.0)),  # These are z-scores
                        'solution_score': float(data.get("solution_score", 0.0)),   # These are z-scores
                        'step': self.step_counter
                    }
                    
                    self.current_batch_feedback = quality_data
                    return quality_data
                    
        except Exception as e:
            print(f"Warning: Failed to get current batch quality at step {self.step_counter}: {e}")
        
        return self._get_cached_quality()
    
    def _get_cached_quality(self):
        """Get most recent quality assessment or neutral default"""
        if self.current_batch_feedback is not None:
            return self.current_batch_feedback
        
        # Return neutral z-scores (0.0) if no feedback available
        return {
            'is_correct': False,
            'reasoning_score': 0.0,  # Neutral z-score
            'solution_score': 0.0,   # Neutral z-score
            'step': self.step_counter
        }
    
    def _track_training_metrics(self, inputs, outputs, loss):
        """Track training metrics"""
        if "attention_mask" in inputs:
            num_tokens = inputs["attention_mask"].sum().item()
            self._total_train_tokens += num_tokens
            self._metrics["train"]["num_tokens"] = [self._total_train_tokens]
        
        if "labels" in inputs:
            token_accuracy = self._compute_token_accuracy(inputs, outputs)
            self._metrics["train"]["mean_token_accuracy"].append(token_accuracy)
    
    def _track_eval_metrics(self, inputs, outputs, loss):
        """Track evaluation metrics without modifications"""
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
        
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            corr = self.accelerator.gather_for_metrics(corr)
            tot = self.accelerator.gather_for_metrics(tot)
            tot_sum = tot.sum()
        else:
            tot_sum = tot
            
        return (corr.sum() / tot_sum).item() if tot_sum > 0 else 0.0

# MODEL class with the final corrected trainer
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

        # Use the final corrected trainer
        self.trainer = FinalCorrectedSFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            args=SFTConfig(
                dataset_text_field="text",
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                gradient_accumulation_steps=1,
                warmup_steps=15,  # Slightly longer warmup for stability
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
        return {
            'quality_history': list(self.trainer.quality_history),
            'quality_ema': self.trainer.quality_ema,
            'current_feedback': self.trainer.current_batch_feedback
        }