from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import Trainer
from datasets import Dataset
from transformers import default_data_collator
import pandas as pd 
from collections import defaultdict
from Agent.evaluation_agent.evaluate import Evaluate
import torch
import numpy as np
import json
from collections import deque
eval = Evaluate()

class CustomSFTTrainer(Trainer):
    _total_train_tokens= 0
    _metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # Added num_items_in_batch
        mode = "eval" if self.control.should_evaluate else "train"
        
        if not hasattr(self, 'performance_history'):
            self.performance_history = deque(maxlen=20) # Maintained original size, good for holding history
        if not hasattr(self, 'feedback_momentum'):
            self.feedback_momentum = {'trend': 0.0, 'strength': 0.0}
        if not hasattr(self, 'step_counter'):
            self.step_counter = 0
        
        if isinstance(inputs, list):
            if len(inputs) > 0 and isinstance(inputs[0], dict):
                inputs = {k: torch.stack([item[k] for item in inputs]) if isinstance(inputs[0][k], torch.Tensor) else [item[k] for item in inputs] for k in inputs[0]}
            else:
                raise ValueError("Expected inputs to be a dictionary or a list of dictionaries")

        (loss, outputs) = super().compute_loss(model, inputs, return_outputs=True) # Removed num_items_in_batch if not used by super
        
        original_loss_for_meta_adjust = loss.detach().clone() # Loss from model directly

        if mode == "train":
            self.step_counter += 1
            # Token counting logic (unchanged)
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        if "labels" in inputs and not self.args.use_liger_kernel: # Assuming args is accessible
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            predictions = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100
            
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

            # ============ META-LEARNING FEEDBACK APPROACH ============
            # COMMENT: Per-item LLM evaluation in eval_instance.evaluate will be slower.
            # Consider increasing feedback_interval or reducing sample_size if training slows too much.
            feedback_interval = 50 
            if mode == "eval" or (mode == "train" and self.step_counter % feedback_interval == 0):
                # Sample size for feedback during training
                sample_size = min(max(10, len(predictions) // 10), 20) if mode == "train" else len(predictions)
                
                # Ensure sample_size does not exceed available predictions
                actual_sample_size = min(sample_size, predictions.size(0))

                preds_sample = predictions[:actual_sample_size].detach().cpu().tolist()
                labels_cpu_sample = shift_labels[:actual_sample_size].detach().cpu().tolist()
                
                labels_decoded = [
                    [tok if tok != -100 else self.tokenizer.pad_token_id for tok in seq]
                    for seq in labels_cpu_sample
                ]
                gt_texts = self.tokenizer.batch_decode(labels_decoded, skip_special_tokens=True)
                pred_texts = self.tokenizer.batch_decode(preds_sample, skip_special_tokens=True)
                
                try:
                    # Ensure gt_texts and pred_texts are not empty before calling evaluate
                    if gt_texts and pred_texts:
                        feedback_result = eval.evaluate(gt_texts, pred_texts) # Use the instantiated eval_instance
                        feedback_content = feedback_result["content"].replace('```json','').replace('```','')
                        feedback_data = json.loads(feedback_content)
                        
                        # Sigmoid on normalized scores from Evaluate (these are already z-scores)
                        # The scores from eval_instance are already normalized (z-scores).
                        # Sigmoid will map these z-scores to [0, 1].
                        solution_score_norm = feedback_data.get("solution_score", 0.0) # This is a z-score
                        reasoning_score_norm = feedback_data.get("reasoning_score", 0.0) # This is a z-score

                        solution_score_sigmoid = 1 / (1 + np.exp(-solution_score_norm))
                        reasoning_score_sigmoid = 1 / (1 + np.exp(-reasoning_score_norm))
                        
                        performance_score = {
                            'is_correct': feedback_data.get("is_correct", False), # Boolean
                            'reasoning_score': float(reasoning_score_sigmoid), # Sigmoid applied
                            'solution_score': float(solution_score_sigmoid),   # Sigmoid applied
                            'token_accuracy': accuracy,
                            'step': self.step_counter,
                            'mode': mode
                        }
                        self.performance_history.append(performance_score)
                        if len(self.performance_history) >= 3: # Need at least 3 for trend
                            self._update_feedback_momentum()
                    else:
                        print(f"Warning: Empty gt_texts or pred_texts at step {self.step_counter}. Skipping feedback.")
                        # Optionally append a neutral or default bad score
                        self.performance_history.append({
                            'is_correct': False, 'reasoning_score': 0.5, 'solution_score': 0.5, # Neutral sigmoid scores
                            'token_accuracy': accuracy, 'step': self.step_counter, 'mode': mode
                        })

                except Exception as e:
                    print(f"Warning: Failed to get {'training' if mode == 'train' else 'eval'} feedback: {e}")
                    self.performance_history.append({
                        'is_correct': False, 'reasoning_score': 0.5, 'solution_score': 0.5, # Neutral sigmoid scores
                        'token_accuracy': accuracy, 'step': self.step_counter, 'mode': mode
                    })
            
            if mode == "train":
                warmup_steps = 50
                min_history_for_trend = 3 # For _update_feedback_momentum
                
                # MODIFIED: base_lr_mult reduced
                base_lr_mult = 0.01 
                
                if len(self.performance_history) >= min_history_for_trend and self.step_counter > warmup_steps:
                    trend = self.feedback_momentum['trend']
                    strength = self.feedback_momentum['strength']
                    
                    if abs(trend) > 0.05 and strength > 0.1: # Adjusted thresholds slightly if needed
                        log_probs = torch.log_softmax(shift_logits, dim=-1)
                        batch_indices, seq_indices = torch.where(mask)
                        
                        if len(batch_indices) > 0:
                            token_logprobs = log_probs[batch_indices, seq_indices, shift_labels[batch_indices, seq_indices]]
                            mean_log_prob = token_logprobs.mean()
                            
                            probs = torch.softmax(shift_logits, dim=-1)
                            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1) # Increased epsilon slightly
                            mean_entropy = entropy[mask].mean()
                            
                            val_loss_trend = 0.0
                            if len(self._metrics["eval"].get("loss", [])) >= 2: # Safer access to eval loss
                                eval_losses = self._metrics["eval"]["loss"]
                                val_loss_trend = eval_losses[-1] - eval_losses[-2]
                            
                            # MODIFIED: adaptive_scale set to 1.0
                            adaptive_scale = 1.0
                            
                            adjustment = torch.tensor(0.0, device=loss.device)
                            # MODIFIED: Consistent L2 regularization
                            l2_reg_strength = 1e-5 
                            l2_reg = l2_reg_strength * torch.norm(shift_logits, p=2)

                            if trend > 0: # Performance improving
                                # Try to make model more confident / less uncertain, reduce loss
                                adjustment = -base_lr_mult * trend * strength * (-mean_log_prob) * adaptive_scale
                            else: # Performance declining or stagnant
                                # Try to encourage exploration / increase uncertainty (entropy), increase loss
                                adjustment = base_lr_mult * abs(trend) * strength * (1.0 / (mean_entropy + 1e-9)) * adaptive_scale
                            
                            loss_after_l2 = original_loss_for_meta_adjust + l2_reg
                            
                            # Clamp adjustment based on magnitude of loss after L2
                            max_adjustment = 0.1 * loss_after_l2.abs().item() 
                            actual_adjustment = torch.clamp(adjustment, -max_adjustment, max_adjustment)
                            
                            loss = loss_after_l2 + actual_adjustment # Apply L2 and then clamped meta-adjustment

                            # MODIFIED: Improved logging
                            if self.step_counter % 20 == 0: # Log more frequently if needed
                                print(f"Step {self.step_counter}: MetaLR Active. Trend: {trend:.4f}, Strength: {strength:.4f}, "
                                      f"ValLossTrend: {val_loss_trend:.4f}. Orig Loss: {original_loss_for_meta_adjust.item():.4f}, "
                                      f"L2: {l2_reg.item():.6f}, Req Adj: {adjustment.item():.6f}, "
                                      f"Clamped Adj: {actual_adjustment.item():.6f}, Final Loss: {loss.item():.4f}")
                        else: # No valid tokens in batch after masking for meta-adjustment, L2 still applies if logits exist
                            loss = original_loss_for_meta_adjust + l2_reg_strength * torch.norm(shift_logits, p=2)

                    else: # Trend/strength too low, or not enough history/warmup
                         # Apply only original loss if no meta-adjustment, or consider if L2 should always apply
                         # For consistency, if we calculate shift_logits, an L2 might be applied.
                         # However, the original code only added L2 within the meta-adjustment block.
                         # Let's ensure loss is what came from super() if no meta-adjustment.
                         loss = original_loss_for_meta_adjust # Revert to original if no meta-adjustment applied
        
        # Ensure 'loss' is the final value to be returned
        return (loss, outputs) if return_outputs else loss

    # MODIFIED: _update_feedback_momentum to use more history and clearer EWMA logic
    def _update_feedback_momentum(self):
        """Update performance trend and strength using EWMA."""
        # Use up to the last 10 scores from performance_history for EWMA
        history_for_ewma = list(self.performance_history)[-10:] # Slice last 10

        # Need at least 3 points to calculate a trend based on 2 EWMA values from 3 scores
        if len(history_for_ewma) < 3: 
            self.feedback_momentum['trend'] = 0.0
            self.feedback_momentum['strength'] = 0.0
            return

        recent_combined_scores = []
        for perf in history_for_ewma:
            r_score = perf.get('reasoning_score', 0.5) # Default to neutral (post-sigmoid)
            s_score = perf.get('solution_score', 0.5) # Default to neutral (post-sigmoid)
            combined_score = (r_score + s_score) / 2.0
            recent_combined_scores.append(combined_score)

        alpha = 0.3 # Smoothing factor for EWMA
        
        ewma_values = []
        if not recent_combined_scores: # Should be caught by len(history_for_ewma) < 3
            self.feedback_momentum['trend'] = 0.0
            self.feedback_momentum['strength'] = 0.0
            return
            
        current_ewma = recent_combined_scores[0]
        ewma_values.append(current_ewma)
        for score in recent_combined_scores[1:]:
            current_ewma = alpha * score + (1 - alpha) * current_ewma
            ewma_values.append(current_ewma)
        
        # Trend is the difference between the last two EWMA values
        trend = ewma_values[-1] - ewma_values[-2] # Requires at least 2 EWMA values (from 2 scores)
                                                # Handled by len(history_for_ewma) < 3 check
        
        score_var = np.var(recent_combined_scores)
        # Add epsilon to prevent division by zero and ensure strength is slightly less than 1 if variance is zero
        strength = 1.0 / (1.0 + score_var + 1e-6) 
        
        self.feedback_momentum['trend'] = float(trend)
        self.feedback_momentum['strength'] = float(strength)


class MODEL:
    def __init__(self, train_data, test_data):
        self.max_seq_length = 2048
        self.lora_rank = 32
        self.train_data = train_data
        self.test_data = test_data
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Qwen3-4B-Base",
            max_seq_length = self.max_seq_length,
            load_in_4bit = False,
            fast_inference = True,
            max_lora_rank = self.lora_rank,
            gpu_memory_utilization = 0.7,
        )
        
        self.model = FastLanguageModel.get_peft_model(
                        self.model,
                        r = self.lora_rank,
                        target_modules = [
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                        ],
                        lora_alpha = self.lora_rank*2,
                        use_gradient_checkpointing = "unsloth",
                        random_state = 3407,
                    )
        
        self.reasoning_start = "<start_working_out>"
        self.reasoning_end   = "<end_working_out>"
        self.solution_start  = "<SOLUTION>"
        self.solution_end    = "</SOLUTION>"

        self.system_prompt = \
        f"""You are given a problem.
        Think about the problem and provide your working out.
        Place it between {self.reasoning_start} and {self.reasoning_end}.
        Then, provide your solution between {self.solution_start}{self.solution_end}"""
        
        chat_template = \
            "{% if messages[0]['role'] == 'system' %}"\
                "{{ messages[0]['content'] + eos_token }}"\
                "{% set loop_messages = messages[1:] %}"\
            "{% else %}"\
                "{{ '{system_prompt}' + eos_token }}"\
                "{% set loop_messages = messages %}"\
            "{% endif %}"\
            "{% for message in loop_messages %}"\
                "{% if message['role'] == 'user' %}"\
                    "{{ message['content'] }}"\
                "{% elif message['role'] == 'assistant' %}"\
                    "{{ message['content'] + eos_token }}"\
                "{% endif %}"\
            "{% endfor %}"\
            "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
            "{% endif %}"

        chat_template = chat_template\
            .replace("'{system_prompt}'",   f"'{self.system_prompt}'")\
            .replace("'{reasoning_start}'", f"'{self.reasoning_start}'")
        self.tokenizer.chat_template = chat_template
    
    def format_dataset(self,x):
        expected_answer = str(x["answer"])
        problem = str(x["question"])
        thoughts = str(x["reasoning"])
        thoughts = thoughts.strip()
        final_prompt = \
            self.reasoning_start + thoughts + self.reasoning_end + \
            self.solution_start + expected_answer + self.solution_end
        return [
            {"role" : "system",    "content" : self.system_prompt},
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : final_prompt},
        ]
    
    def data_collator(self,batch):
        # This custom collator seems to be simplified.
        # SFTTrainer usually handles tokenization if dataset_text_field is used.
        # If this collator is indeed used, ensure it matches trainer expectations.
        # The original code uses SFTConfig(dataset_text_field = "text",...)
        # which means SFTTrainer would by default use its internal collator that
        # tokenizes the "text" field.
        # If you intend to use *this* collator, you might need to pass it explicitly
        # and ensure the data is pre-formatted with a "text" field or adjust.
        # For now, assuming the SFTConfig takes precedence or this is correctly aligned.
        
        # The SFTTrainer is passed `dataset_text_field = "text"`, and this collator is also passed.
        # This collator is re-tokenizing. It should be okay if `batch` contains items with "text" field.
        texts_to_tokenize = [example["text"] for example in batch]

        tokenized_inputs = self.tokenizer(
            texts_to_tokenize, # Changed from [example["text"] for example in batch]
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        return tokenized_inputs
    
    def _train(self):
        self.train_data = pd.DataFrame(self.train_data)
        self.test_data  = pd.DataFrame(self.test_data)
        
        self.train_data["Messages"] = self.train_data.apply(self.format_dataset, axis=1)
        self.test_data["Messages"] = self.test_data.apply(self.format_dataset, axis=1)
        
        self.train_data["text"] = self.tokenizer.apply_chat_template(self.train_data["Messages"].values.tolist(), tokenize = False)
        self.test_data["text"] = self.tokenizer.apply_chat_template(self.test_data["Messages"].values.tolist(), tokenize = False)

        self.train_dataset = Dataset.from_pandas(self.train_data)
        self.test_dataset  = Dataset.from_pandas(self.test_data)
        
        # Make sure CustomSFTTrainer is used
        self.trainer = CustomSFTTrainer( 
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.train_dataset,
            eval_dataset = self.test_dataset,
            args = SFTConfig(
                dataset_text_field = "text", # This tells SFTTrainer to look for "text" column
                per_device_train_batch_size = 4,
                per_device_eval_batch_size = 4,
                gradient_accumulation_steps = 1,
                warmup_steps = 5,
                do_eval=True,
                num_train_epochs = 50,
                learning_rate = 2e-4,
                logging_steps = 200, # This is for Hugging Face logs (separate from custom print)
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                eval_steps=200,
                evaluation_strategy="steps", # Changed from eval_strategy to evaluation_strategy
                seed = 3407,
                report_to = "none",
                remove_unused_columns=False 
            ),
            data_collator=self.data_collator, # Pass the custom data collator
        )
        self.trainer.train()
        
    def _prediction(self):
        # This returns the history collected by CustomSFTTrainer
        return self.trainer.performance_history