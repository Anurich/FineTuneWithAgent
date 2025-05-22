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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Meta-learning inspired feedback loss: Use test performance trends to adjust training dynamics.
        """
        mode = "eval" if self.control.should_evaluate else "train"
        
        # Initialize feedback tracking
        if not hasattr(self, 'performance_history'):
            self.performance_history = deque(maxlen=20)  # Keep last 20 evaluations
        if not hasattr(self, 'feedback_momentum'):
            self.feedback_momentum = {'trend': 0.0, 'strength': 0.0}
        if not hasattr(self, 'step_counter'):
            self.step_counter = 0
        
        if isinstance(inputs, list):
            if len(inputs) > 0 and isinstance(inputs[0], dict):
                inputs = {k: torch.stack([item[k] for item in inputs]) if isinstance(inputs[0][k], torch.Tensor) else [item[k] for item in inputs] for k in inputs[0]}
            else:
                raise ValueError("Expected inputs to be a dictionary or a list of dictionaries")

        # Compute base loss
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        if mode == "train":
            self.step_counter += 1
            # Standard token counting
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Token accuracy computation
        if "labels" in inputs and not self.args.use_liger_kernel:
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
            
            if mode == "eval":
                # Get batch-level feedback from Agent B (single call)
                preds = predictions.detach().cpu().tolist()
                labels_cpu = shift_labels.detach().cpu().tolist()
                labels_decoded = [
                    [tok if tok != -100 else self.tokenizer.pad_token_id for tok in seq]
                    for seq in labels_cpu
                ]
                
                gt_texts = self.tokenizer.batch_decode(labels_decoded, skip_special_tokens=True)
                pred_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
                
                try:
                    # Single batch call to Agent B
                    feedback_result = eval.evaluate(gt_texts, pred_texts)
                    feedback_content = feedback_result.content.replace('```json','').replace('```','')
                    feedback_data = json.loads(feedback_content)
                    
                    # Extract performance metrics
                    performance_score = {
                        'is_correct': feedback_data.get("is_correct", False),
                        'reasoning_score': float(feedback_data.get("reasoning_score", 0.5)),
                        'solution_score': float(feedback_data.get("solution_score", 0.5)),
                        'token_accuracy': accuracy,
                        'step': self.step_counter
                    }
                    
                    # Add to performance history
                    self.performance_history.append(performance_score)
                    
                    # Compute performance trend if we have enough history
                    if len(self.performance_history) >= 3:
                        self._update_feedback_momentum()
                    
                except Exception as e:
                    print(f"Warning: Failed to get feedback: {e}")
                    # Add neutral performance record
                    self.performance_history.append({
                        'is_correct': False,
                        'reasoning_score': 0.5,
                        'solution_score': 0.5,
                        'token_accuracy': accuracy,
                        'step': self.step_counter
                    })
            
            elif mode == "train":
                # Apply meta-learning based adjustments
                warmup_steps = 50
                min_history = 3
                
                if len(self.performance_history) >= min_history and self.step_counter > warmup_steps:
                    # Get current performance trend
                    trend = self.feedback_momentum['trend']
                    strength = self.feedback_momentum['strength']
                    
                    # Only apply if trend is significant
                    if abs(trend) > 0.1 and strength > 0.2:
                        # Compute gradient-based adjustment
                        log_probs = torch.log_softmax(shift_logits, dim=-1)
                        
                        # Get per-token log probabilities
                        batch_indices, seq_indices = torch.where(mask)
                        if len(batch_indices) > 0:
                            token_logprobs = log_probs[batch_indices, seq_indices, shift_labels[batch_indices, seq_indices]]
                            mean_log_prob = token_logprobs.mean()
                            
                            # Compute confidence entropy (uncertainty measure)
                            probs = torch.softmax(shift_logits, dim=-1)
                            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                            mean_entropy = entropy[mask].mean()
                            
                            # Adaptive learning rate based on performance trend
                            base_lr_mult = 0.01
                            
                            if trend > 0:  # Performance improving
                                # Encourage current learning direction (reduce loss slightly)
                                adjustment = -base_lr_mult * trend * strength * (-mean_log_prob)
                            else:  # Performance degrading
                                # Add regularization to prevent overconfidence
                                adjustment = base_lr_mult * abs(trend) * strength * (1.0 / (mean_entropy + 1e-8))
                            
                            # Safety clipping
                            max_adjustment = 0.05 * loss.abs()
                            adjustment = torch.clamp(adjustment, -max_adjustment, max_adjustment)
                            
                            loss = loss + adjustment
                            
                            # Debug logging
                            if self.step_counter % 100 == 0:
                                print(f"Step {self.step_counter}: Trend: {trend:.3f}, Strength: {strength:.3f}, "
                                    f"Adjustment: {adjustment.item():.5f}, Base Loss: {loss.item():.4f}")

        return (loss, outputs) if return_outputs else loss

    def _update_feedback_momentum(self):
        """Update performance trend and strength based on recent history."""
        if len(self.performance_history) < 3:
            return
        
        # Extract recent performance scores
        recent_scores = []
        for perf in list(self.performance_history)[-5:]:  # Last 5 evaluations
            combined_score = (perf['reasoning_score'] + perf['solution_score']) / 2
            recent_scores.append(combined_score)
        
        # Compute trend using linear regression
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        if len(recent_scores) > 1:
            # Simple linear trend
            trend = np.polyfit(x, y, 1)[0]  # Slope
            
            # Trend strength based on R-squared
            y_pred = np.polyval([trend, y[0]], x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))
            
            self.feedback_momentum['trend'] = float(trend)
            self.feedback_momentum['strength'] = float(max(0, r_squared))
        else:
            self.feedback_momentum['trend'] = 0.0
            self.feedback_momentum['strength'] = 0.0


class MODEL:
    def __init__(self, train_data, test_data):
        self.max_seq_length = 2048 # Can increase for longer reasoning traces
        self.lora_rank = 32 # Larger rank = smarter, but slower
        self.train_data = train_data
        self.test_data = test_data
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Qwen3-4B-Base",
            max_seq_length = self.max_seq_length,
            load_in_4bit = False, # False for LoRA 16bit
            fast_inference = True, # Enable vLLM fast inference
            max_lora_rank = self.lora_rank,
            gpu_memory_utilization = 0.7, # Reduce if out of memory
        )
        
        self.model = FastLanguageModel.get_peft_model(
                        self.model,
                        r = self.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                        target_modules = [
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                        ],
                        lora_alpha = self.lora_rank*2, # *2 speeds up training
                        use_gradient_checkpointing = "unsloth", # Reduces memory usage
                        random_state = 3407,
                    )
        
        self.reasoning_start = "<start_working_out>" # Acts as <think>
        self.reasoning_end   = "<end_working_out>"   # Acts as </think>
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

        # Replace with out specific template:
        chat_template = chat_template\
            .replace("'{system_prompt}'",   f"'{self.system_prompt}'")\
            .replace("'{reasoning_start}'", f"'{self.reasoning_start}'")
        self.tokenizer.chat_template = chat_template
    
    
    
    def format_dataset(self,x):
        expected_answer = x["answer"]
        problem = x["question"]
        # Remove generated <think> and </think>
        thoughts = x["reasoning"]
        # Strip newlines on left and right
        thoughts = thoughts.strip()
        # Add our custom formatting
        final_prompt = \
            self.reasoning_start + thoughts + self.reasoning_end + \
            self.solution_start + expected_answer + self.solution_end
        return [
            {"role" : "system",    "content" : self.system_prompt},
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : final_prompt},
        ]

    
    def data_collator(self,batch):
        tokenized_inputs = self.tokenizer(
            [example["text"] for example in batch],
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
        
        self.trainer = CustomSFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.train_dataset,
            eval_dataset = self.test_dataset,
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 4,
                per_device_eval_batch_size = 4,
                gradient_accumulation_steps = 1, # Use GA to mimic batch size!
                warmup_steps = 5,
                do_eval=True,
                num_train_epochs = 50, # Set this for 1 full training run.
                learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
                logging_steps = 200,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                eval_steps=200,
                eval_strategy="steps",
                seed = 3407,
                report_to = "none", # Use this for WandB etc
                remove_unused_columns=False
            ),
            
            data_collator=self.data_collator,
            
        )
        
        
        self.trainer.train()
        
    
    def _prediction(self):
        pass
        