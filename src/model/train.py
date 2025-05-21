from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import Trainer
from datasets import Dataset
import pandas as pd 
import torch
class CustomSFTTrainer(Trainer):
    def __init__(self, *args, reward, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies,
        and augment the loss based on external feedback (solution_score, reasoning_score, is_correct).
        """
        mode = "eval" if self.control.should_evaluate else "train"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

            # ------------------- FEEDBACK-AUGMENTED LOSS STARTS HERE -------------------

            if mode == "train" and all(k in inputs for k in ["solution_score", "reasoning_score", "is_correct"]):
                # Get feedback tensors
                solution_score = inputs["solution_score"].float()  # [batch]
                reasoning_score = inputs["reasoning_score"].float()  # [batch]
                is_correct = inputs["is_correct"].bool()  # [batch]
                feedback_score = (solution_score + reasoning_score) / 2  # [batch]

                # For each sample, gather mean log-prob of correct tokens
                log_probs = torch.log_softmax(shift_logits, dim=-1)  # [batch, seq, vocab]
                batch_indices, seq_indices = torch.where(mask)
                token_logprobs = log_probs[batch_indices, seq_indices, shift_labels[batch_indices, seq_indices]]  # [num_valid_tokens]

                # Sum and average logprobs per sample (handle variable lengths)
                num_samples = shift_labels.size(0)
                sample_token_counts = mask.sum(dim=1)  # [batch]
                sample_token_counts[sample_token_counts == 0] = 1  # avoid division by zero
                sum_logprobs_per_sample = torch.zeros(num_samples, device=shift_logits.device)
                sum_logprobs_per_sample.index_add_(0, batch_indices, token_logprobs)
                mean_logprob_per_sample = sum_logprobs_per_sample / sample_token_counts  # [batch]

                # Feedback reward loss
                lambda_fb = 1.0  # adjust as needed
                reward_loss = -lambda_fb * feedback_score * mean_logprob_per_sample  # [batch]

                # Penalty for incorrect samples, encourage uncertainty on bad feedback
                softmax_probs = torch.softmax(shift_logits, dim=-1)  # [batch, seq, vocab]
                max_probs, _ = softmax_probs.max(dim=-1)  # [batch, seq]
                max_probs_per_sample = (max_probs * mask.float()).sum(dim=1) / sample_token_counts  # [batch]
                penalty_loss = lambda_fb * (1 - feedback_score) * max_probs_per_sample * (~is_correct)  # [batch]

                # If you want to only use penalty for incorrect, otherwise penalty_loss = 0

                # Compose total loss per sample (main loss is already reduction='mean', so we need to re-scale)
                total_feedback_loss = (reward_loss + penalty_loss).mean()
                loss = loss + total_feedback_loss
            # ------------------- FEEDBACK-AUGMENTED LOSS ENDS HERE -------------------

        return (loss, outputs) if return_outputs else loss


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
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 1, # Use GA to mimic batch size!
                warmup_steps = 5,
                num_train_epochs = 2, # Set this for 1 full training run.
                learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
                logging_steps = 5,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "none", # Use this for WandB etc
            ),
        )
        
        
        self.trainer.train()
        
    
    def _prediction(self):
        pass
        