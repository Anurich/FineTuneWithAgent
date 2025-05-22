from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
import numpy as np
import json
class Evaluate:
    def __init__(self):
        llm = ChatOpenAI(model="gpt-4.1-mini") # Ensure API key is configured via .env or environment variables
        self.prompt = """
        Please evaluate the following student solution against the ground truth solution.
            Ground Truth Solution:
            {ground_truth}

            Student Solution:
            {prediction}

            Evaluate the student solution on the following criteria:
            1. Solution score (0-1): How correct is the final answer?
               - 1.0: Exact match or equivalent to ground truth.
               - 0.5–0.9: Partially correct (e.g., correct approach but minor errors).
               - 0.0–0.4: Incorrect or significantly deviates from ground truth.
            2. Reasoning score (0-1): How well-structured and logical is the reasoning process?
               - 1.0: Clear, logical, complete reasoning with all necessary steps.
               - 0.5–0.9: Mostly clear but with minor gaps or unclear steps.
               - 0.0–0.4: Confusing, incomplete, or illogical reasoning.
            3. Is the solution correct overall? (true/false): True if the final answer is fully correct, false otherwise.

            Example:
            Ground Truth: "Solve 2x + 3 = 7. Solution: 2x = 4, x = 2."
            Student Solution: "2x + 3 = 7, 2x = 4, x = 2."
            Evaluation: {{"solution_score": 1.0, "reasoning_score": 1.0, "is_correct": true}}

            Ground Truth: "Solve 2x + 3 = 7. Solution: 2x = 4, x = 2."
            Student Solution: "2x + 3 = 7, 2x = 5, x = 2.5."
            Evaluation: {{"solution_score": 0.3, "reasoning_score": 0.5, "is_correct": false}}

            Return your evaluation as a JSON object:
            {{
            "solution_score": <score from 0-1>,
            "reasoning_score": <score from 0-1>,
            "is_correct": <true or false>
            }}
            
            **DO NOT PROVIDE ANY OTHER EXPLANATION, JUST RESPOND WITH JSON FORMAT SHOWN ABOVE**
        """
        self.template = PromptTemplate.from_template(self.prompt)
        self.chain = self.template | llm
        self.score_stats = {
            "solution_score": {"mean": 0.5, "std": 1.0, "count": 0},
            "reasoning_score": {"mean": 0.5, "std": 1.0, "count": 0}
        }

    def update_stats(self, solution_score, reasoning_score):
        """Update running mean and std for score normalization."""
        for score, key in [(solution_score, "solution_score"), (reasoning_score, "reasoning_score")]:
            stats = self.score_stats[key]
            stats["count"] += 1
            old_mean = stats["mean"]
            stats["mean"] += (score - old_mean) / stats["count"]
            # Corrected std calculation: (old_sum_sq - 2*old_mean*old_sum + count*old_mean^2 + ...), or Welford's
            # For simplicity, using a common approximation, but can be numerically unstable.
            # A more robust way: M_k = M_{k-1} + (x_k - M_{k-1})/k; S_k = S_{k-1} + (x_k - M_{k-1})*(x_k - M_k)
            # For now, keeping original intent if it worked, but be aware.
            # The original formula for std update in the prompt was also incorrect:
            # stats["std"] = np.sqrt(stats["std"]**2 + (score - old_mean) * (score - stats["mean"])) is not a correct running std.
            # Using a simplified approach for now, assuming scores are somewhat bounded.
            # Reverting to a simpler variance update or Welford's algorithm is recommended for numerical stability.
            # For this fix, let's assume the user will refine running std if needed.
            # The original provided code had this std update, keeping it for now.
            if stats["count"] > 1: # Std requires at least 2 points
                 stats["std"] = np.sqrt( ((stats["count"]-2)/(stats["count"]-1))*(stats["std"]**2) + ((score - old_mean)**2)/stats["count"] ) if stats["count"] > 1 else 1.0
            else:
                stats["std"] = 1.0


    def normalize_score(self, score, key):
        """Normalize a score using running mean and std."""
        stats = self.score_stats[key]
        if stats["std"] > 1e-8 and stats["count"] > 1 : # Avoid division by zero and use std if count > 1
            return (score - stats["mean"]) / stats["std"]
        return score - stats["mean"] # If std is too small or count is 1, just center it.

    # MODIFIED: Per-item LLM evaluation
    def evaluate(self, gt_texts, pred_texts):
        if not isinstance(gt_texts, list): gt_texts = [gt_texts]
        if not isinstance(pred_texts, list): pred_texts = [pred_texts]

        if not gt_texts or not pred_texts : # Handle empty inputs
             return {
                "content": json.dumps({
                    "solution_score": self.normalize_score(0.0, "solution_score"),
                    "reasoning_score": self.normalize_score(0.0, "reasoning_score"),
                    "is_correct": False,
                    "avg_correctness_score": 0.0
                })
            }

        if len(gt_texts) != len(pred_texts):
            print(f"Warning: Mismatch in len of ground truth ({len(gt_texts)}) and predictions ({len(pred_texts)}). Skipping evaluation for this batch.")
            return {
                "content": json.dumps({
                    "solution_score": self.normalize_score(0.0, "solution_score"), 
                    "reasoning_score": self.normalize_score(0.0, "reasoning_score"),
                    "is_correct": False,
                    "avg_correctness_score": 0.0
                })
            }

        individual_evaluations = []
        max_retries = 2

        for gt_item, pred_item in zip(gt_texts, pred_texts):
            if not gt_item.strip() or not pred_item.strip(): # Skip if either GT or Pred is empty/whitespace
                individual_evaluations.append({
                    "solution_score": self.normalize_score(0.0, "solution_score"),
                    "reasoning_score": self.normalize_score(0.0, "reasoning_score"),
                    "is_correct": False
                })
                continue

            for attempt in range(max_retries + 1):
                try:
                    output = self.chain.invoke({"ground_truth": gt_item, "prediction": pred_item})
                    feedback_content = output.content.replace('```json', '').replace('```', '').strip()
                    feedback_data = json.loads(feedback_content)
                    
                    # Update stats with raw scores from LLM for this item
                    raw_sol_score = float(feedback_data.get("solution_score", 0.0))
                    raw_reas_score = float(feedback_data.get("reasoning_score", 0.0))
                    self.update_stats(raw_sol_score, raw_reas_score)
                    
                    item_eval = {
                        "solution_score": self.normalize_score(raw_sol_score, "solution_score"),
                        "reasoning_score": self.normalize_score(raw_reas_score, "reasoning_score"),
                        "is_correct": feedback_data.get("is_correct", False) 
                    }
                    individual_evaluations.append(item_eval)
                    break 
                except Exception as e:
                    if attempt == max_retries:
                        print(f"Warning: Failed to parse evaluation for item after {max_retries} retries: {e}. GT: '{gt_item[:50]}...', Pred: '{pred_item[:50]}...'")
                        individual_evaluations.append({
                            "solution_score": self.normalize_score(0.5, "solution_score"), # Default, normalized
                            "reasoning_score": self.normalize_score(0.5, "reasoning_score"),# Default, normalized
                            "is_correct": False
                        })
        
        if individual_evaluations:
            solution_scores = [e["solution_score"] for e in individual_evaluations]
            reasoning_scores = [e["reasoning_score"] for e in individual_evaluations]
            is_correct_flags = [1.0 if e.get("is_correct", False) else 0.0 for e in individual_evaluations]
            
            avg_solution_score = sum(solution_scores) / len(solution_scores) if solution_scores else 0.0
            avg_reasoning_score = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0
            avg_correctness_metric = sum(is_correct_flags) / len(is_correct_flags) if is_correct_flags else 0.0
            
            # MODIFIED: is_correct for the batch based on average correctness
            overall_is_correct_for_batch = avg_correctness_metric >= 0.5 

            return {
                "content": json.dumps({
                    "solution_score": avg_solution_score,
                    "reasoning_score": avg_reasoning_score,
                    "is_correct": overall_is_correct_for_batch,
                    "avg_correctness_score": avg_correctness_metric # NEW: detailed average
                })
            }
        else: # Should ideally be caught by earlier checks if gt_texts/pred_texts were empty
            return {
                "content": json.dumps({
                    "solution_score": self.normalize_score(0.0, "solution_score"),
                    "reasoning_score": self.normalize_score(0.0, "reasoning_score"),
                    "is_correct": False,
                    "avg_correctness_score": 0.0
                })
            }