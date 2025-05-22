from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
import numpy as np
import json
class Evaluate:
    def __init__(self):
        llm = ChatOpenAI(model="gpt-4o-mini")
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
            stats["std"] = np.sqrt(stats["std"]**2 + (score - old_mean) * (score - stats["mean"]))

    def normalize_score(self, score, key):
        """Normalize a score using running mean and std."""
        stats = self.score_stats[key]
        if stats["std"] > 1e-8:
            return (score - stats["mean"]) / stats["std"]
        return score

    def evaluate(self, gt, pred):
        batch_size = len(gt)
        subset_size = 5
        evaluations = []
        max_retries = 2
        
        for i in range(0, batch_size, subset_size):
            gt_subset = gt[i:i+subset_size]
            pred_subset = pred[i:i+subset_size]
            if gt_subset:
                for attempt in range(max_retries + 1):
                    try:
                        output = self.chain.invoke({"ground_truth": gt_subset, "prediction": pred_subset})
                        feedback_content = output.content.replace('```json', '').replace('```', '')
                        feedback_data = json.loads(feedback_content)
                        self.update_stats(feedback_data["solution_score"], feedback_data["reasoning_score"])
                        feedback_data["solution_score"] = self.normalize_score(feedback_data["solution_score"], "solution_score")
                        feedback_data["reasoning_score"] = self.normalize_score(feedback_data["reasoning_score"], "reasoning_score")
                        evaluations.append(feedback_data)
                        break
                    except Exception as e:
                        if attempt == max_retries:
                            print(f"Warning: Failed to parse subset {i//subset_size} after {max_retries} retries: {e}")
                            evaluations.append({
                                "solution_score": self.normalize_score(0.5, "solution_score"),
                                "reasoning_score": self.normalize_score(0.5, "reasoning_score"),
                                "is_correct": False
                            })
        
        if evaluations:
            solution_scores = [e["solution_score"] for e in evaluations]
            reasoning_scores = [e["reasoning_score"] for e in evaluations]
            is_correct = any(e["is_correct"] for e in evaluations)
            avg_solution_score = sum(solution_scores) / len(solution_scores)
            avg_reasoning_score = sum(reasoning_scores) / len(reasoning_scores)
            return {
                "content": json.dumps({
                    "solution_score": avg_solution_score,
                    "reasoning_score": avg_reasoning_score,
                    "is_correct": is_correct
                })
            }
        else:
            return {
                "content": json.dumps({
                    "solution_score": self.normalize_score(0.5, "solution_score"),
                    "reasoning_score": self.normalize_score(0.5, "reasoning_score"),
                    "is_correct": False
                })
            }