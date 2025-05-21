from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

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
            1. Solution score (0-1): How correct is the solution's final answer?
            2. Reasoning score (0-1): How well-structured and logical is the reasoning process?
            3. Is the solution correct overall? (true/false)

            Return your evaluation as a JSON object with the following format:
            {{
            "solution_score": <score from 0-1>,
            "reasoning_score": <score from 0-1>,
            "is_correct": <true or false>
            }}

        """
        
        self.template = PromptTemplate.from_template(self.prompt)
        self.chain = self.template | llm

    def evaluate(self, gt, pred):
        output = self.chain.invoke({"ground_truth": gt, "prediction": pred})        
        return output
        