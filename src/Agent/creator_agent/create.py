from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.output_parsers import JsonOutputToolsParser
import json
import hashlib
from typing import List, Dict, Set
load_dotenv()

class Agent_A:
    def __init__(self, total_number_of_data: int = 100, difficulty: int = 0, domain="math"):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.domain = domain 
        self.total_number_data = total_number_of_data
        self.difficulty = difficulty
        
        # Track generated questions across multiple calls
        self.generated_questions = []  # Store all generated questions
        self.question_hashes = set()   # For fast duplicate checking
        self.call_count = 0           # Track number of calls
        
        self.prompt_template = PromptTemplate(
                input_variables=["difficulty", "domain", "total_number_data_samples", "previous_questions_context", "call_number"],
                template="""You are an expert at generating question-reasoning-answer triplets in JSON format.
                Your task is to generate exactly {total_number_data_samples} unique triplets for training a language model, strictly adhering to the specified difficulty and uniqueness requirements.

                Current difficulty level: {difficulty}/10
                Domain: {domain}
                Required number of samples: {total_number_data_samples}
                Call number: {call_number}

                ---
                IMPORTANT INSTRUCTIONS:
                1.  **EXACT QUANTITY**: You MUST generate exactly {total_number_data_samples} triplets. No more, no less.
                2.  **STRICT JSON FORMAT**: Your entire response MUST be a single, valid JSON array. Do NOT include any introductory text, concluding remarks, or markdown code block delimiters (```json or ```). Start directly with `[` and end with `]`.
                3.  **UNIQUENESS (CALL #{call_number})**:
                    * Generate completely NEW questions different from all previously generated ones.
                    * Explore diverse subtopics and areas within the "{domain}" domain.
                    * Employ various problem types, contexts, and approaches.
                    * Avoid any patterns, structures, or exact phrasing similar to previous questions.
                    * Each question should test distinct aspects of "{domain}" knowledge.
                4.  **DIFFICULTY (LEVEL {difficulty}/10)**:
                    * **Level 1-2**: Basic concepts, single-step problems, direct application of formulas.
                    * **Level 3-4**: Multi-step problems requiring 2-3 reasoning steps, basic concept combinations.
                    * **Level 5-6**: Complex problems requiring 4-5 reasoning steps, intermediate concept integration.
                    * **Level 7-8**: Advanced problems requiring 6+ reasoning steps, multiple concept synthesis, edge cases.
                    * **Level 9-10**: Expert-level problems requiring deep reasoning, multiple advanced concepts, creative problem-solving.
                    Ensure questions are appropriately challenging and require the complexity described for level {difficulty}. Do not generate questions simpler than level {difficulty}.

                ---
                PREVIOUS QUESTIONS CONTEXT (for uniqueness guidance):
                {previous_questions_context}

                ---
                OUTPUT FORMAT:
                Each triplet must be an object with the following keys:
                -   `"question"`: A clear, unambiguous question matching difficulty level {difficulty} and unique.
                -   `"reasoning"`: Step-by-step reasoning with complexity appropriate for level {difficulty}.
                -   `"answer"`: The correct final answer.

                Begin your JSON array now:
                [
                    {{
                        "question": "A completely unique question not similar to any previous ones and matching difficulty {difficulty}",
                        "reasoning": "The step-by-step reasoning (with complexity matching level {difficulty})",
                        "answer": "The final answer"
                    }},
                    {{
                        "question": "Another unique question exploring different aspects of {domain} and matching difficulty {difficulty}",
                        "reasoning": "The step-by-step reasoning (with complexity matching level {difficulty})",
                        "answer": "The final answer"
                    }}
                    // ... continue until you have exactly {total_number_data_samples} triplets
                ]"""
        )
        self.chain = self.prompt_template | self.llm 
    
    def _get_question_hash(self, question: str) -> str:
        """Generate a hash for the question to check for duplicates"""
        # Normalize the question text (lowercase, strip whitespace)
        normalized = question.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _get_previous_questions_context(self, max_examples: int = 15) -> str:
        """Generate context about previously generated questions"""
        if not self.generated_questions:
            return "FIRST CALL: No previous questions generated yet. You can explore any area of the domain."
        
        # Get recent examples to show what to avoid
        recent_questions = self.generated_questions[-max_examples:]
        
        context = f"PREVIOUS QUESTIONS TO AVOID (from {len(self.generated_questions)} total generated):\n"
        context += "Examples of already generated questions (DO NOT create similar ones):\n"
        
        for i, q in enumerate(recent_questions, 1):
            context += f"{i}. {q['question'][:100]}{'...' if len(q['question']) > 100 else ''}\n"
        
        context += f"\nTotal questions generated so far: {len(self.generated_questions)}\n"
        context += f"Explore completely different areas of {self.domain} not covered above.\n"
        
        return context
    
    def _filter_duplicates(self, new_questions: List[Dict]) -> List[Dict]:
        """Remove duplicate questions from the new batch"""
        unique_questions = []
        
        for question_data in new_questions:
            question_text = question_data.get('question', '')
            question_hash = self._get_question_hash(question_text)
            
            if question_hash not in self.question_hashes:
                unique_questions.append(question_data)
                self.question_hashes.add(question_hash)
            else:
                print(f"Filtered duplicate question: {question_text[:80]}...")
        
        return unique_questions
    
    def _parse_response(self, response_content: str) -> List[Dict]:
        """Parse the LLM response and extract questions"""
        try:
            # Try to parse as JSON directly
            if response_content.strip().startswith('['):
                return json.loads(response_content)
            
            # Extract JSON from response if it's wrapped in other text
            start_idx = response_content.find('[')
            end_idx = response_content.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_content[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON array found in response")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response content: {response_content[:500]}...")
            return []
    
    def _create_data(self) -> List[Dict]:
        """Create data ensuring no duplicates across multiple calls"""
        self.call_count += 1
        print(f"Creating data batch #{self.call_count}...")
        
        # Get context about previous questions
        previous_context = self._get_previous_questions_context()
        
        max_attempts = 3
        attempt = 0
        unique_questions = []
        
        while len(unique_questions) < self.total_number_data and attempt < max_attempts:
            attempt += 1
            print(f"  Attempt {attempt}/{max_attempts}")
            
            # Adjust the number of samples to request based on what we still need
            samples_needed = self.total_number_data - len(unique_questions)
            # Request a few extra to account for potential duplicates
            samples_to_request = min(samples_needed + 10, self.total_number_data)
            
            try:
                # Generate new questions
                response = self.chain.invoke({
                    "difficulty": self.difficulty, 
                    "domain": self.domain, 
                    "total_number_data_samples": samples_to_request,
                    "previous_questions_context": previous_context,
                    "call_number": self.call_count
                })
                
                # Parse response
                if hasattr(response, 'content'):
                    response_content = response.content
                else:
                    response_content = str(response)
                
                new_questions = self._parse_response(response_content)
                
                if not new_questions:
                    print(f"  No questions parsed from response in attempt {attempt}")
                    continue
                
                # Filter duplicates
                batch_unique = self._filter_duplicates(new_questions)
                unique_questions.extend(batch_unique)
                
                print(f"  Generated {len(batch_unique)} unique questions (total: {len(unique_questions)})")
                
                # If we have enough unique questions, break
                if len(unique_questions) >= self.total_number_data:
                    unique_questions = unique_questions[:self.total_number_data]
                    break
                    
                # Update context for next attempt if we need more questions
                if attempt < max_attempts:
                    additional_context = f"\n\nATTEMPT {attempt + 1}: Need {self.total_number_data - len(unique_questions)} more UNIQUE questions. Be even more creative and explore completely different areas of {self.domain}."
                    previous_context += additional_context
                    
            except Exception as e:
                print(f"  Error in attempt {attempt}: {e}")
                continue
        
        # Store the generated questions for future deduplication
        self.generated_questions.extend(unique_questions)
        
        print(f"Batch #{self.call_count} completed: {len(unique_questions)} unique questions generated")
        print(f"Total questions generated across all calls: {len(self.generated_questions)}")
        
        return unique_questions
    
    def get_all_generated_questions(self) -> List[Dict]:
        """Get all questions generated across all calls"""
        return self.generated_questions.copy()
    
    def get_statistics(self) -> Dict:
        """Get statistics about generated questions"""
        return {
            "total_calls": self.call_count,
            "total_questions": len(self.generated_questions),
            "unique_questions": len(self.question_hashes),
            "questions_per_call": len(self.generated_questions) / max(self.call_count, 1)
        }
    
    def reset(self):
        """Reset the agent to start fresh"""
        self.generated_questions = []
        self.question_hashes = set()
        self.call_count = 0

