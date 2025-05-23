from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.output_parsers import JsonOutputToolsParser
import json
import hashlib
import numpy as np
import statistics
from typing import List, Dict, Set, Optional
load_dotenv()

class FeedbackGuidedAgentA:
    def __init__(self, total_number_of_data: int = 100, difficulty: int = 0, domain="math"):
        self.llm = ChatOpenAI(model="gpt-4.1-mini")
        self.domain = domain 
        self.total_number_data = total_number_of_data
        self.difficulty = difficulty
        
        # Track generated questions across multiple calls
        self.generated_questions = []
        self.question_hashes = set()
        self.call_count = 0
        
        # NEW: Training feedback integration
        self.training_feedback = None
        self.quality_analysis = None
        self.performance_insights = None
        
        self.prompt_template = PromptTemplate(
            input_variables=[
                "difficulty", "domain", "total_number_data_samples", 
                "previous_questions_context", "call_number", "training_feedback_context"
            ],
            template="""You are an expert at generating question-reasoning-answer triplets in JSON format.
            Your task is to generate exactly {total_number_data_samples} unique triplets for training a language model, strictly adhering to the specified difficulty and uniqueness requirements.

            Current difficulty level: {difficulty}/10
            Domain: {domain}
            Required number of samples: {total_number_data_samples}
            Call number: {call_number}

            ---
            TRAINING PERFORMANCE FEEDBACK:
            {training_feedback_context}
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
            5.  **TRAINING FEEDBACK INTEGRATION**: Use the training performance feedback above to adjust your question generation strategy. Focus on creating questions that will help improve the model's weaknesses while reinforcing its strengths.
            6. Always return the valid JSON.
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
    
    def integrate_training_feedback(self, prediction_results: Dict):
        """
        Integrate training feedback from the model's performance
        
        Args:
            prediction_results: Dictionary containing:
                - quality_history: List of quality scores over training
                - quality_ema: Exponential moving average of quality
                - current_feedback: Latest feedback with z-scores
        """
        self.training_feedback = prediction_results
        self._analyze_training_performance()
        print(f"Training feedback integrated: {len(self.training_feedback.get('quality_history', []))} quality samples analyzed")
    
    def _analyze_training_performance(self):
        """Analyze the training performance to generate insights"""
        if not self.training_feedback:
            return
        
        quality_history = self.training_feedback.get('quality_history', [])
        quality_ema = self.training_feedback.get('quality_ema', 0.5)
        current_feedback = self.training_feedback.get('current_feedback', {})
        
        if not quality_history:
            self.quality_analysis = None
            return
        
        # Statistical analysis of quality progression
        early_performance = np.mean(quality_history[:len(quality_history)//3]) if len(quality_history) > 6 else 0.5
        recent_performance = np.mean(quality_history[-len(quality_history)//3:]) if len(quality_history) > 6 else quality_ema
        
        performance_trend = recent_performance - early_performance
        performance_variance = np.var(quality_history) if len(quality_history) > 1 else 0.0
        
        # Z-score analysis from current feedback
        reasoning_z = current_feedback.get('reasoning_score', 0.0)
        solution_z = current_feedback.get('solution_score', 0.0)
        is_correct = current_feedback.get('is_correct', False)
        
        self.quality_analysis = {
            'overall_ema': quality_ema,
            'performance_trend': performance_trend,
            'performance_variance': performance_variance,
            'early_performance': early_performance,
            'recent_performance': recent_performance,
            'current_reasoning_z': reasoning_z,
            'current_solution_z': solution_z,
            'current_correctness': is_correct,
            'total_samples': len(quality_history)
        }
        
        self._generate_performance_insights()
    
    def _generate_performance_insights(self):
        """Generate actionable insights from the training performance"""
        if not self.quality_analysis:
            self.performance_insights = "No training performance data available. Generate diverse, high-quality questions."
            return
        
        insights = []
        qa = self.quality_analysis
        
        # Overall performance assessment
        if qa['overall_ema'] > 0.7:
            insights.append("✅ STRONG PERFORMANCE: Model is performing well overall. Continue with current question complexity.")
        elif qa['overall_ema'] > 0.5:
            insights.append("⚡ MODERATE PERFORMANCE: Model shows reasonable learning. Consider slightly more structured questions.")
        else:
            insights.append("📈 IMPROVEMENT NEEDED: Model struggling. Focus on clearer, more systematic questions.")
        
        # Performance trend analysis
        if qa['performance_trend'] > 0.1:
            insights.append("📈 POSITIVE TREND: Model is improving rapidly. Can introduce more challenging variations.")
        elif qa['performance_trend'] < -0.1:
            insights.append("📉 DECLINING TREND: Model performance decreasing. Simplify questions and improve clarity.")
        else:
            insights.append("➡️ STABLE TREND: Model performance is steady. Maintain current approach with slight variations.")
        
        # Reasoning vs Solution analysis (z-scores)
        if qa['current_reasoning_z'] < -1.0:
            insights.append("🧠 REASONING WEAKNESS: Model struggles with reasoning steps. Generate questions with:")
            insights.append("   • Clearer step-by-step logical progression")
            insights.append("   • More explicit intermediate steps")
            insights.append("   • Better structured problem-solving approach")
        elif qa['current_reasoning_z'] > 1.0:
            insights.append("🧠 REASONING STRENGTH: Model excels at reasoning. Can use more complex logical chains.")
        
        if qa['current_solution_z'] < -1.0:
            insights.append("🎯 SOLUTION WEAKNESS: Model struggles with final answers. Generate questions with:")
            insights.append("   • Clearer expected answer format")
            insights.append("   • More direct path to solution")
            insights.append("   • Better connection between reasoning and final answer")
        elif qa['current_solution_z'] > 1.0:
            insights.append("🎯 SOLUTION STRENGTH: Model excels at final answers. Can use more complex problem formats.")
        
        # Variance analysis
        if qa['performance_variance'] > 0.1:
            insights.append("🎲 HIGH VARIANCE: Model performance is inconsistent. Focus on:")
            insights.append("   • More consistent question formatting")
            insights.append("   • Standardized reasoning patterns")
            insights.append("   • Reduced ambiguity in problem statements")
        
        # Specific recommendations based on current state
        if not qa['current_correctness']:
            insights.append("❌ RECENT ERRORS: Last prediction was incorrect. Next questions should:")
            insights.append("   • Have unambiguous correct answers")
            insights.append("   • Include verification steps in reasoning")
            insights.append("   • Use familiar mathematical contexts")
        
        # Sample size considerations
        if qa['total_samples'] < 20:
            insights.append("⏰ EARLY TRAINING: Limited data available. Focus on foundational question types.")
        
        self.performance_insights = "\n".join(insights)
    
    def _get_training_feedback_context(self) -> str:
        """Generate context about training performance for the prompt"""
        if not self.training_feedback or not self.quality_analysis:
            return """No training performance data available yet. 
            Generate high-quality, diverse questions that will effectively train the model.
            Focus on clear reasoning steps and unambiguous answers."""
        
        qa = self.quality_analysis
        
        context = f"""TRAINING PERFORMANCE ANALYSIS:
        
📊 CURRENT MODEL STATE:
• Overall Quality Score: {qa['overall_ema']:.3f} (EMA across training)
• Performance Trend: {'Improving' if qa['performance_trend'] > 0.05 else 'Declining' if qa['performance_trend'] < -0.05 else 'Stable'}
• Training Samples: {qa['total_samples']} questions processed
• Performance Consistency: {'High variance' if qa['performance_variance'] > 0.1 else 'Low variance'}

🎯 LATEST ASSESSMENT (Z-Scores):
• Reasoning Quality: {qa['current_reasoning_z']:.2f} ({'Strong' if qa['current_reasoning_z'] > 0.5 else 'Weak' if qa['current_reasoning_z'] < -0.5 else 'Average'})
• Solution Quality: {qa['current_solution_z']:.2f} ({'Strong' if qa['current_solution_z'] > 0.5 else 'Weak' if qa['current_solution_z'] < -0.5 else 'Average'})
• Last Answer Correct: {'Yes' if qa['current_correctness'] else 'No'}

📋 ACTIONABLE INSIGHTS:
{self.performance_insights}

🎯 QUESTION GENERATION STRATEGY:
Based on this analysis, adjust your question generation to address the identified weaknesses while building on strengths. 
Focus on creating questions that will help the model improve in areas where it's struggling.
"""
        
        return context
    
    def _get_question_hash(self, question: str) -> str:
        """Generate a hash for the question to check for duplicates"""
        normalized = question.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _get_previous_questions_context(self, max_examples: int = 15) -> str:
        """Generate context about previously generated questions"""
        if not self.generated_questions:
            return "FIRST CALL: No previous questions generated yet. You can explore any area of the domain."
        
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
            if response_content.strip().startswith('['):
                return json.loads(response_content)
            
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
        """Create data with integrated training feedback"""
        self.call_count += 1
        print(f"Creating feedback-guided data batch #{self.call_count}...")
        
        if self.training_feedback:
            print(f"📊 Using training feedback with {len(self.training_feedback.get('quality_history', []))} quality samples")
            print(f"📈 Current model quality EMA: {self.training_feedback.get('quality_ema', 0.5):.3f}")
        
        # Get contexts
        previous_context = self._get_previous_questions_context()
        training_context = self._get_training_feedback_context()
        
        max_attempts = 3
        attempt = 0
        unique_questions = []
        
        while len(unique_questions) < self.total_number_data and attempt < max_attempts:
            attempt += 1
            print(f"  Attempt {attempt}/{max_attempts}")
            
            samples_needed = self.total_number_data - len(unique_questions)
            samples_to_request = min(samples_needed + 10, self.total_number_data)
            
            try:
                response = self.chain.invoke({
                    "difficulty": self.difficulty, 
                    "domain": self.domain, 
                    "total_number_data_samples": samples_to_request,
                    "previous_questions_context": previous_context,
                    "call_number": self.call_count,
                    "training_feedback_context": training_context
                })
                
                if hasattr(response, 'content'):
                    response_content = response.content
                else:
                    response_content = str(response)
                
                new_questions = self._parse_response(response_content)
                
                if not new_questions:
                    print(f"  No questions parsed from response in attempt {attempt}")
                    continue
                
                batch_unique = self._filter_duplicates(new_questions)
                unique_questions.extend(batch_unique)
                
                print(f"  Generated {len(batch_unique)} unique questions (total: {len(unique_questions)})")
                
                if len(unique_questions) >= self.total_number_data:
                    unique_questions = unique_questions[:self.total_number_data]
                    break
                    
                if attempt < max_attempts:
                    additional_context = f"\n\nATTEMPT {attempt + 1}: Need {self.total_number_data - len(unique_questions)} more UNIQUE questions. Be even more creative and explore completely different areas of {self.domain}."
                    previous_context += additional_context
                    
            except Exception as e:
                print(f"  Error in attempt {attempt}: {e}")
                continue
        
        self.generated_questions.extend(unique_questions)
        
        print(f"Batch #{self.call_count} completed: {len(unique_questions)} unique questions generated")
        print(f"Total questions generated across all calls: {len(self.generated_questions)}")
        
        return unique_questions
    
    def create_feedback_guided_data(self, prediction_results: Optional[Dict] = None) -> List[Dict]:
        """
        Create data with optional training feedback integration
        
        Args:
            prediction_results: Optional training feedback from model
            
        Returns:
            List of generated question-reasoning-answer triplets
        """
        if prediction_results:
            self.integrate_training_feedback(prediction_results)
        
        return self._create_data()
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of training performance analysis"""
        if not self.quality_analysis:
            return {"status": "No training feedback available"}
        
        return {
            "training_samples": self.quality_analysis['total_samples'],
            "current_quality_ema": self.quality_analysis['overall_ema'],
            "performance_trend": self.quality_analysis['performance_trend'],
            "reasoning_strength": "Strong" if self.quality_analysis['current_reasoning_z'] > 0.5 else "Weak" if self.quality_analysis['current_reasoning_z'] < -0.5 else "Average",
            "solution_strength": "Strong" if self.quality_analysis['current_solution_z'] > 0.5 else "Weak" if self.quality_analysis['current_solution_z'] < -0.5 else "Average",
            "recommended_focus": self._get_recommended_focus()
        }
    
    def _get_recommended_focus(self) -> List[str]:
        """Get recommended focus areas based on analysis"""
        if not self.quality_analysis:
            return ["Generate diverse, high-quality questions"]
        
        qa = self.quality_analysis
        recommendations = []
        
        if qa['current_reasoning_z'] < -0.5:
            recommendations.append("Improve reasoning clarity")
        if qa['current_solution_z'] < -0.5:
            recommendations.append("Strengthen solution accuracy")
        if qa['performance_variance'] > 0.1:
            recommendations.append("Increase consistency")
        if qa['overall_ema'] < 0.5:
            recommendations.append("Simplify question complexity")
        
        if not recommendations:
            recommendations.append("Maintain current quality level")
        
        return recommendations
    
    def get_all_generated_questions(self) -> List[Dict]:
        """Get all questions generated across all calls"""
        return self.generated_questions.copy()
    
    def get_statistics(self) -> Dict:
        """Get statistics about generated questions"""
        base_stats = {
            "total_calls": self.call_count,
            "total_questions": len(self.generated_questions),
            "unique_questions": len(self.question_hashes),
            "questions_per_call": len(self.generated_questions) / max(self.call_count, 1)
        }
        
        if self.quality_analysis:
            base_stats.update({
                "training_feedback_integrated": True,
                "model_quality_ema": self.quality_analysis['overall_ema'],
                "performance_trend": self.quality_analysis['performance_trend']
            })
        else:
            base_stats["training_feedback_integrated"] = False
        
        return base_stats
    
    def reset(self):
        """Reset the agent to start fresh"""
        self.generated_questions = []
        self.question_hashes = set()
        self.call_count = 0
        self.training_feedback = None
        self.quality_analysis = None
        self.performance_insights = None

# Example usage
