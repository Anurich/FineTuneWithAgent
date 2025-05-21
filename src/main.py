from Agent.creator_agent.create import Agent_A
import json
from model.train import MODEL
# Usage example:
if __name__ == "__main__":
    # Create agent
    """
        agent = Agent_A(total_number_of_data=100, difficulty=5, domain="mathematics")
        
        # Generate multiple batches
        for i in range(5):
            print(f"\n--- Generating batch {i+1} ---")
            questions = agent._create_data()
            print(f"Generated {len(questions)} questions in batch {i+1}")
        
        # Get statistics
        stats = agent.get_statistics()
        print(f"\nFinal Statistics: {stats}")
        
        # Save all questions
        all_questions = agent.get_all_generated_questions()
        with open('all_generated_questions.json', 'w', encoding="utf-8") as f:
            json.dump(all_questions, f, indent=2)
    """
    
    md = MODEL()
    