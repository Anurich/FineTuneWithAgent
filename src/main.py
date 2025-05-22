from Agent.creator_agent.create import Agent_A
import json
from model.train import MODEL
# Usage example:
if __name__ == "__main__":
    # Create agent
   
        agent = Agent_A(total_number_of_data=500, difficulty=1, domain="mathematics")
        
        # Generate multiple batches
        all_questions = []
        for i in range(5):
            print(f"\n--- Generating batch {i+1} ---")
            questions = agent._create_data()
            # Save all questions
            all_questions = agent.get_all_generated_questions()
            
            train_index = len(all_questions)*0.8
            train_data = all_questions[:int(train_index)]
            test_data = all_questions[int(train_index):]
            
            print(f"Test data: {len(test_data)} & Train data: {len(train_data)}")
            md = MODEL(train_data,  test_data)
            md._train()
            
            # let's do the prediction 
            performance_history = md._prediction()
            print(performance_history)
            break
            
    