from Agent.creator_agent.create import FeedbackGuidedAgentA  # Your original Agent_A
# from Agent.creator_agent.feedback_guided_create import FeedbackGuidedAgentA  # If you want to use the new one
import json
from model.train import MODEL

def main_training_loop():
    """
    Corrected main training loop with proper feedback integration
    """
    print("🚀 Starting iterative training with feedback-guided data generation")
    
    # Initialize agent - use your original Agent_A for now
    agent = FeedbackGuidedAgentA(total_number_of_data=100, difficulty=1, domain="mathematics")  # Reduced size per batch
    
    # Track performance across iterations
    iteration_results = []
    
    for iteration in range(5):
        print(f"\n{'='*60}")
        print(f"🔄 ITERATION {iteration + 1}/5")
        print(f"{'='*60}")
        
        if iteration == 0:
            # First iteration: Generate initial data without feedback
            print("📝 Generating initial dataset (no feedback available yet)...")
            questions = agent._create_data()
        else:
            # Subsequent iterations: Use feedback from previous training
            print(f"🎯 Generating feedback-guided data based on iteration {iteration} results...")
            
            # Get the latest performance feedback
            previous_performance = iteration_results[-1]['performance_feedback']
            
            # Check if your Agent_A has the feedback method, if not, use regular generation
            if hasattr(agent, 'create_feedback_guided_data'):
                questions = agent.create_feedback_guided_data(previous_performance)
            else:
                print("⚠️  Using regular data generation (feedback integration not available)")
                questions = agent._create_data()
        
        # Get all questions generated so far
        all_questions = agent.get_all_generated_questions()
        print(f"📊 Total questions generated: {len(all_questions)}")
        
        # Split into train/test with proper indexing
        train_split_ratio = 0.8
        train_index = int(len(all_questions) * train_split_ratio)
        
        train_data = all_questions[:train_index]
        test_data = all_questions[train_index:]
        
        print(f"📈 Training data: {len(train_data)} questions")
        print(f"📉 Test data: {len(test_data)} questions")
        
        # Ensure we have enough data for both train and test
        if len(test_data) < 10:
            print("⚠️  Warning: Very small test set. Consider generating more data.")
        
        # Train the model
        print(f"🤖 Training model on {len(train_data)} questions...")
        try:
            model = MODEL(train_data, test_data)
            model._train()
            
            # Get performance feedback
            print("📊 Extracting performance feedback...")
            performance_feedback = model._prediction()
            
            # Store results for this iteration
            iteration_result = {
                'iteration': iteration + 1,
                'total_questions': len(all_questions),
                'train_size': len(train_data),
                'test_size': len(test_data),
                'new_questions_this_iteration': len(questions) if iteration > 0 else len(all_questions),
                'performance_feedback': performance_feedback
            }
            iteration_results.append(iteration_result)
            
            # Print performance summary
            print_performance_summary(performance_feedback, iteration + 1)
            
        except Exception as e:
            print(f"❌ Error during training in iteration {iteration + 1}: {e}")
            print("Continuing to next iteration...")
            continue
    
    print(f"\n{'='*60}")
    print("🎉 TRAINING LOOP COMPLETED")
    print(f"{'='*60}")
    
    # Final summary
    print_final_summary(iteration_results, agent)
    
    return iteration_results, agent

def print_performance_summary(performance_feedback, iteration):
    """Print a summary of the current iteration's performance"""
    print(f"\n📊 ITERATION {iteration} PERFORMANCE SUMMARY:")
    print("-" * 40)
    
    quality_history = performance_feedback.get('quality_history', [])
    quality_ema = performance_feedback.get('quality_ema', 0.0)
    current_feedback = performance_feedback.get('current_feedback', {})
    
    if quality_history:
        print(f"📈 Quality samples: {len(quality_history)}")
        print(f"📊 Current EMA: {quality_ema:.3f}")
        print(f"📋 Latest reasoning z-score: {current_feedback.get('reasoning_score', 0.0):.3f}")
        print(f"📋 Latest solution z-score: {current_feedback.get('solution_score', 0.0):.3f}")
        print(f"✅ Latest correctness: {current_feedback.get('is_correct', False)}")
        
        # Trend analysis
        if len(quality_history) > 10:
            early_avg = sum(quality_history[:5]) / 5
            recent_avg = sum(quality_history[-5:]) / 5
            trend = recent_avg - early_avg
            print(f"📈 Performance trend: {'+' if trend > 0 else ''}{trend:.3f}")
    else:
        print("📊 No quality history available")

def print_final_summary(iteration_results, agent):
    """Print final summary of all iterations"""
    print("\n📋 FINAL SUMMARY:")
    print("-" * 40)
    
    # Agent statistics
    stats = agent.get_statistics()
    print(f"🎯 Total questions generated: {stats['total_questions']}")
    print(f"🔄 Total generation calls: {stats['total_calls']}")
    print(f"📊 Questions per call: {stats['questions_per_call']:.1f}")
    
    # Performance progression
    if iteration_results:
        print(f"\n📈 PERFORMANCE PROGRESSION:")
        for result in iteration_results:
            if 'performance_feedback' in result:
                ema = result['performance_feedback'].get('quality_ema', 0.0)
                print(f"   Iteration {result['iteration']}: Quality EMA = {ema:.3f}")
    
    # Save results
    try:
        with open('training_results.json', 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = []
            for result in iteration_results:
                serializable_result = {
                    'iteration': result['iteration'],
                    'total_questions': result['total_questions'],
                    'train_size': result['train_size'],
                    'test_size': result['test_size'],
                    'new_questions_this_iteration': result['new_questions_this_iteration']
                }
                # Add performance metrics if available
                if 'performance_feedback' in result:
                    pf = result['performance_feedback']
                    serializable_result['quality_ema'] = pf.get('quality_ema', 0.0)
                    serializable_result['quality_history_length'] = len(pf.get('quality_history', []))
                    if 'current_feedback' in pf:
                        serializable_result['latest_reasoning_z'] = pf['current_feedback'].get('reasoning_score', 0.0)
                        serializable_result['latest_solution_z'] = pf['current_feedback'].get('solution_score', 0.0)
                        serializable_result['latest_correct'] = pf['current_feedback'].get('is_correct', False)
                
                serializable_results.append(serializable_result)
            
            json.dump(serializable_results, f, indent=2)
            print(f"💾 Results saved to training_results.json")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")

def alternative_loop_with_feedback_agent():
    """
    Alternative version using the FeedbackGuidedAgentA if you want to use it
    """
    print("🚀 Starting training with FeedbackGuidedAgentA")
    
    # Import the new agent (uncomment if you want to use it)
    # from Agent.creator_agent.feedback_guided_create import FeedbackGuidedAgentA
    # agent = FeedbackGuidedAgentA(total_number_of_data=100, difficulty=1, domain="mathematics")
    
    # For now, using regular Agent_A
    agent = Agent_A(total_number_of_data=100, difficulty=1, domain="mathematics")
    
    for iteration in range(5):
        print(f"\n🔄 ITERATION {iteration + 1}")
        
        if iteration == 0:
            questions = agent._create_data()
            performance_feedback = None
        else:
            # Use feedback if available
            questions = agent._create_data()  # Replace with feedback method when available
        
        # Rest of the training logic...
        all_questions = agent.get_all_generated_questions()
        
        # Continue with training as before...

# Usage example:
if __name__ == "__main__":
    try:
        iteration_results, final_agent = main_training_loop()
        print("\n✅ Training completed successfully!")
        
        # You can access the final agent and results here
        final_stats = final_agent.get_statistics()
        print(f"Final dataset size: {final_stats['total_questions']} questions")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise