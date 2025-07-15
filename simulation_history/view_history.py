import numpy as np
import os
import pickle

def inspect_last_round(filepath):
    with open(filepath, 'rb') as f:
        history = pickle.load(f)


        
    # --- Get the data from the very last round ---
    last_round_data = history[-1]
    
    # --- Unpack the tuple ---
    t, actions, rewards = last_round_data
    
    num_users = len(actions)
    
    print(f"\nData from Final Round (Round {t}):")
    print("-" * 50)
    print(f"{'User ID':<10} | {'Chosen Action (Slot)':<25} | {'Received Reward':<20}")
    print("-" * 50)
    
    # --- Print a portion of the data for clarity ---
    num_to_display = min(20, num_users) # Display first 20 users or less
    
    for i in range(num_to_display):
        user_id = i
        action_taken = actions[i]
        reward_received = rewards[i]
        print(f"{user_id:<10} | {action_taken:<25} | {reward_received:<20}")
        
    print("-" * 50)
    print(f"(Showing data for the first {num_to_display} out of {num_users} users)\n")


if __name__ == "__main__":
    output_dir = "simulation_history"

    # Find all .pkl files in the directory
    history_files = [f for f in os.listdir(output_dir) if f.endswith('.pkl')]


    # --- User Selection Menu ---
    print("Available history files:")
    for i, filename in enumerate(history_files):
        print(f"  [{i+1}] {filename}")

    while True:
        try:
            choice = input(f"\nPlease select a file to inspect by number (1-{len(history_files)}), or 'q' to quit: ")
            if choice.lower() == 'q':
                break
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(history_files):
                selected_file = os.path.join(output_dir, history_files[choice_idx])
                inspect_last_round(selected_file)
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An error occurred: {e}")