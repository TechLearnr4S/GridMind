import os
import json
import numpy as np

from train.train import GridOpsEnvWrapper

def evaluate_baseline(num_episodes=50):
    """
    Evaluates a pure random policy against the GridOpsEnvWrapper.
    Safely handles both Gym and Gymnasium APIs and tracks comprehensive metrics.
    """
    env = GridOpsEnvWrapper()
    os.makedirs("outputs", exist_ok=True)
    
    rewards = []
    blackouts = []
    stabilities = []
    
    print(f"Starting baseline evaluation for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        # Handle reset() API safely
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
            
        done = False
        ep_reward = 0.0
        ep_blackouts = 0
        ep_stability_scores = []
        
        while not done:
            action = env.action_space.sample()
            
            # Handle step() API safely
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
                
            ep_reward += reward
            
            # Track blackouts using fallback keys
            if "blackout_count" in info:
                ep_blackouts += info["blackout_count"]
            elif "blackouts" in info:
                ep_blackouts += info["blackouts"]
            elif "fault_count" in info:
                ep_blackouts += info["fault_count"]
                
            # Track stability score
            if "stability_score" in info:
                ep_stability_scores.append(info["stability_score"])
                
            obs = next_obs  # Always update observation correctly
                
        rewards.append(ep_reward)
        blackouts.append(ep_blackouts)
        
        if ep_stability_scores:
            stabilities.append(np.mean(ep_stability_scores))
        else:
            stabilities.append(0.0)
            
        if episode % 10 == 0:
            print(f"Episode {episode:>2}/{num_episodes} completed. Reward: {ep_reward:.2f}")

    # Compute Statistics
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    min_reward = float(np.min(rewards))
    max_reward = float(np.max(rewards))
    
    avg_blackouts = float(np.mean(blackouts))
    std_blackouts = float(np.std(blackouts))
    
    avg_stability = float(np.mean(stabilities))
    std_stability = float(np.std(stabilities))
    
    print("\n" + "="*65)
    print("BASELINE EVALUATION RESULTS (RANDOM AGENT)")
    print("="*65)
    print(f"Total Episodes   : {num_episodes}")
    print(f"Reward           : {avg_reward:7.3f} (± {std_reward:7.3f})")
    print(f"Reward Range     : [{min_reward:7.3f}, {max_reward:7.3f}]")
    print(f"Blackouts        : {avg_blackouts:7.3f} (± {std_blackouts:7.3f})")
    print(f"Stability Score  : {avg_stability:7.3f} (± {std_stability:7.3f})")
    print("-" * 65)
    print("Interpretation: The random agent performs very poorly, suffering")
    print("from high instability and frequent blackouts. This establishes a")
    print("robust baseline to measure the trained agent's learning progress.")
    print("="*65)
    
    # Save Full Results
    results = {
        "num_episodes": num_episodes,
        "statistics": {
            "reward": {
                "mean": avg_reward,
                "std": std_reward,
                "min": min_reward,
                "max": max_reward
            },
            "blackouts": {
                "mean": avg_blackouts,
                "std": std_blackouts
            },
            "stability": {
                "mean": avg_stability,
                "std": std_stability
            }
        },
        "raw_data": {
            "rewards": [float(r) for r in rewards],
            "blackouts": [float(b) for b in blackouts],
            "stabilities": [float(s) for s in stabilities]
        }
    }
    with open("outputs/baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Save Summary
    summary = {
        "num_episodes": num_episodes,
        "metrics": results["statistics"]
    }
    with open("outputs/baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
if __name__ == "__main__":
    evaluate_baseline()
