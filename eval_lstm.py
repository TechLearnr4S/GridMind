import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

MODEL_PATH  = "models/ppo_lstm_final"
VECNORM_PATH = "models/vecnormalize_lstm_final.pkl"
EPISODES    = 50

def evaluate():
    print(f"Running deterministic evaluation over {EPISODES} episodes...")
    
    # ── TASK 1: Correct Environment Setup ────────────────────────
    # Wrap environment identically to training
    env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
    
    # Load VecNormalize statistics from training
    if not os.path.exists(VECNORM_PATH):
        print(f"Error: Could not find VecNormalize file at {VECNORM_PATH}")
        return
        
    env = VecNormalize.load(VECNORM_PATH, env)
    
    # CRITICAL: Disable training mode so moving averages are frozen
    env.training = False
    env.norm_reward = False

    # ── TASK 2: Proper Model Loading ────────────────────────────
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Error: Could not find model file at {MODEL_PATH}.zip")
        return
        
    model = RecurrentPPO.load(MODEL_PATH, env=env)
    
    # ── TASK 5: Metrics Tracking ────────────────────────────────
    total_reward    = 0.0
    total_blackouts = 0.0
    stability_list  = []

    for ep in range(EPISODES):
        obs = env.reset()
        
        # ── TASK 3: Fix LSTM Inference (Initialization) ──────────
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        done = np.zeros((env.num_envs,), dtype=bool)
        
        ep_reward = 0.0
        ep_blackouts = 0.0
        ep_stability = []

        while not done[0]:
            # ── TASK 4: Deterministic Evaluation ────────────────
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )

            # Step environment
            obs, reward, done, info = env.step(action)
            
            # ── TASK 3: Fix LSTM Inference (Update flags) ───────
            episode_starts = done.copy()

            # Accumulate reward (using raw unnormalized reward)
            ep_reward += reward[0]
            
            # Extract precise metrics from the info dictionary
            if "blackout_count" in info[0]:
                ep_blackouts += info[0]["blackout_count"]
            elif "blackouts" in info[0]:
                ep_blackouts += info[0]["blackouts"]
            elif "fault_count" in info[0]:
                ep_blackouts += info[0]["fault_count"]
                
            if "stability_score" in info[0]:
                ep_stability.append(info[0]["stability_score"])

            if done[0]:
                total_reward += ep_reward
                total_blackouts += ep_blackouts
                if ep_stability:
                    stability_list.append(float(np.mean(ep_stability)))

    # Compute averages
    avg_reward = total_reward / EPISODES
    avg_blackouts = total_blackouts / EPISODES
    avg_stability = float(np.mean(stability_list)) if stability_list else 0.0
    
    # ── TASK 6: Output ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print("FINAL EVALUATION RESULTS (PPO LSTM)")
    print("=" * 50)
    print(f"Avg Reward/Episode : {avg_reward:.3f}")
    print(f"Avg Blackouts      : {avg_blackouts:.3f}")
    print(f"Avg Stability      : {avg_stability:.3f}")
    print("=" * 50)

    import json
    results = {
        "metrics": {
            "reward": {"mean": float(avg_reward)},
            "blackouts": {"mean": float(avg_blackouts)},
            "stability": {"mean": float(avg_stability)}
        }
    }
    with open("outputs/eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved results to outputs/eval_results.json")


if __name__ == "__main__":
    evaluate()
