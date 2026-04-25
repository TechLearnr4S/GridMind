import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train.train import GridOpsEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
        return True

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Wrap with Monitor to ensure episode info is logged correctly
    env = DummyVecEnv([lambda: Monitor(GridOpsEnvWrapper())])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    from typing import Callable

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=linear_schedule(1e-4), # Slow, stable annealing
        n_steps=1024,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,              # Relaxed clipping, let PPO handle constraints
        ent_coef=0.005,              # Reduces excessive randomness
        vf_coef=0.7,                 # Prevents exploding value_loss
        max_grad_norm=0.5,           # Strict gradient clipping for LSTM
        target_kl=None,              # REMOVED: KL early stopping breaks LSTM hidden states
        policy_kwargs=dict(
            net_arch=[256, 256],
            enable_critic_lstm=True, # Ensure critic also uses LSTM
        ),
        verbose=1
    )

    logger = RewardLogger()

    model.learn(
        total_timesteps=200_000,
        callback=logger
    )

    model.save("models/ppo_lstm_final")
    env.save("models/vecnormalize_lstm_final.pkl")

    rewards_to_save = logger.rewards if logger.rewards else []
    np.save("outputs/train_rewards_lstm.npy", rewards_to_save)
    print(f"Rewards logged: {len(rewards_to_save)} episodes")
    print("Saved: outputs/train_rewards_lstm.npy")

if __name__ == "__main__":
    train()
