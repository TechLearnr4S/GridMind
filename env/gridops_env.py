import numpy as np


class GridOpsEnv:
    """
    Multi-zone power grid environment with bidding-based allocation.
    Gym-style API (no external RL dependencies).
    """

    def __init__(self, num_zones=3, max_time=50, seed=None):
        self.num_zones = num_zones
        self.max_time = max_time
        self.mode = "baseline"
        self.seed(seed)
        
    def set_mode(self, mode: str):
        self.mode = mode

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.time = 0
        self._init_state()
        return self._get_obs(), {}

    def step(self, action=None):
        if action is None:
            if self.mode == "selfish":
                action = self.selfish_policy()
            else:
                action = self.rng.integers(1, 10, size=self.num_zones)

        action = np.asarray(action, dtype=float)
        if action.shape != (self.num_zones,):
            raise ValueError(
                f"Expected action shape ({self.num_zones},), got {action.shape}"
            )

        self.time += 1
        self._apply_action(action)
        self._dynamics()
        self._local_rewards = self._compute_local_rewards()

        reward = self._compute_reward()
        obs = self._get_obs()
        terminated = False
        truncated = self.time >= self.max_time
        info = self._get_info()

        # Append metrics to history
        self.history["reward"].append(reward)
        self.history["blackouts"].append(info["blackouts"])
        self.history["overloads"].append(info["overloads"])
        self.history["imbalance"].append(info["imbalance"])
        self.history["efficiency"].append(info["efficiency"])

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # State Initialization
    # ------------------------------------------------------------------

    def _init_state(self):
        self.demand = self.rng.integers(8, 17, size=self.num_zones)       # [8, 16]
        self.allocated = np.zeros(self.num_zones, dtype=float)
        self.priority = self.rng.integers(1, 4, size=self.num_zones)      # {1, 2, 3}
        self.total_power = int(self.rng.integers(25, 41))                 # [25, 40]
        self.failed = np.zeros(self.num_zones, dtype=bool)
        self.history = {
            "reward": [], "blackouts": [], "overloads": [],
            "imbalance": [], "efficiency": []
        }

        # Initialize masks so _compute_reward/_get_info are safe pre-step
        self._overload_mask = np.zeros(self.num_zones, dtype=bool)
        self._blackout_mask = np.zeros(self.num_zones, dtype=bool)
        self._local_rewards = np.zeros(self.num_zones, dtype=float)

    # ------------------------------------------------------------------
    # Action → Allocation
    # ------------------------------------------------------------------

    def _apply_action(self, action):
        action = np.clip(action, 0, None)
        total_bids = action.sum() + 1e-8
        self.allocated = (action / total_bids) * self.total_power

    # ------------------------------------------------------------------
    # Stochastic Dynamics
    # ------------------------------------------------------------------

    def _dynamics(self):
        # Demand drift: uniform noise in [-2, +2], floor at 1
        noise = self.rng.integers(-2, 3, size=self.num_zones)
        self.demand = np.maximum(self.demand + noise, 1)

        # Fault conditions
        self._overload_mask = self.allocated > 1.3 * self.demand
        self._blackout_mask = self.allocated < 0.4 * self.demand

        # Zones that overloaded are permanently marked failed
        self.failed = np.logical_or(self.failed, self._overload_mask)

        # Cascade effect and recovery
        if any(self._overload_mask):
            self.total_power = max(int(self.total_power * 0.9), 10)
        else:
            self.total_power = min(int(self.total_power * 1.02), 50)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self):
        served = np.minimum(self.allocated, self.demand)
        weighted_served = float(np.sum(served * self.priority))

        overload_pen = 5.0 * int(self._overload_mask.sum())
        blackout_pen = 5.0 * int(self._blackout_mask.sum())

        return weighted_served - overload_pen - blackout_pen

    def _compute_local_rewards(self):
        served = np.minimum(self.allocated, self.demand)
        local = served.copy().astype(float)
        return local

    # ------------------------------------------------------------------
    # Observation & Info
    # ------------------------------------------------------------------

    def _get_obs(self):
        return {
            "time": self.time,
            "demand": self.demand.copy(),
            "allocated": self.allocated.copy(),
            "priority": self.priority.copy(),
            "total_power": self.total_power,
        }

    def _get_info(self):
        alloc = self.allocated
        total = alloc.sum() + 1e-8
        share = alloc / total
        
        served = np.minimum(alloc, self.demand)
        return {
            "served": float(served.sum()),
            "weighted_served": float(np.sum(served * self.priority)),
            "overloads": int(self._overload_mask.sum()),
            "blackouts": int(self._blackout_mask.sum()),
            "local_rewards": list(self._local_rewards),
            "imbalance": float(np.var(share)),
            "efficiency": float(served.sum() / (self.demand.sum() + 1e-8)),
        }

    # ------------------------------------------------------------------
    # Baseline Policies
    # ------------------------------------------------------------------

    def selfish_policy(self):
        alpha = 1.2
        noise = self.rng.integers(0, 3, size=self.num_zones)
        bid = self.demand * self.priority * alpha + noise
        return np.clip(bid, 0, None).astype(float)

    def get_history(self) -> dict:
        return self.history


# ----------------------------------------------------------------------
# Quick sanity test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    for mode in ["baseline", "selfish"]:
        env = GridOpsEnv(num_zones=3, max_time=50, seed=42)
        env.set_mode(mode)
        env.reset()
        
        for _ in range(10):
            env.step()
            
        avg_reward = np.mean(env.history["reward"])
        avg_blackouts = np.mean(env.history["blackouts"])
        avg_imbalance = np.mean(env.history["imbalance"])
        
        print(f"Mode: {mode}")
        print(f"  Avg Reward:    {avg_reward:.2f}")
        print(f"  Avg Blackouts: {avg_blackouts:.2f}")
        print(f"  Avg Imbalance: {avg_imbalance:.4f}\n")
