import numpy as np


class GridOpsEnv:
    """
    Multi-zone power grid environment with bidding-based allocation.
    Gym-style API (no external RL dependencies).
    """

    def __init__(self, num_zones=3, max_time=50, seed=None):
        self.num_zones = num_zones
        self.max_time = max_time
        self.mode = "baseline"  # "baseline", "selfish", "coordinated"
        self.reward_mode = "local"
        self.reputation = np.ones(self.num_zones, dtype=float)
        self.rep_decay = 0.1
        self.rep_recover = 0.02
        self.rep_min = 0.2
        self.rep_max = 2.0
        self.ethical_weight = 1.5
        self.seed(seed)
        
    def set_mode(self, mode: str):
        self.mode = mode

    def set_reward_mode(self, mode: str):
        self.reward_mode = mode

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.time = 0
        self.reputation = np.ones(self.num_zones, dtype=float)
        self._init_state()
        return self._get_obs(), {}

    def step(self, action=None):
        if action is None:
            if self.mode == "selfish":
                action = self.selfish_policy()
            elif self.mode == "coordinated":
                action = self.coordinated_policy()
            else:
                action = self.rng.integers(1, 10, size=self.num_zones)

        action = np.asarray(action, dtype=float)
        if action.shape != (self.num_zones,):
            raise ValueError(
                f"Expected action shape ({self.num_zones},), got {action.shape}"
            )

        bids = np.array(action, dtype=float)
        overbid = bids > (1.1 * self.demand)
        honesty_pen = overbid.astype(float) * 2.0

        self.reputation = np.clip(
            self.reputation - self.rep_decay * overbid.astype(float) + self.rep_recover,
            self.rep_min, self.rep_max
        )

        self._honesty_pen = honesty_pen
        self._overbid_mask = overbid

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
        self.history["stability"].append(info["stability_score"])
        self.history["avg_reputation"].append(info["avg_reputation"])
        self.history["honesty_violations"].append(info["honesty_violations"])

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
            "imbalance": [], "efficiency": [], "stability": [],
            "avg_reputation": [], "honesty_violations": []
        }

        # Initialize masks so _compute_reward/_get_info are safe pre-step
        self._overload_mask = np.zeros(self.num_zones, dtype=bool)
        self._blackout_mask = np.zeros(self.num_zones, dtype=bool)
        self._local_rewards = np.zeros(self.num_zones, dtype=float)

    # ------------------------------------------------------------------
    # Action → Allocation
    # ------------------------------------------------------------------

    def _apply_action(self, action):
        bids = np.maximum(np.array(action, dtype=float), 0.0)
        weights = bids * self.reputation
        total_w = weights.sum() + 1e-8

        self.allocated = (weights / total_w) * self.total_power

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

        if self.reward_mode == "local":
            return float(served.sum())

        if self.reward_mode == "global":
            priority_weights = self.priority.astype(float)
            priority_weights[self.priority == 3] *= self.ethical_weight
            ethical_served = (served * priority_weights).sum()

            alloc = self.allocated
            share = alloc / (alloc.sum() + 1e-8)
            imbalance = np.var(share)

            fairness_pen = 3 * imbalance

            honesty_pen_total = float(self._honesty_pen.sum())

            reward = (
                ethical_served
                - overload_pen
                - blackout_pen
                - fairness_pen
                - honesty_pen_total
            )
            return float(reward)

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
        imbalance = np.var(share)
        fairness_pen = 3 * imbalance
        
        served = np.minimum(alloc, self.demand)
        overloads = int(self._overload_mask.sum())
        blackouts = int(self._blackout_mask.sum())
        
        return {
            "served": float(served.sum()),
            "weighted_served": float(np.sum(served * self.priority)),
            "overloads": overloads,
            "blackouts": blackouts,
            "local_rewards": list(self._local_rewards),
            "imbalance": float(imbalance),
            "efficiency": float(served.sum() / (self.demand.sum() + 1e-8)),
            "fairness_penalty": float(fairness_pen),
            "stability_score": float(1.0 / (1 + blackouts + overloads)),
            "avg_reputation": float(self.reputation.mean()),
            "min_reputation": float(self.reputation.min()),
            "honesty_violations": int(self._overbid_mask.sum()),
        }

    # ------------------------------------------------------------------
    # Baseline Policies
    # ------------------------------------------------------------------

    def selfish_policy(self):
        alpha = 1.2
        noise = self.rng.integers(0, 3, size=self.num_zones)
        bid = self.demand * self.priority * alpha + noise
        return np.clip(bid, 0, None).astype(float)

    def coordinated_policy(self):
        weights = self.priority * self.demand * self.reputation
        total = weights.sum() + 1e-8
        alloc = (weights / total) * self.total_power
        return alloc.astype(float)

    def get_history(self) -> dict:
        return self.history

    def summarize_episode(self):
        return {
            "avg_reward":       np.mean(self.history["reward"]),
            "avg_blackouts":    np.mean(self.history["blackouts"]),
            "avg_imbalance":    np.mean(self.history["imbalance"]),
            "avg_stability":    np.mean(self.history["stability"]),
            "avg_reputation":   np.mean(self.history["avg_reputation"]),
            "honesty_violations": int(np.sum(self.history["honesty_violations"])),
        }


# ----------------------------------------------------------------------
# Behavioral validation: selfish vs coordinated (global reward)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    configs = [
        ("selfish",     "global"),
        ("coordinated", "global"),
    ]

    results = []
    for mode, reward_mode in configs:
        env = GridOpsEnv(num_zones=3, max_time=50, seed=42)
        env.set_mode(mode)
        env.set_reward_mode(reward_mode)
        env.reset()

        truncated = False
        while not truncated:
            _, _, _, truncated, _ = env.step()

        summary = env.summarize_episode()
        results.append((mode, reward_mode, summary))

    # Print extended comparison table
    keys = [
        ("avg_reward",        "avg_reward"),
        ("avg_blackouts",     "avg_blackouts"),
        ("avg_imbalance",     "avg_imbalance"),
        ("avg_stability",     "avg_stability"),
        ("avg_reputation",    "avg_reputation"),
        ("honesty_violations","violations"),
    ]
    col = 16
    header = f"{'mode':<{col}} {'reward_mode':<{col}}" + "".join(
        f" {label:>{col}}" for _, label in keys
    )
    print(header)
    print("-" * len(header))
    for mode, reward_mode, s in results:
        row = f"{mode:<{col}} {reward_mode:<{col}}"
        for key, _ in keys:
            val = s[key]
            if isinstance(val, int):
                row += f" {val:>{col}d}"
            else:
                row += f" {val:>{col}.4f}"
        print(row)


