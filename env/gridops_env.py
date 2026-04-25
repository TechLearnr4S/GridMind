from __future__ import annotations

import copy
import numpy as np
from dataclasses import dataclass
from typing import List


# ──────────────────────────────────────────────────────────────────────────────
# Structured I/O dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GridAction:
    """Structured action passed into GridOpsEnv.step().

    Parameters
    ----------
    allocation : list[float]
        Power bid per zone. Length must equal num_zones.
    """
    allocation: List[float]

    def to_numpy(self) -> np.ndarray:
        """Convert to a float32 numpy array for internal env use."""
        return np.array(self.allocation, dtype=np.float32)


@dataclass
class GridObservation:
    """Structured observation returned by GridOpsEnv._get_obs().

    All list fields contain plain Python floats so the output is
    JSON-serialisable without a custom encoder.

    Parameters
    ----------
    demand     : per-zone demand (float list)
    supply     : per-zone allocated power (float list)
    reputation : per-zone reputation score (float list)
    faults     : per-zone binary fault flag (float list, 0.0 or 1.0)
    time_step  : current simulation timestep (int)
    """
    demand:     List[float]
    supply:     List[float]
    reputation: List[float]
    faults:     List[float]
    time_step:  int

    def to_dict(self) -> dict:
        """Return a JSON-safe dict (no numpy arrays)."""
        return {
            "time_step":  self.time_step,
            "demand":     self.demand,
            "supply":     self.supply,
            "reputation": self.reputation,
            "faults":     self.faults,
        }


class GridOpsEnv:
    """
    Multi-zone power grid environment with bidding-based allocation.
    Gym-style API: reset() → (obs, info), step() → (obs, reward, terminated, truncated, info).

    Features
    --------
    - Negotiation / coalition detection
    - Strategic misreporting detection (scale-correct)
    - Delayed cascading failures (failure queue)
    - Long-horizon memory (rolling 10-step window)
    - float32 internal arrays for numerical stability
    - Smoothed reward with component tracking
    - Optional debug assertions
    """

    metadata = {
        "name": "GridOps++",
        "description": "Multi-agent power grid coordination under uncertainty",
        "capabilities": [
            "multi-agent coordination",
            "long-horizon planning",
            "reward alignment",
            "uncertainty reasoning"
        ]
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_zones: int = 3,
        max_steps: int = 50,
        seed=None,
        mode: str = "baseline",
        debug: bool = False,
    ):
        self.num_zones   = num_zones
        self.max_steps   = max_steps
        self.mode        = mode        # "baseline" | "selfish" | "coordinated" | "advanced"
        self.reward_mode = "local"     # "local" | "global"
        self.debug       = debug

        # Reputation hyperparameters
        self.rep_decay   = 0.1
        self.rep_recover = 0.02
        self.rep_min     = 0.2
        self.rep_max     = 2.0

        # Ethical prioritisation weight (priority-3 zones in global mode)
        self.ethical_weight = 1.5

        # Coalition bonus threshold and magnitude
        self.coalition_var_threshold = 0.05
        self.coalition_bonus_value   = 2.0

        # Transient state — initialised properly in reset()
        self.prev_reward       = 0.0
        self.reward_components = {"served": 0.0, "blackout": 0.0, "stability": 0.0, "honesty": 0.0}

        self.seed(seed)

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_mode(self, mode: str):
        self.mode = mode

    def set_reward_mode(self, mode: str):
        self.reward_mode = mode

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Action Validation
    # ------------------------------------------------------------------

    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """Sanitise and normalise an external action vector."""
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        action = np.clip(action, 0.0, 1.0)
        if np.sum(action) <= 0:
            action = np.ones(self.num_zones, dtype=np.float32) / self.num_zones
        return action.astype(np.float32)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)

        self.time_step   = 0
        self.prev_reward = 0.0
        self.reward_components = {
            "served": 0.0, "blackout": 0.0, "stability": 0.0, "honesty": 0.0
        }
        self.episode_stats = {
            "total_reward": 0.0,
            "total_blackouts": 0,
            "avg_stability": [],
            "misreport_events": 0
        }
        self.reputation = np.ones(self.num_zones, dtype=np.float32)
        self._init_state()
        return self._get_obs(), {}

    def step(self, action=None):
        # ── 0) Resolve / sanitise action ──────────────────────────────
        if action is not None:
            if isinstance(action, GridAction):
                action = action.to_numpy()
            action = np.asarray(action, dtype=np.float32)
            action = self._validate_action(action)

        if action is None:
            if self.mode == "selfish":
                action = self.selfish_policy()
            elif self.mode == "advanced":
                action = self.advanced_policy()
            elif self.mode == "coordinated":
                action = self.coordinated_policy()
            else:
                raw = self.rng.integers(1, 10, size=self.num_zones).astype(np.float32)
                total = raw.sum() + 1e-8
                action = (raw / total).astype(np.float32)

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.num_zones,):
            raise ValueError(
                f"Expected action shape ({self.num_zones},), got {action.shape}"
            )

        # Hardened inf/nan guard
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0)
        action = np.maximum(action, 0.0)

        # ── 1) Misreporting / honesty check (scale-correct: >1.3× demand) ──
        bids = action.copy()
        misreport_ratio          = bids / (self.demand + 1e-8)
        self._misreport_ratio    = np.clip(misreport_ratio, 0.0, 10.0)
        self._misreport_mask     = misreport_ratio > 1.3          # scale-correct threshold

        overbid     = bids > (1.3 * self.demand)
        honesty_pen = (overbid.astype(np.float32) + self._misreport_mask.astype(np.float32)) * 2.0

        self.reputation = np.clip(
            self.reputation
            - self.rep_decay   * overbid.astype(np.float32)
            - self.rep_decay   * self._misreport_mask.astype(np.float32)
            + self.rep_recover,
            self.rep_min, self.rep_max,
        ).astype(np.float32)
        self.reputation = np.maximum(self.reputation, 0.0)  # clamp ≥ 0

        self._honesty_pen  = honesty_pen
        self._overbid_mask = overbid

        # ── 2) Coalition detection ─────────────────────────────────────
        bid_norm = bids / (np.max(bids) + 1e-8)
        self._coalition_active = bool(np.var(bid_norm) < self.coalition_var_threshold)
        self._coalition_bonus  = self.coalition_bonus_value if self._coalition_active else 0.0

        # ── 3) Safe allocation with conservation guarantee ─────────────
        total_action = float(np.sum(action)) + 1e-8
        allocation   = (action / total_action).astype(np.float32)
        self.allocated = (allocation * float(self.total_power)).astype(np.float32)

        # Assert power conservation (debug)
        alloc_sum = float(np.sum(self.allocated))
        if abs(alloc_sum - self.total_power) > 1e-3:
            # Correct floating-point drift
            self.allocated = (self.allocated / (alloc_sum + 1e-8) * self.total_power).astype(np.float32)

        # Advanced mode: bias allocation toward high-demand zones
        if self.mode == "advanced":
            weights = self.demand / (np.sum(self.demand) + 1e-8)
            self.allocated = (0.8 * self.allocated + 0.2 * weights * self.total_power).astype(np.float32)
            # Re-normalise to conserve total_power
            alloc_sum = float(np.sum(self.allocated))
            if alloc_sum > 1e-8:
                self.allocated = (self.allocated / alloc_sum * self.total_power).astype(np.float32)

        if np.any(np.isnan(self.allocated)):
            raise RuntimeError("Invalid allocation detected: NaN in self.allocated")

        # ── 4) Advance time and dynamics ──────────────────────────────
        self.time_step += 1
        self._dynamics()
        self._local_rewards = self._compute_local_rewards()

        # ── 5) Delayed failure queue ───────────────────────────────────
        delayed_triggered = self._process_failure_queue()
        self._delayed_failures_triggered = delayed_triggered

        # ── 6) Memory update ──────────────────────────────────────────
        self._update_memory()

        # ── 7) Reward (clipped + smoothed) ────────────────────────────
        raw_reward = self._compute_reward()
        reward     = float(np.clip(raw_reward, -5.0, 5.0))
        if not np.isfinite(reward):
            reward = 0.0
            
        reward = float(np.tanh(reward))
        
        # Penalize large reward jumps
        delta = reward - self.prev_reward
        reward -= 0.1 * abs(delta)
        self.prev_reward = reward

        obs        = self._get_obs()
        terminated = bool(self.time_step >= self.max_steps)
        truncated  = False
        info       = self._get_info()
        info["reward_components"] = {k: float(v) for k, v in self.reward_components.items()}

        info["global_score"] = float(reward)

        self.episode_stats["total_reward"] += reward
        self.episode_stats["total_blackouts"] += int(info["blackouts"])
        self.episode_stats["avg_stability"].append(float(info["stability_score"]))
        self.episode_stats["misreport_events"] += int(info["honesty_violations"])

        if terminated or truncated:
            info["episode_summary"] = {
                "total_reward": float(self.episode_stats["total_reward"]),
                "avg_stability": float(np.mean(self.episode_stats["avg_stability"])) if self.episode_stats["avg_stability"] else 0.0,
                "total_blackouts": int(self.episode_stats["total_blackouts"]),
                "misreport_rate": float(self.episode_stats["misreport_events"] / (self.time_step * self.num_zones))
            }

        # ── 8) Debug assertions ────────────────────────────────────────
        if self.debug:
            assert not np.any(np.isnan(self.demand)),    "NaN in demand"
            assert not np.any(np.isnan(self.allocated)), "NaN in allocated"
            assert not np.any(np.isnan(self.reputation)),"NaN in reputation"

        # ── 9) History append ─────────────────────────────────────────
        self.history["reward"].append(reward)
        self.history["blackouts"].append(info["blackouts"])
        self.history["overloads"].append(info["overloads"])
        self.history["imbalance"].append(info["imbalance"])
        self.history["efficiency"].append(info["efficiency"])
        self.history["stability"].append(info["stability_score"])
        self.history["avg_reputation"].append(info["avg_reputation"])
        self.history["honesty_violations"].append(info["honesty_violations"])
        self.history["misreporting_rate"].append(info["misreport_rate"])
        self.history["coalition_rate"].append(float(self._coalition_active))
        self.history["delayed_failures"].append(info["delayed_failures_triggered"])
        self.history["faults"].append(self.failed.tolist())
        self.history["allocations"].append(self.allocated.tolist())

        for k in self.history:
            self.history[k] = self.history[k][-3:]

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # State Initialisation
    # ------------------------------------------------------------------

    def _init_state(self):
        # float32 arrays throughout
        self.demand = self.rng.uniform(0.3, 2.0, size=self.num_zones).astype(np.float32)
        self.demand = np.maximum(self.demand, 0.1).astype(np.float32)  # clamp

        supply_scale     = float(self.rng.uniform(0.8, 1.2))
        self.total_power = float(self.demand.mean() * self.num_zones * supply_scale)
        self.total_power = max(self.total_power, 0.1)

        self.allocated = np.zeros(self.num_zones, dtype=np.float32)
        self.priority  = self.rng.integers(1, 4, size=self.num_zones)   # {1,2,3}
        self.failed    = np.zeros(self.num_zones, dtype=bool)

        self.failure_queue = []

        self.memory = {
            "recent_rewards":   [],
            "recent_blackouts": [],
            "recent_imbalance": [],
            "summary":          {},
        }

        self.history = {
            "reward": [], "blackouts": [], "overloads": [],
            "imbalance": [], "efficiency": [], "stability": [],
            "avg_reputation": [], "honesty_violations": [],
            "misreporting_rate": [], "coalition_rate": [],
            "delayed_failures": [], "faults": [], "allocations": [],
        }

        # Pre-initialise masks so reward/info calls are safe before first step
        n = self.num_zones
        self._overload_mask              = np.zeros(n, dtype=bool)
        self._blackout_mask              = np.zeros(n, dtype=bool)
        self._local_rewards              = np.zeros(n, dtype=np.float32)
        self._honesty_pen                = np.zeros(n, dtype=np.float32)
        self._overbid_mask               = np.zeros(n, dtype=bool)
        self._misreport_mask             = np.zeros(n, dtype=bool)
        self._misreport_ratio            = np.ones(n,  dtype=np.float32)
        self._coalition_active           = False
        self._coalition_bonus            = 0.0
        self._delayed_failures_triggered = 0

    # ------------------------------------------------------------------
    # Stochastic Dynamics + Failure Queue
    # ------------------------------------------------------------------

    def _dynamics(self):
        # ±15% fractional noise — preserves float32 range, stays proportional
        noise      = self.rng.uniform(-0.15, 0.15, size=self.num_zones).astype(np.float32)
        self.demand = np.maximum((self.demand + noise * self.demand), 0.1).astype(np.float32)

        # Fault masks
        self._overload_mask = self.allocated > 1.3 * self.demand
        self._blackout_mask = self.allocated < 0.4 * self.demand

        self.failed = np.logical_or(self.failed, self._overload_mask)

        # Queue delayed failures
        for i in np.where(self._overload_mask)[0]:
            delay      = int(self.rng.integers(1, 4))
            power_loss = float(self.total_power * 0.05)
            self.failure_queue.append({"delay": delay, "power_loss": power_loss})

        # Cascade / recovery — always float
        max_power = float(self.num_zones * 2.0 * 1.2)
        if self._overload_mask.any():
            self.total_power = max(float(self.total_power * 0.9), 0.1)
        else:
            self.total_power = min(float(self.total_power * 1.02), max_power)

    def _process_failure_queue(self) -> int:
        """Decrement delays; apply losses when delay hits 0. Returns count triggered."""
        remaining = []
        triggered = 0
        for event in self.failure_queue:
            event["delay"] -= 1
            if event["delay"] <= 0:
                self.total_power = max(float(self.total_power - event["power_loss"]), 0.1)
                triggered += 1
            else:
                remaining.append(event)
        self.failure_queue = remaining
        return triggered

    # ------------------------------------------------------------------
    # Long-Horizon Memory
    # ------------------------------------------------------------------

    def _update_memory(self):
        reward_now   = self.history["reward"][-1]    if self.history["reward"]    else 0.0
        blackout_now = self.history["blackouts"][-1] if self.history["blackouts"] else 0.0
        imbal_now    = self.history["imbalance"][-1] if self.history["imbalance"] else 0.0

        self.memory["recent_rewards"].append(reward_now)
        self.memory["recent_blackouts"].append(blackout_now)
        self.memory["recent_imbalance"].append(imbal_now)

        for key in ("recent_rewards", "recent_blackouts", "recent_imbalance"):
            self.memory[key] = self.memory[key][-10:]

        if self.time_step % 10 == 0:
            self.memory["summary"] = {
                "avg_reward":      float(np.mean(self.memory["recent_rewards"])),
                "blackout_rate":   float(np.mean(self.memory["recent_blackouts"])),
                "imbalance_trend": float(np.mean(self.memory["recent_imbalance"])),
            }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        served       = np.minimum(self.allocated, self.demand)

        fault_count = float(self.failed.sum())
        misreport_rate = float(self._misreport_mask.mean())

        served_norm    = float(served.sum() / (self.demand.sum() + 1e-8))
        blackout_norm  = float(fault_count / self.num_zones)
        weighted       = self.allocated * self.demand
        stability_norm = 1.0 - np.std(weighted) / (np.mean(weighted) + 1e-8)
        stability_norm = float(np.clip(stability_norm, 0.0, 1.0))
        honesty_norm   = misreport_rate

        self.reward_components = {
            "served":    float(served_norm),
            "blackout":  float(blackout_norm),
            "stability": float(stability_norm),
            "honesty":   float(honesty_norm),
        }

        progress = min(0.5, self.time_step / self.max_steps)
        w_served = 1.5 + 0.5 * progress
        w_stability = 1.0 + 0.5 * progress

        core = (
            w_served * served_norm
            - 2.0 * blackout_norm
            + w_stability * stability_norm
            - 2.0 * honesty_norm
        )

        reward = 0.0

        threshold = 1.0 * np.mean(self.allocated)
        if np.std(self.allocated) > threshold:
            reward -= 0.5

        if fault_count > 0:
            reward -= 0.2 * fault_count

        if len(self.history["faults"]) >= 2:
            past_faults = np.array(self.history["faults"][-2])
            if np.any(past_faults) and np.any(self._blackout_mask):
                reward -= 0.3

        if fault_count == 0:
            reward += 0.02
        if np.any(self._overload_mask):
            reward -= 0.02
        if np.std(self.allocated) < 0.1 * np.mean(self.allocated):
            reward += 0.01

        temporal_bonus = 0.0
        if len(self.history["allocations"]) > 0:
            prev_alloc = np.array(self.history["allocations"][-1])
            change = float(np.mean(np.abs(self.allocated - prev_alloc)))
            temporal_bonus = 0.03 * float(1.0 - np.clip(change, 0.0, 1.0))

        # Fairness reward
        spread = (np.max(self.allocated) - np.min(self.allocated)) / (np.mean(self.allocated) + 1e-8)
        fairness_bonus = 0.03 * float(np.clip(1.0 - spread, 0.0, 1.0))

        # Priority-aware reward
        priority_score = np.sum(self.allocated * self.priority) / (np.sum(self.priority) + 1e-8)
        priority_bonus = 0.1 * float(priority_score)

        bonus = temporal_bonus + fairness_bonus + priority_bonus
        bonus = float(np.clip(bonus, 0.0, 0.2))
        reward += bonus

        # Advanced-mode exclusive bonuses
        if self.mode == "advanced":
            reward += 0.15 * float(priority_score)
            reward += 0.05 * float(stability_norm)
            rep_factor = float(np.mean(self.reputation))
            reward += 0.05 * rep_factor

        reward = float(np.clip(0.8 * core + 0.2 * reward, -5.0, 5.0))
        assert -5.0 <= reward <= 5.0, f"Reward out of bounds: {reward}"
        return reward

    def _compute_local_rewards(self) -> np.ndarray:
        return np.minimum(self.allocated, self.demand).astype(np.float32)

    # ------------------------------------------------------------------
    # Observation & Info
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        """Return a GridObservation as a JSON-safe dict."""
        obs = GridObservation(
            demand     = self.demand.tolist(),
            supply     = self.allocated.tolist(),
            reputation = self.reputation.tolist(),
            faults     = self.failed.astype(float).tolist(),
            time_step  = int(self.time_step),
        ).to_dict()

        obs["priority"]    = self.priority.tolist()
        obs["total_power"] = float(self.total_power)
        if self.memory["summary"]:
            obs["memory_summary"] = {
                k: float(v) for k, v in self.memory["summary"].items()
            }
        return obs

    def _get_info(self) -> dict:
        alloc     = self.allocated
        share     = alloc / (alloc.sum() + 1e-8)
        imbalance = float(np.var(share))

        served    = np.minimum(alloc, self.demand)
        overloads = int(self._overload_mask.sum())
        blackouts = int(self._blackout_mask.sum())

        if self.reward_mode == "local" and self.mode == "selfish":
            blackout_pen = 2.0 * blackouts
        else:
            blackout_pen = 5.0 * blackouts

        demand_gap = float(np.sum(np.maximum(0.0, self.demand - alloc)))

        return {
            "served":                     float(served.sum()),
            "weighted_served":            float(np.sum(served * self.priority)),
            "overloads":                  overloads,
            "blackouts":                  blackouts,
            "blackout_penalty":           float(blackout_pen),
            "fault_count":                int(self.failed.sum()),
            "demand_supply_gap":          float(demand_gap),
            "avg_reputation":             float(self.reputation.mean()),
            "min_reputation":             float(self.reputation.min()),
            "misreport_rate":             float(self._misreport_mask.mean()),
            "honesty_violations":         int(self._overbid_mask.sum()),
            "misreporting_rate":          float(self._misreport_mask.mean()),  # legacy alias
            "coalition_rate":             float(self._coalition_active),
            "local_rewards":              [float(v) for v in self._local_rewards],
            "imbalance":                  imbalance,
            "efficiency":                 float(served.sum() / (self.demand.sum() + 1e-8)),
            "fairness_penalty":           float(3.0 * imbalance),
            "stability_score":            float(1.0 / (1 + blackouts + overloads)),
            "delayed_failures_triggered": int(self._delayed_failures_triggered),
            "reward_components":          {k: float(v) for k, v in self.reward_components.items()},
            "reward_explanation": {
                "served_high": bool(self.reward_components.get("served", 0.0) > 0.8),
                "no_faults": bool(int(self.failed.sum()) == 0),
                "stable": bool(self.reward_components.get("stability", 0.0) > 0.7),
                "honest": bool(self.reward_components.get("honesty", 0.0) < 0.1)
            }
        }

    # ------------------------------------------------------------------
    # Policies  (all outputs: non-negative, finite, normalised)
    # ------------------------------------------------------------------

    def _safe_normalise(self, action: np.ndarray) -> np.ndarray:
        """Ensure action is non-negative, finite, and normalised to sum=1."""
        action = np.nan_to_num(action.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        action = np.maximum(action, 0.0)
        if np.sum(action) <= 0:
            action = np.ones(self.num_zones, dtype=np.float32)
        return (action / (np.sum(action) + 1e-8)).astype(np.float32)

    def selfish_policy(self) -> np.ndarray:
        """Overreport demand by alpha + fractional noise. Output is normalised."""
        alpha = 1.2
        noise = self.rng.uniform(0.0, 0.2, size=self.num_zones).astype(np.float32) * self.demand
        bid   = self.demand * self.priority.astype(np.float32) * alpha + noise
        return self._safe_normalise(bid)

    def coordinated_policy(self) -> np.ndarray:
        weights = self.priority.astype(np.float32) * self.demand * self.reputation
        return self._safe_normalise(weights)

    def advanced_policy(self) -> np.ndarray:
        """Reputation² weighting — amplifies trust differential."""
        weights = self.priority.astype(np.float32) * self.demand * (self.reputation ** 2)
        return self._safe_normalise(weights)

    # ------------------------------------------------------------------
    # History / Summary / State access
    # ------------------------------------------------------------------

    def get_history(self) -> dict:
        return self.history

    def get_state(self) -> dict:
        """Return deep copies of all core state fields as a JSON-safe dict."""
        return {
            "demand":     copy.deepcopy(self.demand).tolist(),
            "supply":     copy.deepcopy(self.allocated).tolist(),
            "reputation": copy.deepcopy(self.reputation).tolist(),
            "faults":     copy.deepcopy(self.failed).astype(float).tolist(),
            "time_step":  int(self.time_step),
        }

    def summarize_episode(self) -> dict:
        return {
            "avg_reward":         float(np.mean(self.history["reward"])),
            "avg_blackouts":      float(np.mean(self.history["blackouts"])),
            "avg_imbalance":      float(np.mean(self.history["imbalance"])),
            "avg_stability":      float(np.mean(self.history["stability"])),
            "avg_reputation":     float(np.mean(self.history["avg_reputation"])),
            "honesty_violations": int(np.sum(self.history["honesty_violations"])),
            "misreporting_rate":  float(np.mean(self.history["misreporting_rate"])),
            "coalition_rate":     float(np.mean(self.history["coalition_rate"])),
            "delayed_failures":   int(np.sum(self.history["delayed_failures"])),
        }

    def get_task_spec(self) -> dict:
        return {
            "goal": "Maximize served demand while minimizing blackouts and misreporting",
            "constraints": [
                "limited total power",
                "partial observability",
                "delayed cascade failures"
            ],
            "metrics": [
                "reward",
                "stability",
                "misreport_rate",
                "blackouts"
            ]
        }

    def compare_to_baseline(self, baseline_reward: float) -> dict:
        return {
            "improvement": float(self.prev_reward - baseline_reward)
        }

    def describe_mode(self) -> dict:
        return {
            "baseline": "no coordination",
            "selfish": "local optimization",
            "coordinated": "global alignment",
            "advanced": "reputation + coalition + stability"
        }

    def explain_step(self) -> dict:
        return {
            "what_happened": "allocation, faults, reward change",
            "why": "overload / coordination / misreport"
        }


# ----------------------------------------------------------------------
# Behavioural validation
# ----------------------------------------------------------------------

if __name__ == "__main__":
    configs = [
        ("baseline",    "local"),
        ("selfish",     "global"),
        ("coordinated", "global"),
    ]

    results = []
    for mode, reward_mode in configs:
        env = GridOpsEnv(num_zones=3, max_steps=50, seed=42, mode=mode, debug=True)
        env.set_reward_mode(reward_mode)
        env.reset()

        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step()

        summary = env.summarize_episode()
        results.append((mode, reward_mode, summary))

    keys = [
        ("avg_reward",        "avg_reward"),
        ("avg_blackouts",     "blackouts"),
        ("avg_stability",     "stability"),
        ("avg_reputation",    "reputation"),
        ("misreporting_rate", "misreport"),
        ("coalition_rate",    "coalition"),
        ("delayed_failures",  "delayed_fail"),
    ]
    col = 14
    header = f"{'mode':<{col}} {'reward_mode':<{col}}" + "".join(
        f" {label:>{col}}" for _, label in keys
    )
    print(header)
    print("-" * len(header))
    for mode, reward_mode, s in results:
        row = f"{mode:<{col}} {reward_mode:<{col}}"
        for key, _ in keys:
            val = s[key]
            row += f" {val:>{col}.3f}" if isinstance(val, float) else f" {val:>{col}d}"
        print(row)
