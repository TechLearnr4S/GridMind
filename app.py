import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from train.train import GridOpsEnvWrapper
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_contrib import RecurrentPPO
    DEPENDENCIES_LOADED = True
except ImportError:
    DEPENDENCIES_LOADED = False

MODEL_PATH = "models/ppo_lstm_final"
VECNORM_PATH = "models/vecnormalize_lstm_final.pkl"

ZONE_NAMES = ["Residential", "Commercial", "Hospital"]

def _extract_array_obs(obs, length=3):
    """Safely extract a flat numeric array from obs (dict or ndarray)."""
    try:
        if isinstance(obs, dict):
            for key in ("observation", "obs", "state", list(obs.keys())[0]):
                if key in obs:
                    arr = np.asarray(obs[key]).flatten()
                    return arr[:length].tolist() if len(arr) >= length else [0.33] * length
        arr = np.asarray(obs).flatten()
        return arr[:length].tolist() if len(arr) >= length else [0.33] * length
    except Exception:
        return [0.33] * length

def _heuristic_action(demand):
    """
    Deterministic heuristic: allocate proportionally to demand,
    with a 20 % bonus weight on Zone 3 (Hospital / Critical).
    """
    weights = [demand[0], demand[1], demand[2] * 1.2]
    total = sum(weights)
    if total <= 0:
        return np.array([[0.25, 0.35, 0.40]])
    return np.array([[w / total for w in weights]])

class GridSimulator:
    def __init__(self):
        self.ready = False
        self.obs = None
        if not DEPENDENCIES_LOADED:
            return

        self.env = DummyVecEnv([lambda: GridOpsEnvWrapper()])
        if os.path.exists(VECNORM_PATH):
            self.env = VecNormalize.load(VECNORM_PATH, self.env)
        self.env.training = False
        self.env.norm_reward = False

        if os.path.exists(MODEL_PATH + ".zip"):
            self.model = RecurrentPPO.load(MODEL_PATH, env=self.env)
            self.ready = True
            self.reset()
        else:
            # Demo mode: heuristic only, no model needed
            self.model = None
            self.ready = True
            self.reset()

    def reset(self):
        if DEPENDENCIES_LOADED and hasattr(self, "env"):
            self.obs = self.env.reset()
        else:
            self.obs = np.array([[0.33, 0.33, 0.33, 0.33, 0.33, 0.33]])
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.done = False
        self.steps = 0
        self.total_reward = 0.0
        self.blackouts = 0
        self.stability_history = []
        self.stability = 1.0
        self.current_demand = [0.33, 0.33, 0.33]
        self.current_supply = [0.33, 0.33, 0.33]
        self.action_taken = [0.33, 0.33, 0.33]
        self.fault_status = [False, False, False]
        self.explanation = "System reset. Ready for AI power allocation."

    def step(self, action=None, manual=False):
        if self.obs is None:
            return self._safe_error_state("⚠️ Please click Reset before running the AI")

        if self.done:
            return self.get_ui_state()

        if not manual:
            if self.model is not None:
                action, self.lstm_states = self.model.predict(
                    self.obs,
                    state=self.lstm_states,
                    episode_start=self.episode_starts,
                    deterministic=True,
                )
            else:
                demand = _extract_array_obs(self.obs[0] if hasattr(self.obs, '__len__') else self.obs)
                action = _heuristic_action(demand)
        else:
            total = sum(action)
            action = np.array([[a / total if total > 0 else 0.33 for a in action]])

        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated | truncated
        else:
            obs, reward, done, info = result

        self.obs = obs
        self.episode_starts = done.copy()

        ep_info = info[0] if isinstance(info, (list, tuple)) else info

        self.steps += 1
        self.total_reward += float(reward[0]) if hasattr(reward, '__len__') else float(reward)
        self.done = bool(done[0]) if hasattr(done, '__len__') else bool(done)
        self.action_taken = action[0].tolist() if hasattr(action[0], 'tolist') else list(action[0])

        for key in ("blackout_count", "blackouts", "fault_count"):
            if ep_info.get(key):
                self.blackouts += int(ep_info[key])
                break

        self.stability = float(ep_info.get("stability_score", ep_info.get("stability", 1.0)))
        self.stability_history.append(self.stability)

        try:
            raw_obs = self.env.get_original_obs() if hasattr(self.env, 'get_original_obs') else self.obs
            raw = raw_obs[0] if hasattr(raw_obs, '__len__') else raw_obs
            self.current_demand = _extract_array_obs(raw, 3)
        except Exception:
            self.current_demand = [0.33, 0.33, 0.33]

        self.current_supply = self.action_taken[:3]
        self.generate_explanation()
        return self.get_ui_state()

    def generate_explanation(self):
        z1, z2, z3 = self.action_taken[:3]
        d1, d2, d3 = self.current_demand[:3]
        imbalance = [abs(self.action_taken[i] - self.current_demand[i]) for i in range(3)]
        max_imb_zone = ZONE_NAMES[imbalance.index(max(imbalance))]

        if self.blackouts > 0 and self.steps == len(self.stability_history):
            self.explanation = (
                f"⚠️ Blackout detected! AI is aggressively rerouting power "
                f"to restore stability and prevent cascading failures."
            )
            return

        if z3 > 0.45 or d3 > 0.5:
            self.explanation = (
                f"AI prioritized the Hospital (Zone 3) — demand at {d3:.0%}. "
                f"Critical load protection activated to prevent life-safety failures."
            )
        elif z3 > z1 and z3 > z2:
            self.explanation = (
                f"AI is routing extra capacity to the Hospital ({z3:.0%} share) "
                f"to maintain critical-infrastructure uptime."
            )
        elif z1 > z2 and z1 > z3:
            self.explanation = (
                f"AI shifted surplus to the Residential Zone ({z1:.0%} share) "
                f"to balance baseline load and prevent localized faults."
            )
        elif z2 > z1 and z2 > z3:
            self.explanation = (
                f"AI allocated maximum capacity to the Commercial Zone ({z2:.0%} share) "
                f"to stabilize an immediate demand spike."
            )
        else:
            self.explanation = (
                "AI balanced power evenly across all zones — "
                "grid is stable, no critical hotspots detected."
            )

        if max(imbalance) > 0.25:
            self.explanation += f" Largest imbalance in {max_imb_zone} zone; AI is adapting."

        if self.done:
            self.explanation += " Episode complete."

    def get_ui_state(self):
        stab = self.stability
        blk = self.blackouts

        stab_icon = "🟢" if stab >= 0.75 else ("🟡" if stab >= 0.45 else "🔴")
        stab_str = f"{stab:.2f} / 1.00  {stab_icon}"

        blk_icon = "🟢" if blk == 0 else ("🟡" if blk <= 2 else "🔴")
        blk_str = f"{int(blk)}  {blk_icon}"

        rew_str = f"{self.total_reward:.1f}"
        step_str = str(self.steps)

        recent = self.stability_history[-5:] if self.stability_history else [stab]
        trend_str = " → ".join(f"{v:.2f}" for v in recent)
        expl_with_trend = f"{self.explanation}\n\n📈 Stability Trend: {trend_str}"

        fig_trend, ax_trend = plt.subplots(figsize=(6, 3))
        history = self.stability_history if self.stability_history else [stab]
        ax_trend.plot(history, color="#10b981", linewidth=2, marker="o", markersize=3)
        ax_trend.set_title("Grid Stability Over Time", fontsize=10, pad=10)
        ax_trend.set_ylim(0, 1.1)
        ax_trend.axhline(0.75, color="#f59e0b", linewidth=1, linestyle="--", alpha=0.6, label="Warning")
        ax_trend.axhline(0.45, color="#ef4444", linewidth=1, linestyle="--", alpha=0.6, label="Critical")
        ax_trend.legend(fontsize=7, loc="lower right")
        ax_trend.grid(True, linestyle="--", alpha=0.3)
        ax_trend.spines['top'].set_visible(False)
        ax_trend.spines['right'].set_visible(False)
        plt.tight_layout()

        fig_grid, ax_grid = plt.subplots(figsize=(6, 3))
        x = np.arange(len(ZONE_NAMES))
        width = 0.35

        demand_vals = [min(max(float(v), 0), 1) for v in self.current_demand]
        supply_vals = [min(max(float(v), 0), 1) for v in self.current_supply]

        ax_grid.bar(x - width / 2, demand_vals, width, label="Demand", color="#ef4444", alpha=0.85)
        ax_grid.bar(x + width / 2, supply_vals, width, label="Supply (AI)", color="#3b82f6", alpha=0.85)

        ax_grid.set_title("Demand vs. Supply Allocation", fontsize=10, pad=10)
        ax_grid.set_xticks(x)
        ax_grid.set_xticklabels(ZONE_NAMES)
        ax_grid.set_ylim(0, 1.15)
        ax_grid.legend(loc="upper right", fontsize=8)
        ax_grid.grid(axis="y", linestyle="--", alpha=0.3)
        ax_grid.spines['top'].set_visible(False)
        ax_grid.spines['right'].set_visible(False)
        plt.tight_layout()

        return stab_str, blk_str, rew_str, step_str, expl_with_trend, fig_trend, fig_grid

    def _safe_error_state(self, message):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        return ("—", "—", "—", "—", message, fig, fig)

sim = GridSimulator()
_ERR = ("Error", "Error", "Error", "Error", "Model not loaded.", None, None)

def ui_reset():
    if not sim.ready:
        return _ERR
    sim.reset()
    return sim.get_ui_state()

def ui_ai_step():
    if not sim.ready:
        return _ERR
    if sim.obs is None:
        return sim._safe_error_state("⚠️ Please click Reset before running the AI")
    return sim.step()

def ui_auto_run():
    if not sim.ready:
        yield _ERR
        return
    if sim.obs is None:
        yield sim._safe_error_state("⚠️ Please click Reset before running the AI")
        return

    while not sim.done and sim.steps < 100:
        yield sim.step()
        time.sleep(0.1)

def ui_manual_step(z1, z2, z3):
    if not sim.ready:
        return _ERR
    if sim.obs is None:
        return sim._safe_error_state("⚠️ Please click Reset before running the AI")
    return sim.step(action=[z1, z2, z3], manual=True)

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            """
            # ⚡ GridMind: AI for Power Grid Stability
            ### AI dynamically allocates power to prevent cascading blackouts in real-time
            **Goal:** Stability ↑ &nbsp;|&nbsp; Blackouts ↓ &nbsp;|&nbsp; Hospital always protected
            """
        )

    with gr.Row():
        btn_reset = gr.Button("🔄 Reset", variant="secondary", size="lg")
        btn_ai_step = gr.Button("🤖 AI Step", variant="primary", size="lg")
        btn_auto = gr.Button("▶️ Auto Run", variant="primary", size="lg")

    with gr.Row():
        kpi_stability = gr.Textbox(label="⚡ Grid Stability", value="1.00 / 1.00  🟢", interactive=False)
        kpi_blackouts = gr.Textbox(label="🚨 Blackouts", value="0  🟢", interactive=False)
        kpi_reward = gr.Textbox(label="🏆 Total Reward", value="0.0", interactive=False)
        kpi_steps = gr.Textbox(label="⏱️ Steps", value="0", interactive=False)

    with gr.Row():
        explanation = gr.Textbox(
            label="🤖 AI Decision & Stability Trend",
            value="System ready. Click Reset then AI Step (or Auto Run) to begin.",
            interactive=False,
            lines=4,
            max_lines=4,
        )

    with gr.Row():
        plot_trend = gr.Plot(label="📈 Live Stability Trend")
        plot_grid = gr.Plot(label="📊 Demand vs Supply")

    with gr.Accordion("🎮 Manual Control — Can you beat the AI?", open=False):
        gr.Markdown(
            "Manually allocate power across zones. Values are auto-normalised to sum to 1."
        )
        with gr.Row():
            slider_z1 = gr.Slider(0, 1, value=0.33, step=0.01, label="Zone 1 — Residential")
            slider_z2 = gr.Slider(0, 1, value=0.33, step=0.01, label="Zone 2 — Commercial")
            slider_z3 = gr.Slider(0, 1, value=0.34, step=0.01, label="Zone 3 — Hospital (Critical)")
        btn_manual_step = gr.Button("🎮 Manual Step", variant="secondary")

    outputs = [kpi_stability, kpi_blackouts, kpi_reward, kpi_steps, explanation, plot_trend, plot_grid]

    btn_reset.click(fn=ui_reset, outputs=outputs)
    btn_ai_step.click(fn=ui_ai_step, outputs=outputs)
    btn_auto.click(fn=ui_auto_run, outputs=outputs)
    btn_manual_step.click(fn=ui_manual_step, inputs=[slider_z1, slider_z2, slider_z3], outputs=outputs)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base())
