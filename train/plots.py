import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODES   = ["baseline", "selfish", "coordinated"]
COLORS  = {"baseline": "#e07b39", "selfish": "#c0392b", "coordinated": "#27ae60"}


# ----------------------------------------------------------------------
# Helper: mean curve across seeds
# ----------------------------------------------------------------------

def mean_curve(histories, mode, key):
    """Stack per-seed timeseries and return the element-wise mean."""
    arrays = [np.array(h[key]) for h in histories[mode]]
    min_len = min(len(a) for a in arrays)
    matrix = np.stack([a[:min_len] for a in arrays])
    return matrix.mean(axis=0)


# ----------------------------------------------------------------------
# Individual metric plots (one mode per file)
# ----------------------------------------------------------------------

def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_reward_curves(histories, save_dir):
    fig, ax = plt.subplots()
    for mode in MODES:
        curve = mean_curve(histories, mode, "reward")
        ax.plot(curve, label=mode, color=COLORS[mode])
    ax.set_title("Reward Over Time (mean across seeds)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend()
    _save(fig, os.path.join(save_dir, "reward_curve.png"))


def plot_blackouts(histories, save_dir):
    fig, ax = plt.subplots()
    for mode in MODES:
        curve = mean_curve(histories, mode, "blackouts")
        ax.plot(curve, label=mode, color=COLORS[mode])
    ax.set_title("Blackouts Over Time (mean across seeds)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Blackouts")
    ax.legend()
    _save(fig, os.path.join(save_dir, "blackouts.png"))


def plot_imbalance(histories, save_dir):
    fig, ax = plt.subplots()
    for mode in MODES:
        curve = mean_curve(histories, mode, "imbalance")
        ax.plot(curve, label=mode, color=COLORS[mode])
    ax.set_title("Grid Imbalance Over Time (mean across seeds)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Imbalance")
    ax.legend()
    _save(fig, os.path.join(save_dir, "imbalance.png"))


def plot_stability(histories, save_dir):
    fig, ax = plt.subplots()
    for mode in MODES:
        curve = mean_curve(histories, mode, "stability")
        ax.plot(curve, label=mode, color=COLORS[mode])
    ax.set_title("Grid Stability Over Time (mean across seeds)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Stability Score")
    ax.legend()
    _save(fig, os.path.join(save_dir, "stability.png"))


def plot_reputation(histories, save_dir):
    """Only generated when the history contains avg_reputation data."""
    if not histories["baseline"][0].get("avg_reputation"):
        return
    fig, ax = plt.subplots()
    for mode in MODES:
        curve = mean_curve(histories, mode, "avg_reputation")
        ax.plot(curve, label=mode, color=COLORS[mode])
    ax.set_title("Average Agent Reputation Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reputation")
    ax.legend()
    _save(fig, os.path.join(save_dir, "reputation.png"))


# ----------------------------------------------------------------------
# Comparison (hero) plot
# ----------------------------------------------------------------------

def plot_comparison(histories, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode in MODES:
        curve = mean_curve(histories, mode, "reward")
        ax.plot(curve, label=mode, color=COLORS[mode], linewidth=2)
    ax.set_title("Learning and Coordination Improves Grid Stability", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(save_dir, "comparison.png"))


# ----------------------------------------------------------------------
# Entrypoint called from train.py
# ----------------------------------------------------------------------

def generate_all_plots(histories, save_dir="plots"):
    print(f"\nGenerating plots in '{save_dir}/' …")
    os.makedirs(save_dir, exist_ok=True)
    plot_reward_curves(histories, save_dir)
    plot_blackouts(histories, save_dir)
    plot_imbalance(histories, save_dir)
    plot_stability(histories, save_dir)
    plot_reputation(histories, save_dir)
    plot_comparison(histories, save_dir)
    print("All plots saved.\n")


# ----------------------------------------------------------------------
# Stand-alone mode
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from train.train import run_all
    histories, _ = run_all()
    generate_all_plots(histories)
