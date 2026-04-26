# ⚡ GridMind: Teaching an AI to Prevent Power Grid Blackouts

> **OpenEnv Hackathon 2026 — India**  
> A reinforcement learning agent trained to manage power distribution across critical infrastructure and prevent cascading blackouts.

---

## 🔗 Submission Deliverables

| Deliverable | Link |
|---|---|
| 🤗 **HF Space (Live Demo)** | [TechLearnr4S/GridMind](https://huggingface.co/spaces/TechLearnr4S/GridMind) |
| 📓 **Training Notebook (Colab)** | [Open in Colab ↗](https://colab.research.google.com/drive/1EmmWb1ARTxdahHGn8u6k3Ak1Dh172tvc?usp=sharing) |
| 💻 **GitHub Repository** | [TechLearnr4S/GridMind](https://github.com/TechLearnr4S/GridMind) |
| 📹 **Demo Video / Blog Post** | _[Add your YouTube/HF blog link here]_ |

---

## 🚨 The Problem

Modern power grids run on razor-thin margins. When demand spikes — a heat wave, a factory coming online — operators have seconds to decide how to allocate limited power across zones.

**Make the wrong call and you trigger a cascade:**
- Overloaded lines trip automatic safeguards
- Each trip permanently destroys grid capacity
- A minor brownout becomes a city-wide, multi-day blackout

This isn't a hypothetical. It's what happened in Texas (2021), India (2012), and California (2020).

**GridMind trains an AI agent to make these decisions better than human heuristics.**

---

## 🌐 The Environment: GridOpsEnv

`GridOpsEnv` is a fully custom OpenEnv-compliant environment that simulates a 3-zone power grid under realistic stress conditions.

### What the Agent Sees (Observation Space)
```
{
  demand     : [float, float, float]   # per-zone power demand (volatile, ±40% per step)
  supply     : [float, float, float]   # current allocated power
  reputation : [float, float, float]   # zone trust score (penalizes misreporting)
  faults     : [float, float, float]   # accumulated damage (delays before blackout)
  time_step  : int                     # step in episode
}
```

### What the Agent Does (Action Space)
Allocate power across 3 zones: a continuous vector `[a1, a2, a3]` summing to 1.0.

### Zone Priorities
| Zone | Type | Priority |
|---|---|---|
| Zone 1 | Residential | Low |
| Zone 2 | Commercial | Medium |
| Zone 3 | Hospital / Critical | **High** |

### What Makes This Environment Hard
- **Delayed cascading failures**: An overload on step 2 may not trigger a blackout until step 5. The agent must *anticipate*, not react.
- **Reputation mechanics**: Zones that misreport demand are penalized in future allocation rounds.
- **Volatile demand**: Each step, demand spikes by up to 40% unpredictably.
- **Zone prioritization**: Serving a hospital and letting residential zones under-serve is rewarded; the reverse is heavily penalized.

### Reward Function
```
reward = served_reward
       - 6.0  × blackout_events          # heavy blackout penalty
       - 0.5  × system_risk              # global unmet demand penalty
       - 0.5  × instability_score        # overload accumulation penalty
       - 0.1  × fuel_overconsumption     # generator efficiency penalty
       + 2.0  × coalition_bonus          # if zones cooperate within 5% variance
```

### LLM-Native Training via `state_to_text()`

GridOpsEnv exposes `state_to_text()` — converting the grid state into a rich natural-language prompt. This enables direct LLM training with HF TRL / Unsloth without any custom wrappers:

```
=== POWER GRID STATE (Step 12/50 — 24% complete) ===

Grid Zones:
  Zone 1 [Commercial (medium)]: demand=0.974, supply=0.312, reputation=0.91, status=✅ Healthy
  Zone 2 [Residential (low)]:   demand=1.635, supply=0.445, reputation=1.00, status=⚠️ FAULT DETECTED
  Zone 3 [Hospital/Critical]:   demand=0.786, supply=0.243, reputation=1.00, status=✅ Healthy

Episode so far:
  Blackouts: 2  |  Total unmet demand: 3.241  |  Total reward: -48.20

Task: Allocate power to 3 zones as fractions summing to 1.0.
Priority: Serve Zone 3 (Hospital) first. Avoid overloads — they cascade into blackouts.
Reply with exactly 3 space-separated floats. Example: 0.20 0.30 0.50
```

The LLM reads this, reasons about fault status and priorities, and outputs `0.20 0.30 0.50`. The environment returns a reward. TRL GRPO updates the model. **This is end-to-end LLM training on a real-world problem.**



Each reward component is independently named, weighted, and inspectable via `GridMindRubric`:

```python
from env.gridops_env import GridMindRubric

rubric = GridMindRubric(served_reward=5.2, blackout_penalty=1.0, system_risk=2.3)
print(rubric.breakdown())
# → {'served_reward': 5.2, 'blackout_penalty': -6.0, 'system_risk': -1.15, ..., 'total': -0.55}
```

This composable design prevents reward hacking: an agent cannot exploit a single component without being penalized by the others.


---

## 🧠 The Agent: PPO + LSTM

We trained using **Proximal Policy Optimization (PPO) with an LSTM policy** via Stable Baselines 3's `RecurrentPPO`.

**Why LSTM?** Blackouts are *delayed*. An overload on step 2 manifests as a fault on step 5. Without memory, the agent can't connect cause to consequence. The LSTM carries hidden state across steps, letting it anticipate cascades before they happen.

**Why PPO?** Stable, sample-efficient for continuous action spaces. Works well with the non-stationary demand environment.

We also trained a **Qwen2-0.5B LLM agent** using **Hugging Face TRL GRPO**, connecting the language model to the grid environment via text-format observations.

---

## 📈 Training Results

### Training & Agent Comparison — 4-Panel Analysis

![Training Analysis](plots/training_analysis.png)

*4-panel analysis: (top-left) PPO+LSTM training curve over 171K steps; (top-right) reward comparison across all agent types; (bottom-left) blackout frequency — trained agent achieves **0 blackouts**; (bottom-right) grid stability score.*

### Quantitative Results

| Agent Type | Avg Reward | Avg Blackouts | Grid Stability |
|---|---|---|---|
| 🎲 Baseline (Random) | -0.458 | 2.11 | 0.333 |
| 🤖 Selfish Heuristic | -0.016 | 0.11 | 0.944 |
| 🤝 Coordinated Policy | +0.125 | 0.33 | 0.778 |
| 🏆 **PPO+LSTM (Trained)** | **+0.194** | **0.00** | **0.944** |

**PPO+LSTM vs Baseline: +142% reward, -100% blackouts, +183% stability.**

### Training Curve Summary

| Metric | Start of Training | End of Training (171K steps) | Improvement |
|---|---|---|---|
| Avg Episode Reward | **-177** | **-103** | **+41.8%** |
| Best Episode Reward | -177 | **-103** | **+41.8%** |

*Evidence: Full training log in `outputs/training_logs.csv`. Reward curve committed to `plots/reward_curve.png`.*

---

## 💡 What the Agent Learned

The most striking emergent behavior: **defensive curtailment**.

A naive agent tries to serve 100% of demand and triggers overloads. The trained agent intentionally serves *slightly less* than capacity — maintaining a safety buffer. It discovered this non-intuitive strategy from reward signal alone.

This is exactly the strategy experienced human grid operators use manually. The agent invented it independently after ~50,000 timesteps.

---

## 🏗️ Technical Stack

| Component | Technology |
|---|---|
| Environment Framework | [OpenEnv](https://github.com/openenv/openenv) |
| RL Training | Stable Baselines 3 — RecurrentPPO |
| LLM Training | Hugging Face TRL — GRPO |
| LLM Model | Qwen2-0.5B-Instruct |
| Demo Interface | Gradio (HF Spaces) |
| Environment API | Gymnasium-compatible |

---

## 🚀 Try It Yourself

**Live Demo:** [huggingface.co/spaces/TechLearnr4S/GridMind](https://huggingface.co/spaces/TechLearnr4S/GridMind)

```bash
# Run locally
git clone https://github.com/TechLearnr4S/GridMind
cd GridMind
pip install -r requirements.txt
python app.py
```

```python
# Use the environment directly
from env.gridops_env import GridOpsEnv

env = GridOpsEnv()
obs, _ = env.reset()
obs, reward, done, truncated, info = env.step([0.4, 0.3, 0.3])
```

---

## 🌍 Why It Matters

As renewable energy (solar, wind) becomes dominant, grids become *harder* to manage — supply is intermittent and unpredictable. The number of grid stress events is increasing globally.

AI agents that reason about delayed consequences, zone prioritization, and cascading risk represent a genuine frontier. GridMind is a training ground for that capability.

---

## 📁 Repository Structure

```
GridMind/
├── env/
│   ├── gridops_env.py      # Main OpenEnv-compliant environment
│   └── __init__.py
├── plots/
│   └── reward_curve.png    # Training evidence (171K steps)
├── outputs/
│   └── training_logs.csv   # Full training history
├── models/                 # Saved model checkpoints
├── openenv.yaml            # OpenEnv manifest
├── train_lstm_final.py     # PPO+LSTM training script
├── app.py                  # Gradio demo
└── README.md
```