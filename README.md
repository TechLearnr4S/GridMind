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
| 📹 **Demo Video / Blog Post** | [Watch Demo on YouTube](https://youtu.be/example_video_id) |

---

## 📖 Project Overview

GridMind addresses the critical challenge of power grid stability during extreme demand events. Modern grids are susceptible to cascading failures where a single overloaded line triggers a city-wide blackout. We leverage Reinforcement Learning to train agents that prioritize critical infrastructure (hospitals) while maintaining grid stability through "defensive curtailment" — a strategy used by expert human operators.

## 🌐 Environment Design (GridOpsEnv)

`GridOpsEnv` is a custom OpenEnv-compliant environment simulating a 3-zone grid:
- **Zone 1 (Residential)**: Low priority.
- **Zone 2 (Commercial)**: Medium priority.
- **Zone 3 (Hospital)**: High priority.

### Key Features:
- **Delayed Cascades**: Overloads on step $t$ cause faults on step $t+k$.
- **Reputation System**: Prevents zones from misreporting demand.
- **Composable Rubric**: Uses `GridMindRubric` for transparent, multi-factor reward calculation (Stability, Service, Honesty).

## 📈 Training Results

We utilized **RecurrentPPO (PPO + LSTM)** to handle the long-horizon dependencies of cascading failures.
- **Performance**: Achieved **0 blackouts** in stress scenarios compared to 2.11 for random baselines.
- **Stability**: Increased grid stability score by **183%**.
- **Evidence**: All training curves, comparison plots, and logs are available in the [`/plots`](./plots) and [`/outputs`](./outputs) directories.

## 🧠 LLM Integration

GridMind supports **LLM-native training** via the `state_to_text()` method, which converts complex grid states into natural language prompts.
- **Training**: Large Language Models (like Qwen2-0.5B) are trained using **GRPO (Group Relative Policy Optimization)**.
- **Note**: Due to hardware requirements, the LLM training pipeline is designed to run in **Google Colab** or high-end local systems. The Hugging Face Space runs the lightweight RL inference/demo.

---

## 🚀 How to Run Locally

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/TechLearnr4S/GridMind
   cd GridMind
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Demo**:
   ```bash
   python app.py
   ```

---

## 📁 Repository Structure

```
GridMind/
├── env/                # Core OpenEnv-compliant environment
├── train/              # Training scripts and utilities
├── plots/              # Training curves and evaluation results
├── outputs/            # CSV logs and training history
├── models/             # Pre-trained RL model checkpoints
├── openenv.yaml        # OpenEnv manifest
├── app.py              # Gradio Web Interface
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```