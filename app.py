import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.gridops_env import GridOpsEnv

# Initialize Environment
env = GridOpsEnv(num_zones=3, max_steps=50)

def create_plot(demand, supply, faults):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(demand))
    width = 0.35
    
    ax.bar(x - width/2, demand, width, label='Demand', color='#FF9999')
    ax.bar(x + width/2, supply, width, label='Supply', color='#99FF99')
    
    # Highlight faults
    for i, f in enumerate(faults):
        if f > 0:
            ax.text(i, max(demand[i], supply[i]) + 0.1, "⚠️ FAULT", 
                    ha='center', color='red', fontweight='bold')
            
    ax.set_ylabel('Power Units')
    ax.set_title('Grid Status per Zone')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Zone {i+1}' for i in range(len(demand))])
    ax.legend()
    ax.set_ylim(0, max(max(demand), max(supply)) + 0.5)
    
    plt.tight_layout()
    return fig

def reset_env():
    obs, info = env.reset()
    state_text = env.state_to_text()
    plot = create_plot(obs['demand'], obs['supply'], obs['faults'])
    return state_text, plot, 0.0, False, "Environment reset. Waiting for action..."

def step_env(a1, a2, a3):
    action = [float(a1), float(a2), float(a3)]
    # Normalize if user inputs don't sum to 1
    total = sum(action)
    if total > 0:
        action = [x/total for x in action]
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state_text = env.state_to_text()
    plot = create_plot(obs['demand'], obs['supply'], obs['faults'])
    
    status = "Step completed."
    if done:
        status = "Episode Finished! Click Reset to start again."
        
    return state_text, plot, round(reward, 4), done, status

# Gradio CSS for a premium look
css = """
.container { max-width: 900px; margin: auto; }
.stat-box { font-family: monospace; background: #f0f2f5; padding: 15px; border-radius: 8px; border: 1px solid #ddd; }
"""

with gr.Blocks(css=css, title="GridMind Demo") as demo:
    gr.Markdown("# ⚡ GridMind: Power Grid Coordination AI")
    gr.Markdown("Interactive demo of the `GridOpsEnv` environment. Allocate power to prevent cascading blackouts.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 Controls")
            a1 = gr.Slider(0, 1, value=0.33, label="Zone 1 Allocation (Residential)")
            a2 = gr.Slider(0, 1, value=0.33, label="Zone 2 Allocation (Commercial)")
            a3 = gr.Slider(0, 1, value=0.34, label="Zone 3 Allocation (Hospital)")
            
            with gr.Row():
                btn_step = gr.Button("🚀 Take Step", variant="primary")
                btn_reset = gr.Button("🔄 Reset Env")
                
            reward_out = gr.Number(label="Last Step Reward")
            done_out = gr.Checkbox(label="Episode Done")
            status_out = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Live Grid State")
            plot_out = gr.Plot(label="Demand vs Supply")
            state_out = gr.Textbox(label="Environment Description", lines=12, elem_classes="stat-box")

    # Event handlers
    btn_step.click(
        fn=step_env, 
        inputs=[a1, a2, a3], 
        outputs=[state_out, plot_out, reward_out, done_out, status_out]
    )
    btn_reset.click(
        fn=reset_env, 
        outputs=[state_out, plot_out, reward_out, done_out, status_out]
    )
    
    # Init state
    demo.load(fn=reset_env, outputs=[state_out, plot_out, reward_out, done_out, status_out])

if __name__ == "__main__":
    demo.launch()
