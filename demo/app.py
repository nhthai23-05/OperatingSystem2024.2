# demo/app.py
import gradio as gr
import os
import subprocess
import sys

def launch(policy, workload):
    # Fix path for running from demo directory
    project_root = "d:\\Projects\\Hust\\Operating system 2024.2"
    os.chdir(project_root)
    
    # Use subprocess to capture output instead of os.system
    try:
        result = subprocess.run(
            [sys.executable, "demo/run_simulation.py", "--policy", policy, "--workload", workload],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode == 0:
            output = f"[SUCCESS] Simulation completed successfully\n"
            output += f"Policy: {policy.upper()}, Workload: {workload.upper()}\n"
            output += "=" * 60 + "\n"
            if result.stdout:
                output += result.stdout
            return output
        else:
            return f"[ERROR] Simulation failed\nError Details:\n{result.stderr}"
            
    except Exception as e:
        return f"[ERROR] Error running simulation: {str(e)}"

def compare_all_policies(workload):
    """Comprehensive comparison of all scheduling policies"""
    project_root = "d:\\Projects\\Hust\\Operating system 2024.2"
    os.chdir(project_root)
    
    policies = ["fifo", "rr", "sjf", "xgb", "ppo"]
    results = f"SCHEDULER COMPARISON - {workload.upper()} WORKLOAD\n"
    results += "=" * 70 + "\n\n"
    
    policy_results = {}
    
    for policy in policies:
        try:
            result = subprocess.run(
                [sys.executable, "demo/run_simulation.py", "--policy", policy, "--workload", workload],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            if result.returncode == 0:
                results += f"{policy.upper()} RESULTS:\n"
                results += result.stdout + "\n"
                results += "-" * 50 + "\n\n"
                
                # Extract key metrics for summary
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Total Reward:' in line:
                        reward = line.split(':')[1].strip()
                        policy_results[policy.upper()] = reward
            else:
                results += f"ERROR: {policy.upper()} execution failed\n\n"
                
        except Exception as e:
            results += f"ERROR: {policy.upper()} - {str(e)}\n\n"
    
    # Add summary comparison
    results += "PERFORMANCE SUMMARY:\n"
    results += "=" * 30 + "\n"
    for policy, reward in policy_results.items():
        results += f"{policy}: {reward}\n"
    
    results += "\nNote: Higher rewards indicate better performance"
    results += "\nFor visual charts, run: python demo/generate_charts.py --workload " + workload
    
    return results

# Create Gradio interface
with gr.Blocks(title="AI-Powered CPU Scheduler Demo") as demo:
    gr.Markdown("""
    # AI-Powered CPU Scheduler Demo
    
    Compare different CPU scheduling algorithms including AI models (PPO, XGBoost) vs traditional methods (FIFO, RR, SJF)
    
    ## What This Demo Shows:
    - **Traditional Algorithms**: FIFO, Round Robin (RR), Shortest Job First (SJF)
    - **AI-Powered Algorithms**: XGBoost (Machine Learning), PPO (Reinforcement Learning)
    - **Performance Metrics**: Rewards, completion rates, scheduling decisions
    """)
    
    with gr.Tab("Single Policy Test"):
        gr.Markdown("### Test individual scheduling policies")
        with gr.Row():
            policy_input = gr.Dropdown(["fifo", "rr", "sjf", "xgb", "ppo"], label="Scheduling Policy", value="xgb")
            workload_input = gr.Dropdown(["cpu", "io", "mixed", "idle", "ram-heavy", "mixed-heavy"], label="Workload Type", value="mixed")
        
        run_button = gr.Button("Run Simulation", variant="primary")
        single_output = gr.Textbox(label="Results", lines=20, max_lines=30)
        
        run_button.click(launch, inputs=[policy_input, workload_input], outputs=single_output)
    
    with gr.Tab("Compare All Policies"):
        gr.Markdown("### Compare All Scheduling Algorithms")
        workload_compare = gr.Dropdown(["cpu", "io", "mixed", "idle", "ram-heavy", "mixed-heavy"], label="Workload Type", value="mixed")
        compare_button = gr.Button("Run Comparison", variant="primary", size="lg")
        compare_output = gr.Textbox(label="Comparison Results", lines=30, max_lines=50)
        
        compare_button.click(compare_all_policies, inputs=[workload_compare], outputs=compare_output)

demo.allow_flagging = "never"
demo.launch(share=True)