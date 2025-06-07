# demo/generate_charts.py
import argparse
import os
import sys
import subprocess
from create_comparison_charts import save_comparison_results

def run_all_policies_and_generate_charts(workload):
    """Run all policies and generate visual charts"""
    
    project_root = "d:\\Projects\\Hust\\Operating system 2024.2"
    os.chdir(project_root)
    
    policies = ["fifo", "rr", "sjf", "xgb", "ppo"]
    results = {}
    
    print(f"🚀 Running all policies for {workload.upper()} workload...")
    print("=" * 60)
    
    # Run each policy and collect metrics
    for policy in policies:
        print(f"Running {policy.upper()}...")
        try:
            result = subprocess.run(
                [sys.executable, "demo/run_simulation.py", "--policy", policy, "--workload", workload],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            if result.returncode == 0:
                print(f"✅ {policy.upper()} completed")
                
                # Parse results from output
                lines = result.stdout.split('\n')
                total_reward = 0
                completion_rate = 0
                total_steps = 0
                for line in lines:
                    if 'Total Reward:' in line:
                        total_reward = float(line.split(':')[1].strip())
                    elif 'Processes Completed:' in line:
                        # Parse "Processes Completed: 1/5 (20.0%)" format
                        percentage_part = line.split('(')[1].split(')')[0].replace('%', '')
                        completion_rate = float(percentage_part)
                    elif 'Total Steps:' in line:
                        total_steps = int(line.split(':')[1].strip())
                
                # Store results for chart generation
                results[policy] = {
                    'total_reward': total_reward,
                    'avg_reward': total_reward / max(total_steps, 1),
                    'completion_rate': completion_rate,
                    'total_steps': total_steps
                }
                
            else:
                print(f"❌ {policy.upper()} failed")
                
        except Exception as e:
            print(f"❌ {policy.upper()} error: {str(e)}")
    
    # Generate charts if we have results
    if results:
        print(f"\n📊 Generating visual charts...")
        try:
            chart_results = save_comparison_results(results, workload, "demo/results")
            
            print(f"\n🎉 Charts generated successfully!")
            print(f"📁 Results saved to: demo/results/")
            print(f"📈 Performance Chart: {os.path.basename(chart_results['chart_path'])}")
            print(f"🚀 Improvement Chart: {os.path.basename(chart_results['improvement_path'])}")
            print(f"📋 Text Report: {os.path.basename(chart_results['report_path'])}")
            
            return chart_results
            
        except Exception as e:
            print(f"❌ Chart generation failed: {str(e)}")
            return None
    else:
        print(f"❌ No valid results to generate charts from")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison charts for CPU scheduling algorithms")
    parser.add_argument("--workload", choices=["cpu", "io", "mixed", "idle", "ram-heavy", "mixed-heavy"], default="mixed",
                        help="Workload type to test (default: mixed)")
    
    args = parser.parse_args()
    
    print("🤖 CPU Scheduler Chart Generator")
    print("=" * 50)
    
    result = run_all_policies_and_generate_charts(args.workload)
    
    if result:
        print(f"\n✨ Open the generated PNG files to see the visual comparisons!")
    else:
        print(f"\n💥 Chart generation failed. Check the error messages above.")