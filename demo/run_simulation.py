# demo/run_simulation.py
import argparse
import sys
import os
from policy_loader import load_policy
from baselines import baseline_policy
from CPUSchedulerEnv import CPUSchedulerEnv  # import from Step3 if needed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["fifo", "rr", "sjf", "xgb", "ppo"], required=True)
    parser.add_argument("--workload", default="mixed", choices=["cpu", "io", "mixed", "idle", "ram-heavy", "mixed-heavy"])
    parser.add_argument("--render", action="store_true", help="Print scheduling timeline")
    args = parser.parse_args()

    # Pass workload parameter to environment
    env = CPUSchedulerEnv(num_processes=5, max_steps=100, workload_type=args.workload)

    if args.policy in ["fifo", "rr", "sjf"]:
        policy_fn = lambda obs: baseline_policy(obs, mode=args.policy)
    else:
        model = load_policy(args.policy)
        policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]

    obs = env.reset()
    done, rewards, steps = False, [], 0
    total_reward = 0
    finished_processes = 0
    important_events = []
    current_process = None
    process_start_step = 1
    
    print(f"\n=== Starting {args.policy.upper()} Scheduler Simulation ===")
    print(f"Workload Type: {args.workload}")
    print(f"Environment: {env.num_processes} processes, max {env.max_steps} steps")
    print("-" * 50)

    while not done:
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        total_reward += reward
        steps += 1
        
        # Track process switches and completions
        if action < env.num_processes:
            # Check if we switched to a different process
            if current_process != action:
                if current_process is not None:
                    duration = steps - process_start_step
                    important_events.append(f"Steps {process_start_step}-{steps-1}: Process {current_process} ran for {duration} steps")
                current_process = action
                process_start_step = steps
            
            # Check if process completed
            if info.get('finished', False):
                finished_processes += 1
                duration = steps - process_start_step + 1
                important_events.append(f"Step {steps}: [COMPLETED] Process {action} finished after {duration} steps (Total completed: {finished_processes})")
                important_events.append(f"Step {steps}: [SPAWNED] New Process {action} created")
                process_start_step = steps + 1
        else:
            # NOOP action
            if current_process is not None:
                duration = steps - process_start_step
                important_events.append(f"Steps {process_start_step}-{steps-1}: Process {current_process} ran for {duration} steps")
                current_process = None
            important_events.append(f"Step {steps}: NOOP (No operation)")
            process_start_step = steps + 1
            
        if args.render:
            env.render()
    
    # Add final running process info
    if current_process is not None:
        duration = steps - process_start_step + 1
        important_events.append(f"Steps {process_start_step}-{steps}: Process {current_process} ran for {duration} steps (Incomplete)")

    # Calculate performance metrics
    avg_reward = total_reward / steps if steps > 0 else 0
    completion_rate = finished_processes / env.num_processes * 100
    
    print(f"\n=== SIMULATION RESULTS ===")
    print(f"Policy: {args.policy.upper()}")
    print(f"Workload: {args.workload}")
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {total_reward:.3f}")
    print(f"Average Reward per Step: {avg_reward:.3f}")
    print(f"Processes Completed: {finished_processes}/{env.num_processes} ({completion_rate:.1f}%)")
    
    if len(rewards) > 0:
        print(f"Best Step Reward: {max(rewards):.3f}")
        print(f"Worst Step Reward: {min(rewards):.3f}")
    
    # Show important events instead of all decisions
    print(f"\n=== SCHEDULING EVENTS ===")
    for event in important_events:
        print(event)
        
    print(f"\n[SUCCESS] Simulation completed in {steps} steps.")


if __name__ == "__main__":
    main()
