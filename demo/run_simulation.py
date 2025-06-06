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
    parser.add_argument("--workload", default="mixed", choices=["cpu", "io", "mixed"])
    parser.add_argument("--render", action="store_true", help="Print scheduling timeline")
    args = parser.parse_args()

    # Fixed this section
    env = CPUSchedulerEnv(num_processes=5, max_steps=50)

    if args.policy in ["fifo", "rr", "sjf"]:
        policy_fn = lambda obs: baseline_policy(obs, mode=args.policy)
    else:
        model = load_policy(args.policy)
        policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]

    obs = env.reset()
    done, rewards, steps = False, [], 0
    total_reward = 0
    policy_decisions = []
    finished_processes = 0
    
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
        
        # Track policy decisions
        if action < env.num_processes:
            policy_decisions.append(f"Step {steps}: Scheduled Process {action}")
        else:
            policy_decisions.append(f"Step {steps}: No operation (NOOP)")
            
        # Track finished processes
        if info.get('finished', False):
            finished_processes += 1
            
        if args.render:
            env.render()

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
    
    # Show some policy decisions
    print(f"\n=== SCHEDULING DECISIONS (Last 10) ===")
    for decision in policy_decisions[-10:]:
        print(decision)
        
    print(f"\n[SUCCESS] Simulation completed in {steps} steps.")


if __name__ == "__main__":
    main()
