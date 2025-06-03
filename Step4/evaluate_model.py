#!/usr/bin/env python3
"""
Step 4: Model Evaluation Script
Evaluate and analyze the performance of the trained PPO agent.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import pandas as pd

# Add Step3 directory to path to import the environment
step3_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step3', 'OSpro')
sys.path.append(step3_path)

from cpuscheduler_env import CPUSchedulerEnv

def load_trained_model(model_path):
    """Load the trained PPO model"""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("🔍 Available models:")
        models_dir = "./models/"
        best_model_dir = "./best_model/"
        
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.zip'):
                    print(f"  - {os.path.join(models_dir, f)}")
        
        if os.path.exists(best_model_dir):
            for f in os.listdir(best_model_dir):
                if f.endswith('.zip'):
                    print(f"  - {os.path.join(best_model_dir, f)}")
        return None
    
    try:
        model = PPO.load(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def evaluate_agent(model, env, num_episodes=10):
    """Evaluate the agent's performance over multiple episodes"""
    print(f"\n🧪 Evaluating agent over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    detailed_stats = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        episode_info = []
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            episode_info.append({
                'step': steps,
                'action': action,
                'reward': reward,
                'info': info
            })
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        detailed_stats.append(episode_info)
        
        print(f"Episode {episode + 1:2d}: {steps:2d} steps, Total reward: {total_reward:7.3f}")
    
    return episode_rewards, episode_lengths, detailed_stats

def analyze_performance(episode_rewards, episode_lengths, detailed_stats):
    """Analyze and visualize agent performance"""
    print("\n📊 Performance Analysis:")
    print("=" * 50)
    
    # Basic statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print(f"Average Episode Reward: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"Average Episode Length: {avg_length:.1f} ± {std_length:.1f}")
    print(f"Best Episode Reward: {max(episode_rewards):.3f}")
    print(f"Worst Episode Reward: {min(episode_rewards):.3f}")
    
    # Reward breakdown analysis
    print("\n🔍 Reward Component Analysis:")
    reward_components = {
        'throughput_reward': [],
        'mem_penalty': [],
        'prio_bonus': [],
        'io_penalty': [],
        'fairness_penalty': [],
        'switch_penalty': [],
        'switch_action_penalty': []
    }
    
    for episode_info in detailed_stats:
        for step_info in episode_info:
            info = step_info['info']
            for component in reward_components:
                if component in info:
                    reward_components[component].append(info[component])
    
    for component, values in reward_components.items():
        if values:
            avg_val = np.mean(values)
            print(f"  {component:20s}: {avg_val:7.3f}")
    
    # Create visualizations
    create_performance_plots(episode_rewards, episode_lengths, reward_components)

def create_performance_plots(episode_rewards, episode_lengths, reward_components):
    """Create performance visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('RL Agent Performance Analysis', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, 'b-o', markersize=4)
    axes[0, 0].axhline(y=np.mean(episode_rewards), color='r', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, 'g-o', markersize=4)
    axes[0, 1].axhline(y=np.mean(episode_lengths), color='r', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 0].hist(episode_rewards, bins=min(10, len(episode_rewards)), alpha=0.7, color='blue')
    axes[1, 0].axvline(x=np.mean(episode_rewards), color='r', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Total Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward components (average values)
    components = []
    values = []
    for component, comp_values in reward_components.items():
        if comp_values:
            components.append(component.replace('_', '\n'))
            values.append(np.mean(comp_values))
    
    if components:
        colors = ['green' if v >= 0 else 'red' for v in values]
        bars = axes[1, 1].bar(range(len(components)), values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Average Reward Components')
        axes[1, 1].set_xlabel('Component')
        axes[1, 1].set_ylabel('Average Value')
        axes[1, 1].set_xticks(range(len(components)))
        axes[1, 1].set_xticklabels(components, rotation=45, ha='right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                           f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("./evaluation_results/", exist_ok=True)
    plt.savefig("./evaluation_results/performance_analysis.png", dpi=300, bbox_inches='tight')
    print(f"\n📈 Performance plots saved to: ./evaluation_results/performance_analysis.png")
    plt.show()

def compare_with_random_policy(env, num_episodes=5):
    """Compare trained agent with random policy baseline"""
    print(f"\n🎲 Comparing with random policy baseline ({num_episodes} episodes)...")
    
    random_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        
        while True:
            action = env.action_space.sample()  # Random action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        random_rewards.append(total_reward)
    
    avg_random_reward = np.mean(random_rewards)
    print(f"Random Policy Average Reward: {avg_random_reward:.3f} ± {np.std(random_rewards):.3f}")
    
    return avg_random_reward

def main():
    print("🔍 Step 4: Model Evaluation and Analysis")
    print("=" * 50)
    
    # Create environment
    env = CPUSchedulerEnv(num_processes=5, max_steps=50)
    env = Monitor(env)
    
    # Try to load the best model first, then final model
    model_paths = [
        "./best_model/best_model.zip",
        "./models/ppo_cpu_scheduler_final.zip"
    ]
    
    model = None
    for path in model_paths:
        model = load_trained_model(path)
        if model is not None:
            break
    
    if model is None:
        print("❌ No trained model found. Please run train_rl_agent.py first.")
        return
    
    # Evaluate the trained agent
    episode_rewards, episode_lengths, detailed_stats = evaluate_agent(model, env, num_episodes=10)
    
    # Analyze performance
    analyze_performance(episode_rewards, episode_lengths, detailed_stats)
    
    # Compare with random baseline
    random_avg = compare_with_random_policy(env, num_episodes=5)
    agent_avg = np.mean(episode_rewards)
    
    print(f"\n🏆 Performance Comparison:")
    print(f"Trained Agent: {agent_avg:.3f}")
    print(f"Random Policy: {random_avg:.3f}")
    print(f"Improvement: {((agent_avg - random_avg) / abs(random_avg) * 100):+.1f}%")
    
    print("\n✅ Evaluation completed!")
    print("📁 Results saved in ./evaluation_results/")

if __name__ == "__main__":
    main()