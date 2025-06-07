# demo/create_comparison_charts.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple
import os

# Fix font issues by configuring matplotlib to handle missing glyphs gracefully
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def create_performance_bar_chart(results: Dict[str, Dict], workload: str, save_path: str = None):
    """Create a comprehensive bar chart comparing all scheduling algorithms"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
      # Extract data for plotting
    policies = list(results.keys())
    metrics = ['Total Reward', 'Avg Reward/Step', 'Completion Rate (%)']
      # Prepare data
    total_rewards = [results[policy]['total_reward'] for policy in policies]
    avg_rewards = [results[policy]['avg_reward'] for policy in policies]
    completion_rates = [results[policy]['completion_rate'] for policy in policies]
    
    # Calculate scheduling efficiency as a composite metric
    # Efficiency = (normalized_reward * 0.7) + (completion_rate * 0.3)
    # This balances reward performance with task completion
    max_reward = max(total_rewards) if total_rewards else 1
    min_reward = min(total_rewards) if total_rewards else 0
    reward_range = max_reward - min_reward if max_reward != min_reward else 1
    
    scheduling_efficiency = []
    for policy in policies:
        # Normalize reward to 0-100 scale
        normalized_reward = ((results[policy]['total_reward'] - min_reward) / reward_range) * 100
        # Combine normalized reward (70%) with completion rate (30%)
        efficiency = (normalized_reward * 0.7) + (results[policy]['completion_rate'] * 0.3)
        scheduling_efficiency.append(efficiency)
    
    # Create figure with subplots (2x2 layout)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'AI vs Traditional CPU Scheduling Comparison\nWorkload: {workload.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Colors for different algorithm types
    colors = []
    for policy in policies:
        if policy.upper() in ['FIFO', 'RR', 'SJF']:
            colors.append('#FF6B6B')  # Red for traditional
        elif policy.upper() == 'XGB':
            colors.append('#4ECDC4')  # Teal for ML
        elif policy.upper() == 'PPO':
            colors.append('#45B7D1')  # Blue for RL
        else:
            colors.append('#96CEB4')  # Green for others
    
    # 1. Total Reward Comparison
    bars1 = ax1.bar(policies, total_rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Total Reward Comparison', fontweight='bold')
    ax1.set_ylabel('Total Reward')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, total_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Average Reward per Step
    bars2 = ax2.bar(policies, avg_rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Average Reward per Step', fontweight='bold')
    ax2.set_ylabel('Avg Reward/Step')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars2, avg_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Completion Rate
    bars3 = ax3.bar(policies, completion_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_title('Process Completion Rate', fontweight='bold')
    ax3.set_ylabel('Completion Rate (%)')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars3, completion_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')      # 4. Scheduling Efficiency (composite metric: 70% reward performance + 30% completion rate)
    bars4 = ax4.bar(policies, scheduling_efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_title('Overall Scheduling Efficiency', fontweight='bold')
    ax4.set_ylabel('Efficiency Score (0-100)')
    ax4.set_ylim(0, 100)
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars4, scheduling_efficiency):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('Scheduling Algorithm')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#FF6B6B', alpha=0.8, label='Traditional (FIFO, RR, SJF)'),
        plt.Rectangle((0,0),1,1, facecolor='#4ECDC4', alpha=0.8, label='Machine Learning (XGBoost)'),
        plt.Rectangle((0,0),1,1, facecolor='#45B7D1', alpha=0.8, label='Reinforcement Learning (PPO)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    return fig

def create_improvement_percentage_chart(results: Dict[str, Dict], baseline_policy: str = 'FIFO', save_path: str = None):
    """Create a chart showing percentage improvement over baseline"""
    
    if baseline_policy.upper() not in [p.upper() for p in results.keys()]:
        baseline_policy = list(results.keys())[0]  # Use first policy as baseline
    
    # Find baseline values
    baseline_key = None
    for key in results.keys():
        if key.upper() == baseline_policy.upper():
            baseline_key = key
            break
    
    if not baseline_key:
        print(f"Baseline policy {baseline_policy} not found")
        return None
    
    baseline_reward = results[baseline_key]['total_reward']
    baseline_completion = results[baseline_key]['completion_rate']
    
    # Calculate improvements
    policies = []
    reward_improvements = []
    completion_improvements = []
    
    for policy, data in results.items():
        if policy.upper() != baseline_policy.upper():
            policies.append(policy)
            
            # Calculate percentage improvement with zero-division protection
            if abs(baseline_reward) > 0.001:  # Avoid division by very small numbers
                reward_imp = ((data['total_reward'] - baseline_reward) / abs(baseline_reward)) * 100
            else:
                reward_imp = 0.0
                
            if baseline_completion > 0.001:  # Avoid division by very small numbers
                completion_imp = ((data['completion_rate'] - baseline_completion) / baseline_completion) * 100
            else:
                completion_imp = 0.0
            
            reward_improvements.append(reward_imp)
            completion_improvements.append(completion_imp)
    
    if not policies:
        print("No policies to compare with baseline")
        return None
    
    # Create the chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Performance Improvement over {baseline_policy.upper()} Baseline', 
                 fontsize=14, fontweight='bold')
    
    # Colors for AI vs Traditional
    colors = []
    for policy in policies:
        if policy.upper() in ['XGB', 'PPO']:
            colors.append('#4CAF50')  # Green for AI models
        else:
            colors.append('#FF9800')  # Orange for traditional
    
    # Reward improvement chart
    bars1 = ax1.bar(policies, reward_improvements, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Total Reward Improvement (%)')
    ax1.set_ylabel('Improvement (%)')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars1, reward_improvements):
        height = bar.get_height()
        y_pos = height + (5 if height >= 0 else -8)
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', color='green' if value > 0 else 'red')
    
    # Completion rate improvement chart
    bars2 = ax2.bar(policies, completion_improvements, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Completion Rate Improvement (%)')
    ax2.set_ylabel('Improvement (%)')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars2, completion_improvements):
        height = bar.get_height()
        y_pos = height + (1 if height >= 0 else -2)
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontweight='bold', color='green' if value > 0 else 'red')
    
    # Rotate labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        improvement_path = save_path.replace('.png', '_improvement.png')
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
        print(f"Improvement chart saved to: {improvement_path}")
    
    return fig

def create_summary_report(results: Dict[str, Dict], workload: str) -> str:
    """Generate a text summary of the comparison results"""
    
    # Find best performers
    best_reward = max(results.keys(), key=lambda k: results[k]['total_reward'])
    best_completion = max(results.keys(), key=lambda k: results[k]['completion_rate'])
    
    # Separate AI vs Traditional
    ai_models = [k for k in results.keys() if k.upper() in ['XGB', 'PPO']]
    traditional = [k for k in results.keys() if k.upper() in ['FIFO', 'RR', 'SJF']]
    
    report = f"""
AI-POWERED CPU SCHEDULER COMPARISON REPORT
{'='*60}

WORKLOAD: {workload.upper()}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

TOP PERFORMERS:
   Best Total Reward: {best_reward.upper()} ({results[best_reward]['total_reward']:.3f})
   Best Completion Rate: {best_completion.upper()} ({results[best_completion]['completion_rate']:.1f}%)

DETAILED RESULTS:
"""
    
    for policy, data in results.items():
        category = "AI MODEL" if policy.upper() in ['XGB', 'PPO'] else "TRADITIONAL"
        report += f"""
   {category} - {policy.upper()}:
      • Total Reward: {data['total_reward']:.3f}
      • Avg Reward/Step: {data['avg_reward']:.3f}
      • Completion Rate: {data['completion_rate']:.1f}%
      • Total Steps: {data['total_steps']}
"""

    # Calculate AI advantage
    if ai_models and traditional:
        ai_avg_reward = np.mean([results[k]['total_reward'] for k in ai_models])
        traditional_avg_reward = np.mean([results[k]['total_reward'] for k in traditional])
        
        if abs(traditional_avg_reward) > 0.001:  # Avoid division by zero
            improvement = ((ai_avg_reward - traditional_avg_reward) / abs(traditional_avg_reward)) * 100
            
            report += f"""
AI MODEL ADVANTAGES:
   • Average AI Reward: {ai_avg_reward:.3f}
   • Average Traditional Reward: {traditional_avg_reward:.3f}
   • AI Performance Improvement: {improvement:+.1f}%
   
CONCLUSION:
   AI-powered scheduling shows {"significant improvement" if improvement > 10 else "improvement" if improvement > 0 else "mixed results"} 
   over traditional rule-based algorithms for {workload} workloads.
"""

    report += f"""
KEY INSIGHTS:
   • XGBoost: Machine learning approach optimized on historical data
   • PPO: Reinforcement learning agent that learned through trial and error
   • Traditional algorithms follow fixed rules without adaptation
   
RECOMMENDATION:
   {"Use AI models for better performance" if ai_models and any(results[k]['total_reward'] > max([results[t]['total_reward'] for t in traditional]) if traditional else -999 for k in ai_models) else "Continue testing with different workloads"}
"""
    
    return report

def save_comparison_results(results: Dict[str, Dict], workload: str, output_dir: str = "demo/results"):
    """Save comprehensive comparison results including charts and reports"""
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Create and save performance chart
        chart_path = os.path.join(output_dir, f'performance_comparison_{workload}_{timestamp}.png')
        try:
            create_performance_bar_chart(results, workload, chart_path)
        except Exception as e:
            print(f"Error creating performance chart: {e}")
            return None
        
        # Create and save improvement chart
        improvement_path = os.path.join(output_dir, f'improvement_analysis_{workload}_{timestamp}.png')
        try:
            create_improvement_percentage_chart(results, save_path=improvement_path)
        except Exception as e:
            print(f"Error creating improvement chart: {e}")
        
        # Generate and save text report
        try:
            report = create_summary_report(results, workload)
            report_path = os.path.join(output_dir, f'comparison_report_{workload}_{timestamp}.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
        except Exception as e:
            print(f"Error creating report: {e}")
            report = "Report generation failed"
            report_path = None
        
        print(f"\nResults saved to: {output_dir}")
        print(f"   Performance Chart: {os.path.basename(chart_path)}")
        if 'improvement_path' in locals():
            print(f"   Improvement Chart: {os.path.basename(improvement_path)}")
        if report_path:
            print(f"   Text Report: {os.path.basename(report_path)}")
        
        return {
            'chart_path': chart_path,
            'improvement_path': improvement_path if 'improvement_path' in locals() else None,
            'report_path': report_path,
            'report_text': report
        }
        
    except Exception as e:
        print(f"Chart generation failed: {e}")
        return None

if __name__ == "__main__":
    # Example usage with dummy data
    sample_results = {
        'FIFO': {'total_reward': -5.2, 'avg_reward': -0.104, 'completion_rate': 60.0, 'total_steps': 50},
        'RR': {'total_reward': -4.8, 'avg_reward': -0.096, 'completion_rate': 65.0, 'total_steps': 50},
        'SJF': {'total_reward': -3.9, 'avg_reward': -0.078, 'completion_rate': 70.0, 'total_steps': 50},
        'XGB': {'total_reward': 2.3, 'avg_reward': 0.046, 'completion_rate': 85.0, 'total_steps': 50},
        'PPO': {'total_reward': 3.1, 'avg_reward': 0.062, 'completion_rate': 90.0, 'total_steps': 50}
    }
    
    print("Testing chart generation with sample data...")
    save_comparison_results(sample_results, "mixed", "demo/test_results")
    print("Test completed!")
