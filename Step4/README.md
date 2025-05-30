# 🤖 Step 4: RL Agent Training for CPU Scheduler

## 🎯 Overview
This step implements Reinforcement Learning (RL) agent training using **PPO (Proximal Policy Optimization)** to learn optimal CPU scheduling policies. The agent interacts with the custom environment from Step 3 to develop intelligent scheduling strategies.

## 🏗 Project Structure
```
Step4/
├── train_rl_agent.py      # Main training script
├── evaluate_model.py      # Model evaluation and analysis
├── requirements.txt       # Dependencies
├── README.md             # This file
├── STEP4.pdf             # Original requirements
├── models/               # Saved model checkpoints (created during training)
├── best_model/           # Best performing model (created during training)
├── logs/                 # Training logs (created during training)
├── tensorboard_logs/     # TensorBoard logs (created during training)
└── evaluation_results/   # Evaluation plots and results (created during evaluation)
```

## 🚀 Quick Start

### 1. Train the RL Agent
```bash
cd "d:\Projects\Hust\Operating system 2024.2\Step4"
python train_rl_agent.py
```

### 2. Monitor Training (Optional)
Open a new terminal and run:
```bash
tensorboard --logdir ./tensorboard_logs/
```
Then open http://localhost:6006 in your browser.

### 3. Evaluate the Trained Model
```bash
python evaluate_model.py
```

## 🧠 Algorithm Details

### PPO Configuration
- **Policy**: MlpPolicy (Multi-Layer Perceptron)
- **Learning Rate**: 3e-4
- **Steps per Update**: 2,048
- **Batch Size**: 64
- **Training Epochs**: 10
- **Discount Factor (γ)**: 0.99
- **GAE Lambda**: 0.95
- **Clip Range**: 0.2

### Training Parameters
- **Total Timesteps**: 100,000
- **Evaluation Frequency**: Every 5,000 steps
- **Checkpoint Frequency**: Every 10,000 steps

## 📊 Environment Integration

The agent learns to interact with the `CPUSchedulerEnv` from Step 3:

### Observation Space
- **Size**: 35 dimensions (5 processes × 7 features)
- **Features per Process**:
  - CPU utilization (normalized 0-1)
  - Memory utilization (normalized 0-1)
  - Priority (0.0, 0.5, 1.0)
  - I/O wait time (normalized 0-1)
  - Context switches (normalized 0-1)
  - Is currently running (0 or 1)
  - Remaining execution time (normalized 0-1)

### Action Space
- **Size**: 6 discrete actions
- **Actions**: 
  - 0-4: Schedule process 0-4
  - 5: No operation (NOOP)

### Reward Function
The agent learns to optimize multiple objectives:
- **Throughput**: Reward for efficient CPU utilization
- **Memory Management**: Penalty for high memory usage
- **Priority Handling**: Bonus for scheduling high-priority processes
- **I/O Efficiency**: Penalty for high I/O wait times
- **Fairness**: Penalty for process starvation
- **Context Switch Minimization**: Penalty for excessive switching

## 🏆 Expected Results

A well-trained agent should achieve:
- **Positive average rewards** (>0)
- **Consistent performance** (low variance)
- **Better than random** policy (>20% improvement)
- **Balanced reward components** (no single dominant penalty)

---

🎉 **Good luck with your RL training!** The agent should learn effective CPU scheduling strategies through trial and error.