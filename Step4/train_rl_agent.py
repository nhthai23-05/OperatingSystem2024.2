# #!/usr/bin/env python3
# """
# Step 4: RL Agent Training for CPU Scheduler
# Train a PPO agent to learn optimal CPU scheduling policies using the environment from Step 3.
# """

# import os
# import sys
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv
# import torch

# # Add Step3 directory to path to import the environment
# step3_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step3', 'OSpro')
# sys.path.append(step3_path)

# from cpuscheduler_env import CPUSchedulerEnv

# def create_env():
#     """Create and configure the CPU Scheduler environment"""
#     env = CPUSchedulerEnv(num_processes=5, max_steps=50)
#     return env

# def setup_directories():
#     """Create necessary directories for logging and model saving"""
#     directories = [
#         "./models/",
#         "./logs/", 
#         "./tensorboard_logs/",
#         "./best_model/"
#     ]
    
#     for directory in directories:
#         os.makedirs(directory, exist_ok=True)
#         print(f"✓ Directory created/verified: {directory}")

# def main():
#     print("🚀 Starting Step 4: RL Agent Training for CPU Scheduler")
#     print("=" * 60)
    
#     # Setup directories
#     setup_directories()
    
#     # Set random seeds for reproducibility
#     np.random.seed(42)
#     torch.manual_seed(42)
    
#     # Create training environment
#     print("🔧 Creating training environment...")
#     train_env = create_env()
#     train_env = Monitor(train_env, "./logs/train")
#     train_env = DummyVecEnv([lambda: train_env])
    
#     # Create evaluation environment  
#     print("🔧 Creating evaluation environment...")
#     eval_env = create_env()
#     eval_env = Monitor(eval_env, "./logs/eval")
    
#     # Configure PPO agent
#     print("🧠 Initializing PPO agent...")
#     model = PPO(
#         policy="MlpPolicy",
#         env=train_env,
#         learning_rate=3e-4,
#         n_steps=2048,        # Number of steps to run for each environment per update
#         batch_size=64,       # Minibatch size
#         n_epochs=10,         # Number of epoch when optimizing the surrogate loss
#         gamma=0.99,          # Discount factor
#         gae_lambda=0.95,     # Factor for trade-off of bias vs variance for GAE
#         clip_range=0.2,      # Clipping parameter for PPO
#         ent_coef=0.01,       # Entropy coefficient for the loss calculation
#         vf_coef=0.5,         # Value function coefficient for the loss calculation
#         max_grad_norm=0.5,   # Maximum value for the gradient clipping
#         verbose=1,
#         tensorboard_log="./tensorboard_logs/"
#     )
    
#     print("📊 Model configuration:")
#     print(f"  - Policy: MlpPolicy")
#     print(f"  - Learning rate: 3e-4")
#     print(f"  - Steps per update: 2048")
#     print(f"  - Batch size: 64")
#     print(f"  - Training epochs: 10")
#     print(f"  - Discount factor (gamma): 0.99")
    
#     # Setup callbacks
#     print("⚙️ Setting up training callbacks...")
#       # Evaluation callback - evaluates the agent periodically
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path="./best_model/",
#         log_path="./logs/",
#         eval_freq=10000,     # Increased from 5000 to 10000 for longer training
#         deterministic=True,
#         render=False,
#         verbose=1
#     )
#       # Checkpoint callback - saves model periodically
#     checkpoint_callback = CheckpointCallback(
#         save_freq=25000,     # Increased from 10000 to 25000 for longer training
#         save_path="./models/",
#         name_prefix="ppo_cpu_scheduler"
#     )
    
#     callbacks = [eval_callback, checkpoint_callback]
#       # Train the agent
#     print("🏋️ Starting training...")
#     print(f"📈 Training for 500,000 timesteps...")
#     print("💡 You can monitor progress with TensorBoard:")
#     print("   tensorboard --logdir ./tensorboard_logs/")
#     print("-" * 60)
    
#     try:
#         model.learn(
#             total_timesteps=500000,  # Increased from 100,000 to 500,000
#             callback=callbacks,
#             tb_log_name="ppo_cpu_scheduler"
#         )
#         print("✅ Training completed successfully!")
        
#     except KeyboardInterrupt:
#         print("\n⚠️ Training interrupted by user")
        
#     except Exception as e:
#         print(f"❌ Training failed with error: {e}")
#         raise
    
#     # Save final model
#     print("💾 Saving final model...")
#     model.save("./models/ppo_cpu_scheduler_final")
#     print("✅ Final model saved as: ./models/ppo_cpu_scheduler_final.zip")
    
#     # Test the trained agent
#     print("\n🧪 Testing trained agent...")
#     test_agent(model, eval_env)
    
#     print("\n🎉 Step 4 completed successfully!")
#     print("📁 Check the following directories for results:")
#     print("  - ./models/ - Saved model checkpoints")
#     print("  - ./best_model/ - Best performing model")
#     print("  - ./logs/ - Training and evaluation logs")
#     print("  - ./tensorboard_logs/ - TensorBoard logs")

# def test_agent(model, env, num_episodes=3):
#     """Test the trained agent for a few episodes"""
#     print(f"Running {num_episodes} test episodes...")
    
#     for episode in range(num_episodes):
#         obs = env.reset()
#         total_reward = 0
#         steps = 0
        
#         print(f"\n--- Episode {episode + 1} ---")
        
#         while True:
#             action, _states = model.predict(obs, deterministic=True)
#             obs, reward, done, info = env.step(action)
#             total_reward += reward
#             steps += 1
            
#             if steps <= 3:  # Show first few steps
#                 print(f"Step {steps}: Action={action}, Reward={reward:.3f}")
            
#             if done:
#                 break
        
#         print(f"Episode {episode + 1} completed: {steps} steps, Total reward: {total_reward:.3f}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Step 4: RL Agent Training for CPU Scheduler
Train a PPO agent to learn optimal CPU scheduling policies using the environment from Step 3.
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

# Add Step3 directory to path to import the environment
step3_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step3', 'OSpro')
sys.path.append(step3_path)

from demo.CPUSchedulerEnv import CPUSchedulerEnv

def create_env():
    """Create and configure the CPU Scheduler environment"""
    env = CPUSchedulerEnv(num_processes=5, max_steps=50)
    return env

def setup_directories():
    """Create necessary directories for logging and model saving"""
    directories = [
        "./models/",
        "./logs/", 
        "./tensorboard_logs/",
        "./best_model/"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory created/verified: {directory}")

def main():
    print("🚀 Starting Step 4: RL Agent Training for CPU Scheduler")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create training environment
    print("🔧 Creating training environment...")
    train_env = create_env()
    train_env = Monitor(train_env, "./logs/train")
    train_env = DummyVecEnv([lambda: train_env])
    
    # Create evaluation environment  
    print("🔧 Creating evaluation environment...")
    eval_env = create_env()
    eval_env = Monitor(eval_env, "./logs/eval")
    
    # Configure PPO agent
    print("🧠 Initializing PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,        # Number of steps to run for each environment per update
        batch_size=64,       # Minibatch size
        n_epochs=10,         # Number of epoch when optimizing the surrogate loss
        gamma=0.99,          # Discount factor
        gae_lambda=0.95,     # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,      # Clipping parameter for PPO
        ent_coef=0.01,       # Entropy coefficient for the loss calculation
        vf_coef=0.5,         # Value function coefficient for the loss calculation
        max_grad_norm=0.5,   # Maximum value for the gradient clipping
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("📊 Model configuration:")
    print(f"  - Policy: MlpPolicy")
    print(f"  - Learning rate: 3e-4")
    print(f"  - Steps per update: 2048")
    print(f"  - Batch size: 64")
    print(f"  - Training epochs: 10")
    print(f"  - Discount factor (gamma): 0.99")
    
    # Setup callbacks
    print("⚙️ Setting up training callbacks...")
    
    # Evaluation callback - evaluates the agent periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=5000,      # Evaluate every 5000 steps
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,     # Save every 10000 steps
        save_path="./models/",
        name_prefix="ppo_cpu_scheduler"
    )
    
    callbacks = [eval_callback, checkpoint_callback]
    
    # Train the agent
    print("🏋️ Starting training...")
    print(f"📈 Training for 100,000 timesteps...")
    print("💡 You can monitor progress with TensorBoard:")
    print("   tensorboard --logdir ./tensorboard_logs/")
    print("-" * 60)
    
    try:
        model.learn(
            total_timesteps=100000,
            callback=callbacks,
            tb_log_name="ppo_cpu_scheduler"
        )
        print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise
    
    # Save final model
    print("💾 Saving final model...")
    model.save("./models/ppo_cpu_scheduler_final")
    print("✅ Final model saved as: ./models/ppo_cpu_scheduler_final.zip")
    
    # Test the trained agent
    print("\n🧪 Testing trained agent...")
    test_agent(model, eval_env)
    
    print("\n🎉 Step 4 completed successfully!")
    print("📁 Check the following directories for results:")
    print("  - ./models/ - Saved model checkpoints")
    print("  - ./best_model/ - Best performing model")
    print("  - ./logs/ - Training and evaluation logs")
    print("  - ./tensorboard_logs/ - TensorBoard logs")

def test_agent(model, env, num_episodes=3):
    """Test the trained agent for a few episodes"""
    print(f"Running {num_episodes} test episodes...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps <= 3:  # Show first few steps
                print(f"Step {steps}: Action={action}, Reward={reward:.3f}")
            
            if done:
                break
        
        print(f"Episode {episode + 1} completed: {steps} steps, Total reward: {total_reward:.3f}")

if __name__ == "__main__":
    main()