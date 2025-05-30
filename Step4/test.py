import sys
sys.path.append('../Step3/OSpro')
from cpuscheduler_env import CPUSchedulerEnv

# Test environment
env = CPUSchedulerEnv(num_processes=5, max_steps=10)
obs = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# Test one step
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
print(f"Step completed! Reward: {reward}")