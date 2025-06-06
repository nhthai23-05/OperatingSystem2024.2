# demo/baselines.py
import numpy as np

def baseline_policy(observation, mode="fifo"):
    """
    Baseline scheduling policies for comparison with ML models.
    
    Args:
        observation: Flattened array of shape (35,) representing 5 processes × 7 features
        mode: Scheduling algorithm ("fifo", "rr", "sjf")
    
    Returns:
        action: Integer representing which process to schedule (0-4)
    """
    # Reshape observation from (35,) to (5, 7) matrix
    obs_matrix = observation.reshape(5, 7)
    
    # Extract features for each process
    # Feature indices: [cpu_util, mem_util, priority, io_wait, ctx_switches, is_running, remaining_time]
    remaining_times = obs_matrix[:, 6]  # Index 6: remaining execution time
    priorities = obs_matrix[:, 2]       # Index 2: priority (0.0, 0.5, 1.0)
    is_running = obs_matrix[:, 5]       # Index 5: is currently running
    
    # Find processes that are not finished (remaining_time > 0)
    active_processes = np.where(remaining_times > 0)[0]
    
    if len(active_processes) == 0:
        # No active processes, return NOOP action
        return 5
    
    if mode == "fifo":
        # First In, First Out - choose the first active process
        return active_processes[0]
    
    elif mode == "rr":
        # Round Robin - choose based on process ID rotation
        # Simple RR: select process with lowest ID among active ones
        return active_processes[0]
    
    elif mode == "sjf":
        # Shortest Job First - choose process with shortest remaining time
        active_remaining_times = remaining_times[active_processes]
        shortest_idx = np.argmin(active_remaining_times)
        return active_processes[shortest_idx]
    
    else:
        # Default to FIFO if unknown mode
        return active_processes[0]
