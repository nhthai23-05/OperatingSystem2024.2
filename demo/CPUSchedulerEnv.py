import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CPUSchedulerEnv(gym.Env):
    """
    CPU Scheduler Environment - Improved version for Operating Systems project.
    State per process: [cpu_util, mem_util, prio_norm, io_wait, ctx_switches, is_running, remaining_time]
    All values normalized to [0, 1] for consistency.
    """

    def __init__(self, num_processes=5, max_steps=50):
        super(CPUSchedulerEnv, self).__init__()
        self.num_processes = num_processes
        self.max_steps = max_steps

        # Normalized observation space: all values between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_processes * 7,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_processes + 1)  # +1 for 'noop'

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)

    def _generate_state(self):
        cpu = np.random.uniform(0.2, 0.8, size=self.num_processes)      # 0-1 normalized
        mem = np.random.uniform(0.2, 0.8, size=self.num_processes)
        prio = np.random.choice([0.0, 0.5, 1.0], size=self.num_processes)
        io = np.random.uniform(0, 0.5, size=self.num_processes)         # 0-1
        ctx = np.random.uniform(0, 0.2, size=self.num_processes)        # normalized
        is_running = np.zeros(self.num_processes)
        remaining_time = np.random.randint(10, 51, size=self.num_processes) / 50.0  # normalized

        return np.vstack([cpu, mem, prio, io, ctx, is_running, remaining_time]).T

    def reset(self):
        self.state = self._generate_state()
        self.steps = 0
        self.last_action = None
        self.wait_times = np.zeros(self.num_processes)
        return self.state.flatten().astype(np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        self.steps += 1

        state = self.state.copy()
        reward = 0
        done = False
        info = {}

        # Tăng thời gian chờ của tất cả tiến trình
        self.wait_times += 1

        # Reset is_running flags
        state[:, 5] = 0

        if action == self.num_processes:
            # NOOP (không chọn tiến trình nào)
            reward = -0.1
            info['noop'] = True
            self.state = state
            done = self.steps >= self.max_steps
            return state.flatten().astype(np.float32), reward, done, info

        # Chọn tiến trình
        selection = state[action].copy()
        selection[5] = 1  # is_running
        self.wait_times[action] = 0

        # Giảm CPU, MEM, IO khi được chạy
        selection[0] = max(selection[0] * 0.7, 0)
        selection[1] = max(selection[1] * 0.9, 0)
        selection[3] = max(selection[3] - 0.2, 0)

        # Tăng CTX switch nhẹ cho tất cả, trừ tiến trình đang chạy
        state[:, 4] = np.minimum(state[:, 4] + 0.01, 1.0)
        selection[4] = 0.0  # reset CTX cho tiến trình đang chạy

        # Giảm thời gian còn lại
        selection[6] -= 0.02

        # Phần thưởng
        throughput_reward = (1.0 - selection[0]) * 0.5
        mem_penalty = -selection[1] * 0.3
        prio_bonus = selection[2] * 1.5
        io_penalty = -1.0 if selection[3] > 0.4 else 0
        fairness_penalty = -0.01 * np.mean(self.wait_times)
        switch_penalty = -0.01 * np.mean(state[:, 4])
        switch_action_penalty = -0.5 if self.last_action is not None and self.last_action != action else 0

        finished = False
        if selection[6] <= 0:
            finished = True
            reward += 3.0  # phần thưởng thêm khi hoàn tất
            selection = self._generate_state()[0]

        state[action] = selection
        self.state = state
        self.last_action = action

        reward += (
            throughput_reward + mem_penalty + prio_bonus +
            io_penalty + fairness_penalty + switch_penalty + switch_action_penalty
        )

        done = self.steps >= self.max_steps
        obs = self.state.flatten().astype(np.float32)
        info.update({
            'throughput_reward': throughput_reward,
            'mem_penalty': mem_penalty,
            'prio_bonus': prio_bonus,
            'io_penalty': io_penalty,
            'fairness_penalty': fairness_penalty,
            'switch_penalty': switch_penalty,
            'switch_action_penalty': switch_action_penalty,
            'finished': finished
        })
        return obs, reward, done, info

    def render(self, mode='human'):
        df = pd.DataFrame(self.state, columns=[
            'CPU', 'MEM', 'PRIO', 'IO_WAIT', 'CTX', 'IS_RUN', 'REMAIN'
        ])
        print(f"Step: {self.steps}")
        print(df.round(3))