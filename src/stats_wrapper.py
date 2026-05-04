import gymnasium as gym
import os
import traci
import numpy as np

from env import SumoTLSControlEnv
from history import History
from traci_utils import calc_cur_stats


class StatsWrapper(gym.Wrapper):
    def __init__(self, 
                 env: gym.Env, 
                 step_freq: int = 3):
        super().__init__(env)

        self.episode_history = History(['time', 'n_vehicles', 'avg_speed', 'reward_sum'])
        self.step_freq = step_freq
        self.epoch = 1


    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        cur_step = info['step']

        if cur_step % self.step_freq == 0:
            conn = traci.getConnection(info['conn'])
            time, n, avg_speed = calc_cur_stats(conn)
            self.episode_history.update([time, n, avg_speed, float(reward)])

        if terminated or truncated:
            avg_speed = np.mean(self.episode_history['avg_speed'])
            avg_n_vehicles = np.mean(self.episode_history['n_vehicles'])
            reward = np.sum(self.episode_history['reward_sum'])

            info['episode_info'] = {
                'epoch': self.epoch,
                'avg_speed': avg_speed,
                'avg_n_vehicles': avg_n_vehicles,
                'reward_sum': reward
            }

        return obs, reward, terminated, truncated, info
    

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.episode_history.clear()
        self.epoch += 1

        return obs, info
