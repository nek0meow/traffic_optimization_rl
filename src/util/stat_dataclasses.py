from dataclasses import dataclass
import numpy as np

@dataclass
class StepStat:
    episode: int
    avg_speed: float
    n_vehicles: int
    avg_wait: float
    reward: float

@dataclass
class OverallStat:
    episode: int
    avg_speed: float
    avg_vehicles: float
    avg_wait: float
    avg_reward: float

def aggregate_eq_episode_steps(step_infos: list[StepStat], episode: int) -> OverallStat:
    avg_speed = np.mean([el.avg_speed for el in step_infos])
    avg_vehicles = np.mean([el.n_vehicles for el in step_infos])
    avg_wait = np.mean([el.avg_wait for el in step_infos])
    avg_reward = np.mean([el.reward for el in step_infos])

    return OverallStat(
        episode=episode,
        avg_speed=float(avg_speed),
        avg_vehicles=float(avg_vehicles),
        avg_wait=float(avg_wait),
        avg_reward=float(avg_reward)
    )