from dataclasses import dataclass
from config import CPU_THREADS

@dataclass
class TrainConfig:
    total_env_steps: int = 10_000
    n_envs: int = CPU_THREADS
    base_seed: int = 42
    base_port: int = 24000
    rollout_steps: int = 128
    update_steps: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    reward_scale: float = 0.05
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    stat_every: int = 5
    eval_every: int = 2_000
    eval_episodes: int = 3
    eval_seed: int = 10_000
    early_stop_evals: int = 15


@dataclass
class EnvConfig:
    delta_time: int = 5
    yellow_time: int = 3
    num_steps: int = 100
    max_lanes: int = 12
    max_phases: int = 8
    max_neighbors: int = 4
    neighbor_reward_coef: float = 0.2
    speed_scale: int = 10
    switch_cost: int = 5
    distance_scale: float = 500.0
