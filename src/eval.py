import numpy as np
import torch

from rl.shared_ppo import SharedTLSPolicy
from envs.env import SumoTLSControlEnv
from util.config_dataclasses import EnvConfig
from rl.nn_utils import tensor, masked_dist
from util.traci_utils import calc_cur_stats


def evaluate(
    policy: SharedTLSPolicy,
    tls_data,
    device: torch.device,
    port: int,
    env_config: EnvConfig,
    episodes: int = 3,
    base_seed: int | None = None,
    verbose = True
) -> tuple[float, dict[str, float]]:
    rewards = []
    avg_speeds = []
    avg_vehicles = []
    avg_waits = []
    action_counts = np.zeros(0, dtype=int)

    was_training = policy.training
    policy.eval()

    try:
        if verbose:
            print("STARTING EVAL")
        for episode_idx in range(episodes):
            env = SumoTLSControlEnv(env_config, tls_data, port=port)
            try:
                seed = None if base_seed is None else base_seed + episode_idx
                obs, info = env.reset(seed=seed)
                done = False
                episode_rewards = []
                speeds = []
                vehicles = []
                waits = []

                while not done:
                    with torch.no_grad():
                        obs_t = tensor(obs, device)
                        mask_t = tensor(info['action_mask'], device)
                        logits, _ = policy(obs_t)
                        actions = masked_dist(logits, mask_t).probs.argmax(dim=-1)

                    obs, reward, terminated, truncated, info = env.step(
                        actions.cpu().numpy()
                    )
                    if action_counts.size == 0:
                        action_counts = np.zeros(env.max_phases, dtype=int)
                    action_counts += np.bincount(
                        actions.cpu().numpy(),
                        minlength=env.max_phases
                    )
                    done = terminated or truncated
                    episode_rewards.append(float(reward))

                    _, n_vehicles, avg_speed, avg_wait = calc_cur_stats(env)
                    speeds.append(avg_speed)
                    vehicles.append(n_vehicles)
                    waits.append(avg_wait)

                rewards.append(float(np.sum(episode_rewards)))
                avg_speeds.append(float(np.mean(speeds)) if speeds else 0.0)
                avg_vehicles.append(float(np.mean(vehicles)) if vehicles else 0.0)
                avg_waits.append(float(np.mean(waits)) if waits else 0.0)
            finally:
                env.close()
    finally:
        if was_training:
            policy.train()


    info = {
        'reward_sum': float(np.mean(rewards)),
        'avg_speed': float(np.mean(avg_speeds)),
        'avg_n_vehicles': float(np.mean(avg_vehicles)),
        'avg_wait': float(np.mean(avg_waits)),
        'action_counts': action_counts.tolist(),
    }

    if (verbose):
        print(f"reward sum={info['reward_sum']:.2f}, avg_speed: {info['avg_speed']:.2f}, avg_wait: {info['avg_wait']:.2f}")
        print("action stats: ", info['action_counts'])

    return (info['reward_sum'], info)
