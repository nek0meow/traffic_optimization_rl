import argparse
import os
import sys

import numpy as np
import torch

from config import SUMO_CONFIG_AUTO, SUMO_CONFIG_MANUAL
from envs.env import SumoTLSControlEnv, EnvConfig
from rl.nn_utils import masked_dist, tensor
from rl.shared_ppo import SharedTLSPolicy
from util.traci_utils import get_tls_data


def main() -> None:
    os.environ["GYM_DISABLE_WARNINGS"] = "1"

    seed = 42
    port = 25000
    greedy = True

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tls_data = get_tls_data(SUMO_CONFIG_AUTO)
    env = SumoTLSControlEnv(
        conf=EnvConfig(port=port),
        sumo_cfg=SUMO_CONFIG_MANUAL,
        tls_data=tls_data
    )

    model_path = "data\\gridnet3x3_shared_ppo_coop_2026-05-15\\models\\best_shared_policy.pt"
    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n
    policy = SharedTLSPolicy(obs_dim, int(action_dim)).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    try:
        obs, info = env.reset(seed=seed)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            with torch.no_grad():
                obs_t = tensor(obs, device)
                mask_t = tensor(info['action_mask'], device)
                logits, _ = policy(obs_t)
                dist = masked_dist(logits, mask_t)
                if greedy:
                    actions = dist.probs.argmax(dim=-1)
                else:
                    actions = dist.sample()

            obs, reward, terminated, truncated, info = env.step(
                actions.cpu().numpy()
            )
            print(
                f"step={info['step']} reward={reward:.3f} "
                f"actions={info['actions'].tolist()} "
                f"obs_min={obs.min():.3f} obs_max={obs.max():.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
