import os
import sys
from datetime import date
import numpy as np
import torch

from util.history import OverallHistory
from util.traci_utils import get_tls_data
from util.config_util import transform_dict_configs
from train import PPOTrainer
from rl.shared_ppo import SharedTLSPolicy
from config import CPU_THREADS, PROJ_DIR, DATA_DIR, SUMO_CONFIG_AUTO
from envs.env import SumoTLSControlEnv
from eval import evaluate


def save_dict(d: dict[str, list[float | int]], path: str) -> None:
    if not path.endswith('.npz'):
        path += '.npz'

    np.savez_compressed(path, **d)


def run_training(
        config_dict: dict
    ):
    print("initializing...")
    model_name = config_dict["map"] + "_" + config_dict["model_name"]

    train_conf, env_conf = transform_dict_configs(config_dict)
    torch_threads = max(1, CPU_THREADS - train_conf.n_envs)
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)

    device = torch.device("cpu")

    data_path = os.path.join(PROJ_DIR, DATA_DIR, model_name)
    plot_dir_path = os.path.join(data_path, 'plots')
    model_dir_path = os.path.join(data_path, 'models')
    history_dir_path = os.path.join(data_path, 'history')

    for path in [plot_dir_path, model_dir_path, history_dir_path]:
        os.makedirs(path, exist_ok=True)

    tls_data = get_tls_data(SUMO_CONFIG_AUTO)

    history = OverallHistory()
    envs = [SumoTLSControlEnv(env_conf, tls_data, port=24000+i) for i in range(CPU_THREADS)]
    obs_dim = envs[0].single_observation_space.shape[0]
    action_dim = envs[0].single_action_space.n
    policy = SharedTLSPolicy(obs_dim, int(action_dim)).to(device)

    try:
        trainer = PPOTrainer(envs, policy, device, history, train_conf)
        trainer.train(
            model_dir_path=model_dir_path,
            eval_cb=lambda: evaluate(policy, tls_data, device, port=25000, env_config=env_conf, base_seed=4242)[0],
        )
    finally:
        torch.save(
            policy.state_dict(),
            os.path.join(model_dir_path, 'last_shared_policy.pt')
        )
        for env in envs:
            env.close()

    history.make_plots(
        save_to=os.path.join(plot_dir_path, f'overall_{model_name}.png'),
        show=True
    )
    save_dict(history.to_dict(), os.path.join(history_dir_path, model_name))


if __name__ == '__main__':
    res = run_training(
    {
    "model_name": "shared-ppo-standard",
    "map": "gridnet3x3",
    "train": {
        "total_env_steps": 150000,
        "rollout_steps": 128,
        "learning_rate": 0.0003,
        "reward_scale": 0.05,
        "early_stop_evals": 15
    },
    "env": {
        "delta_time": 5,
        "yellow_time": 10,
        "num_steps": 100
    }})
