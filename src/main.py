import os
import sys
from datetime import date
import numpy as np
import torch

from util.history import OverallHistory
from util.traci_utils import get_tls_data
from my_train import TrainConfig, PPOTrainer
from rl.shared_ppo import SharedTLSPolicy
from config import *
from envs.env import SumoTLSControlEnv, EnvConfig
from eval import evaluate


def save_dict(d: dict[str, list[float | int]], path: str) -> None:
    if not path.endswith('.npz'):
        path += '.npz'

    np.savez_compressed(path, **d)

if __name__ == "__main__":
    os.environ["GYM_DISABLE_WARNINGS"] = "1"

    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please define SUMO_HOME")

    config = TrainConfig()

    # torch.manual_seed(config.seed)
    # np.random.seed(config.seed)

    torch_threads = max(1, CPU_THREADS - config.n_envs)
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)

    device = torch.device("cpu")

    model_name = f"{MAP_NAME}_shared_ppo_coop_{date.today()}"
    data_path = os.path.join(PROJ_DIR, DATA_DIR, model_name)
    plot_dir_path = os.path.join(data_path, 'plots')
    model_dir_path = os.path.join(data_path, 'models')
    history_dir_path = os.path.join(data_path, 'history')
    for path in [plot_dir_path, model_dir_path, history_dir_path]:
        os.makedirs(path, exist_ok=True)

    tls_data = get_tls_data(SUMO_CONFIG_AUTO)

    history = OverallHistory()
    envs = [SumoTLSControlEnv(EnvConfig(port=24000+i), tls_data) for i in range(CPU_THREADS)]
    obs_dim = envs[0].single_observation_space.shape[0]
    action_dim = envs[0].single_action_space.n
    policy = SharedTLSPolicy(obs_dim, int(action_dim)).to(device)

    try:
        trainer = PPOTrainer(envs, policy, device, history, config)
        trainer.train(
            model_dir_path=model_dir_path,
            eval_cb=lambda: evaluate(policy, tls_data, device, port=25000, base_seed=424242)[0],
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


