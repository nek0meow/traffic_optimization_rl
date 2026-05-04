import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from datetime import date
import numpy as np

from env import SumoTLSControlEnv
from stats_wrapper import StatsWrapper
from traci_utils import get_tls_data
from episode_stats_callback import EpisodeStatsCallback
from plot_utils import plots_over_time

def save_dict(d: dict[str, list[float | int]], path: str):
    if not (len(path) > 4 and path[-4::] == '.npz'):
        path += '.npz'

    npz_path = path
    np.savez_compressed(npz_path, **d)

def load_dict(path: str) -> dict:
    data = np.load(path)
    return {key: data[key] for key in data.files}

PROJ_DIR: str = os.path.dirname(os.path.abspath(__file__) + '\\..\\..\\..')
MAP_NAME = 'gridnet'
DATA_DIR = 'data'
SUMO_CONF_FILE: str = os.path.join('maps', MAP_NAME, f'{MAP_NAME}.sumocfg')
THREAD_COUNT = 12

SUMO_CONFIG_MANUAL: list[str] = [
    'sumo-gui',
    '-c', os.path.join(PROJ_DIR, SUMO_CONF_FILE),
    '--step-length', '1',
    '--delay', '40',
    '-e', '400',
    '-S', '-Q'
]
SUMO_CONFIG_AUTO: list[str] = [
    'sumo',
    '-c', os.path.join(PROJ_DIR, SUMO_CONF_FILE),
    '--step-length', '1',
    '-e', '400'
]



if __name__ == "__main__":
    os.environ["GYM_DISABLE_WARNINGS"] = "1"

    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please define SUMO_HOME")

    
    algorithm = "PPO"
    learning_timesteps = 10000

    model_name = MAP_NAME + '_' + algorithm + '_' + str(date.today())
    data_path = os.path.join(PROJ_DIR, DATA_DIR, model_name)
    plot_dir_path = os.path.join(data_path, 'plots')
    model_dir_path = os.path.join(data_path, 'models')
    history_dir_path = os.path.join(data_path, 'history')
    [os.makedirs(path, exist_ok=True) for path in [plot_dir_path, model_dir_path, history_dir_path]]

    tls_data = get_tls_data(SUMO_CONFIG_AUTO)

    # env factory
    def make_env(port):
        def _init():
            env =  SumoTLSControlEnv(
                sumo_cfg=SUMO_CONFIG_AUTO,
                tls_data=tls_data, 
                port=port,
            )
            return StatsWrapper(env)
        return _init

    env = SubprocVecEnv([make_env(24000 + i) for i in range(THREAD_COUNT)])

    eval_env_base = SumoTLSControlEnv(SUMO_CONFIG_AUTO, tls_data, 25000)
    eval_env = Monitor(eval_env_base)


    # early stopping
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=10,
        verbose=1
    )

    adjusted_eval_freq = max(3000 // THREAD_COUNT, 1)
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=adjusted_eval_freq,
        best_model_save_path=os.path.join(model_dir_path),
        callback_after_eval=stop_callback,
        verbose=1
    )

    episode_stats_callback = EpisodeStatsCallback()

    model = PPO(
        "MlpPolicy",
        env,       
        learning_rate=1e-3,
        n_steps=256,
        verbose=1
    )

    model.learn(
        total_timesteps=learning_timesteps,
        callback=[eval_callback, episode_stats_callback],
        progress_bar=True
    )

    # TODO: general history not conflicting with threads

    episode_history = {
        'avg_speeds': episode_stats_callback.avg_speeds,
        'avg_vehicles': episode_stats_callback.avg_vehicles,
        'reward_sums': episode_stats_callback.reward_sums
    }
    episode_n = list(range(len(episode_history['avg_speeds'])))

    plots_over_time(episode_n, 
                    list(episode_history.values()),
                    list(episode_history.keys()), 
                    save_to=os.path.join(plot_dir_path, f'overall_{model_name}.png'), 
                    show=True)
    save_dict(episode_history, os.path.join(history_dir_path, model_name))
    
    env.close()

