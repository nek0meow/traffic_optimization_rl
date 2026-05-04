import os
import sys
from stable_baselines3 import PPO

from main import *

if __name__ == "__main__":
    os.environ["GYM_DISABLE_WARNINGS"] = "1"

    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please define SUMO_HOME")

    tls_data = get_tls_data(SUMO_CONFIG_AUTO)

    model = PPO.load('data\\gridnet_PPO_2026-05-03\\models\\gridnet_PPO_2026-05-03\\best_model.zip')
    env = SumoTLSControlEnv(
        sumo_cfg=SUMO_CONFIG_MANUAL,
        tls_data=tls_data,
        port=30000
    )

    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)


