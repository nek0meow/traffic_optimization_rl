import numpy as np
import torch
import os

from config import PROJ_DIR
from util.traci_utils import get_tls_data
from util.config_util import transform_dict_configs
from envs.env import SumoTLSControlEnv
from rl.nn_utils import masked_dist, tensor
from rl.shared_ppo import SharedTLSPolicy


class GUIManager:
    def __init__(self, config: dict):
        self.config = config
        self.map_name = config["map"]
        self.model_path = config["model_path"]

        _, self.env_conf = transform_dict_configs(config)

        self.seed = config.get("seed", 42)
        self.port = config.get("port", 26000)
        self.greedy = config.get("greedy", True)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.sumo_cfg_auto = [
            'sumo',
            '-c',
            f'{PROJ_DIR}/maps/{self.map_name}/{self.map_name}.sumocfg',
            '--no-step-log',
            '--no-warnings'
        ]

        self.sumo_cfg_gui = [
            'sumo-gui',
            '-c',
            os.path.join(PROJ_DIR, 'maps', self.map_name, self.map_name + '.sumocfg'),
            '--step-length', '1',
            '--delay', '40',
            '-S', '-Q',
            '--no-step-log',
            '--no-warnings'
        ]

        self.env = None
        self.policy = None

    def setup_env(self):
        tls_data = get_tls_data(self.sumo_cfg_auto)

        self.env = SumoTLSControlEnv(
            conf=self.env_conf,
            sumo_cfg=self.sumo_cfg_gui,
            tls_data=tls_data,
            port=self.port
        )

    def setup_policy(self):
        obs_dim = self.env.single_observation_space.shape[0]
        action_dim = self.env.single_action_space.n

        self.policy = SharedTLSPolicy(
            obs_dim,
            int(action_dim)
        ).to(self.device)

        self.policy.load_state_dict(
            torch.load(
                self.model_path,
                map_location=self.device
            )
        )

        self.policy.eval()

    def select_actions(self, obs, info):
        with torch.no_grad():
            obs_t = tensor(obs, self.device)
            mask_t = tensor(info['action_mask'], self.device)

            logits, _ = self.policy(obs_t)

            dist = masked_dist(logits, mask_t)

            if self.greedy:
                actions = dist.probs.argmax(dim=-1)
            else:
                actions = dist.sample()

        return actions.cpu().numpy()

    def run(self):
        self.setup_env()
        self.setup_policy()
        assert(self.env is not None)

        try:
            obs, info = self.env.reset(seed=self.seed)
            terminated = False
            truncated = False

            while not (terminated or truncated):
                actions = self.select_actions(obs, info)
                obs, reward, terminated, truncated, info = (
                    self.env.step(actions)
                )
        finally:
            if self.env is not None:
                self.env.close()