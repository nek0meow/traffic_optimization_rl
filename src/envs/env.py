import gymnasium as gym
import numpy as np
import traci
import traci.constants as tc
from dataclasses import dataclass
from typing import cast

from util.tls_map_data import TlsMapData
from collections import deque
from config import SUMO_CONFIG_AUTO

SPEED_SCALE = 15
SWITCH_COST = 2
DISTANCE_SCALE = 500.0

@dataclass
class EnvConfig:
    port: int = 24000
    delta_time: int = 5
    yellow_time: int = 10
    num_steps: int = 100
    max_lanes: int = 12
    max_phases: int = 8
    max_neighbors: int = 4
    neighbor_reward_coef: float = 0.2

class SumoTLSControlEnv(gym.Env):
    def __init__(self, 
                 conf: EnvConfig,
                 tls_data: TlsMapData,
                 sumo_cfg: list[str] = SUMO_CONFIG_AUTO
    ):
        super().__init__()

        self.tls_data = tls_data
        self.sumo_cfg = sumo_cfg
        self.port = conf.port
        self.delta_time = conf.delta_time
        self.yellow_time = conf.yellow_time
        self.num_steps = conf.num_steps
        self.max_lanes = conf.max_lanes
        self.max_phases = conf.max_phases
        self.max_neighbors = conf.max_neighbors
        self.neighbor_reward_coef = conf.neighbor_reward_coef

        self.real_time_step: int = 1
        self.cur_step: int = 1
        self.episode_idx: int = 0
        self.conn: traci.connection.Connection | None = None
        
        self.tls_ids = list(self.tls_data.tls.keys())
        self.num_agents = len(self.tls_ids)
        if self.num_agents == 0:
            raise ValueError("TlsMapData must contain at least one controllable TLS.")

        max_num_phases = max(self.tls_data.tls[tls_id].num_phases for tls_id in self.tls_ids)
        if max_num_phases > self.max_phases:
            raise ValueError(
                f"max_phases={self.max_phases} is smaller than a TLS phase count "
                f"({max_num_phases}). Increase max_phases to use this map."
            )

        self.tls_index = {tls_id: i for i, tls_id in enumerate(self.tls_ids)}
        self.neighbors = self._build_neighbors()
        self.subscribed_lanes = self._get_all_lanes()
        self.current_phases = np.zeros(self.num_agents, dtype=int)
        self.time_in_phase = np.zeros(self.num_agents, dtype=int)
        self.last_waiting_times = np.zeros(self.num_agents, dtype=float)
        self.change_queue: deque[tuple[int, str, int]] = deque()

        tls_feature_size = (self.max_lanes * 3) + (self.max_phases * 2) + 1
        neighbor_feature_size = tls_feature_size + 1
        one_obs_size = tls_feature_size + (self.max_neighbors * neighbor_feature_size)

        self.single_observation_space = gym.spaces.Box(
            low=0, high=1, shape=(one_obs_size,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Discrete(self.max_phases)
        self.action_space = gym.spaces.MultiDiscrete(
            np.full(self.num_agents, self.max_phases, dtype=np.int64)
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.num_agents, one_obs_size), dtype=np.float32
        )

    def _build_neighbors(self) -> dict[str, list[tuple[str, float]]]:
        neighbors: dict[str, list[tuple[str, float]]] = {}

        for tls_id in self.tls_ids:
            filtered = [
                (neighbor_id, dist)
                for neighbor_id, dist in self.tls_data.adjacency.get(tls_id, [])
                if neighbor_id in self.tls_index
            ]
            neighbors[tls_id] = sorted(filtered, key=lambda item: item[1])[:self.max_neighbors]

        return neighbors


    def reset(self, seed = None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if self.conn is not None:
            self.conn.close()

        self.episode_idx += 1
        label = f"sim_multi_{self.port}_{self.episode_idx}"
        sumo_cfg = self._sumo_cfg_with_seed(seed)
        traci.start(sumo_cfg, label=label, port=self.port)
        self.conn = traci.getConnection(label)
        self._subscribe_lanes()
        self.real_time_step = 1
        self.cur_step = 1
        self.change_queue.clear()

        for _ in range(10):
            self.conn.simulationStep()
            self.real_time_step += 1

        self.current_phases.fill(0)
        self.time_in_phase.fill(0)
        self.last_waiting_times.fill(0)
        # we need it because tls would use static sumo program initially
        self._apply_all_green_phases(self.current_phases)

        info = self._get_info()
        return self._get_obs(), info


    def _get_all_lanes(self) -> list[str]:
        lanes = []

        for tls_info in self.tls_data.tls.values():
            lanes.extend(tls_info.lanes)
            lanes.extend(tls_info.out_lanes)

        return list(dict.fromkeys(lanes))


    def _get_obs(self) -> np.ndarray:
        # (agent_num, agent_obs_size)
        return np.array([
            self._get_agent_obs(tls_id) for tls_id in self.tls_ids
        ], dtype=np.float32)


    def _sumo_cfg_with_seed(self, seed: int | None) -> list[str]:
        cfg = list(self.sumo_cfg)
        if seed is None:
            return cfg

        if '--seed' in cfg:
            seed_idx = cfg.index('--seed') + 1
            if seed_idx < len(cfg):
                cfg[seed_idx] = str(seed)
                return cfg

        cfg.extend(['--seed', str(seed)])
        return cfg


    def _get_agent_obs(self, tls_id: str) -> np.ndarray:
        obs_parts = [self._get_tls_features(tls_id)]

        for neighbor_id, dist in self.neighbors[tls_id]:
            obs_parts.append(np.concatenate([
                self._get_tls_features(neighbor_id),
                np.array([
                    min(dist / DISTANCE_SCALE, 1.0),
                ], dtype=np.float32)
            ]))

        neighbor_feature_size = self._tls_feature_size() + 1
        missing_neighbors = self.max_neighbors - len(self.neighbors[tls_id])
        if missing_neighbors > 0:
            obs_parts.append(
                np.zeros(missing_neighbors * neighbor_feature_size, dtype=np.float32)
            )
    
        # (feature_size + (feature_size + 1) * max_neighbors, )
        return np.concatenate(obs_parts).astype(np.float32)


    def _get_tls_features(self, tls_id: str) -> np.ndarray:
        i = self.tls_index[tls_id]
        lanes = self.tls_data.tls[tls_id].lanes[:self.max_lanes]
        occupancy = [
            min(self._lane_occupancy(l) / 100.0, 1.0)
            for l in lanes
        ]
        speed = [
            min(self._lane_speed(l) / SPEED_SCALE, 1.0)
            for l in lanes
        ]
        wait = [min(self._lane_wait(l) / 100.0, 1.0) for l in lanes]

        padding_size = self.max_lanes - len(lanes)
        obs_parts = [
            [0.0] * padding_size + occupancy,
            [0.0] * padding_size + speed,
            [0.0] * padding_size + wait
        ]

        phase_one_hot = np.zeros(self.max_phases, dtype=np.float32)
        phase_one_hot[self.current_phases[i]] = 1.0

        valid_phase_mask = self._get_action_mask(tls_id)
        norm_time_in_phase = np.array(
            [min(self.time_in_phase[i] / 60.0, 1.0)],
            dtype=np.float32
        )

        # (max_lanes * 3 + max_phases * 2 + 1, )
        return np.concatenate([
            *obs_parts,
            phase_one_hot,
            valid_phase_mask,
            norm_time_in_phase
        ]).astype(np.float32)

    def _tls_feature_size(self) -> int:
        return (self.max_lanes * 3) + (self.max_phases * 2) + 1

    def _get_action_mask(self, tls_id: str | None = None) -> np.ndarray:
        if tls_id is None:
            return self._get_action_masks()

        mask = np.zeros(self.max_phases, dtype=np.float32)
        mask[:self.tls_data.tls[tls_id].num_phases] = 1.0
        return mask

    def _get_action_masks(self) -> np.ndarray:
        return np.array([
            self._get_action_mask(tls_id)
            for tls_id in self.tls_ids
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            'step': self.cur_step,
            'conn': self.conn.getLabel(),
            'tls_ids': self.tls_ids,
            'action_mask': self._get_action_masks()
        }

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        actions = np.asarray(actions, dtype=int).reshape(-1)
        
        if len(actions) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} actions, got {len(actions)}."
            )

        switch_penalties = np.zeros(self.num_agents, dtype=np.float32)
        effective_actions = actions.copy()
        phase_changed = np.zeros(self.num_agents, dtype=bool)

        for i, tls_id in enumerate(self.tls_ids):
            action = int(actions[i])

            if action < 0 or action >= self.tls_data.tls[tls_id].num_phases:
                effective_actions[i] = int(self.current_phases[i])
                switch_penalties[i] -= SWITCH_COST
                continue

            phase_changed[i] = action != self.current_phases[i]
            if phase_changed[i]:
                switch_penalties[i] -= SWITCH_COST
                self._set_yellow_phase(tls_id, action)

        for _ in range(1, self.delta_time + 1):
            self.conn.simulationStep()
            self.real_time_step += 1
            if len(self.change_queue) > 0 and self.real_time_step == self.change_queue[0][0]:
                self._apply_green_phases(self.change_queue)

        self.time_in_phase += self.delta_time

        for i, changed in enumerate(phase_changed):
            if changed:
                self.current_phases[i] = int(effective_actions[i])
                self.time_in_phase[i] = 0

        local_rewards = self._get_agent_rewards()
        agent_rewards = self._get_cooperative_rewards(local_rewards) + switch_penalties
        reward = float(np.mean(agent_rewards))

        terminated: bool = False
        truncated: bool = self.cur_step >= self.num_steps
        obs = self._get_obs()

        info = self._get_info()
        info['agent_rewards'] = agent_rewards
        info['actions'] = effective_actions

        self.cur_step += 1
        return obs, reward, terminated, truncated, info

    def _apply_all_green_phases(self, actions: np.ndarray) -> None:
        for i, tls_id in enumerate(self.tls_ids):
            action = int(actions[i])
            self.conn.trafficlight.setRedYellowGreenState(
                tls_id,
                self.tls_data.tls[tls_id].green_states[action]
            )

    def _apply_green_phases(self, q: deque[tuple[int, str, int]]) -> None:
        while len(q) > 0 and self.real_time_step == q[0][0]:
            _, tls_id, action = q.popleft()
            self.conn.trafficlight.setRedYellowGreenState(
                tls_id,
                self.tls_data.tls[tls_id].green_states[action]
            )

    def _get_agent_rewards(self) -> np.ndarray:
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        for i, tid in enumerate(self.tls_ids):
            lanes = self.tls_data.tls[tid].lanes
            out_lanes = self.tls_data.tls[tid].out_lanes

            curr_wait = sum([self._lane_wait(l) for l in lanes])
            reward_waiting = self.last_waiting_times[i] - curr_wait

            flow_reward = abs(
                sum([self._lane_vehicle_count(l) for l in lanes]) -
                sum([self._lane_vehicle_count(l) for l in out_lanes])
            )

            rewards[i] = (reward_waiting * 0.25) - (flow_reward * 0.5)
            self.last_waiting_times[i] = curr_wait

        return rewards

    def _get_cooperative_rewards(self, local_rewards: np.ndarray) -> np.ndarray:
        rewards = local_rewards.copy()

        for i, tls_id in enumerate(self.tls_ids):
            rewards[i] += self._get_neighbor_reward(tls_id, local_rewards)

        return rewards

    def _get_neighbor_reward(self, tls_id: str, agent_rewards: np.ndarray) -> float:
        weighted_reward = 0.0
        total_weight = 0.0

        for neighbor_id, dist in self.neighbors[tls_id]:
            neighbor_idx = self.tls_index[neighbor_id]
            weight = 1.0 / max(dist / DISTANCE_SCALE, 1.0)
            weighted_reward += float(agent_rewards[neighbor_idx]) * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return self.neighbor_reward_coef * weighted_reward / total_weight


    def _set_yellow_phase(self, tls_id: str, new_phase: int) -> None:
        # e.g. "GGggrrrr"
        current_state: str = self.conn.trafficlight.getRedYellowGreenState(tls_id)
        target_state: str = self.tls_data.tls[tls_id].green_states[new_phase]

        # G, g -> y
        yellow_state = ''.join(['y' if (cur in ['G', 'g'] and new == 'r') else cur for cur, new in zip(current_state, target_state)])
        self.conn.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
        self.change_queue.append((self.real_time_step + self.yellow_time, tls_id, new_phase))

    def _subscribe_lanes(self) -> None:
        var_ids = [
            tc.LAST_STEP_OCCUPANCY,
            tc.LAST_STEP_MEAN_SPEED,
            tc.VAR_WAITING_TIME,
            tc.LAST_STEP_VEHICLE_NUMBER,
        ]

        for lane_id in self.subscribed_lanes:
            try:
                self.conn.lane.subscribe(lane_id, var_ids)
            except traci.TraCIException:
                pass

    def _lane_subscription_value(self, lane_id: str, var_id: int):
        results = self.conn.lane.getSubscriptionResults(lane_id)
        if results is None:
            return None

        return results.get(var_id)

    def _lane_occupancy(self, lane_id: str) -> float:
        value = self._lane_subscription_value(lane_id, tc.LAST_STEP_OCCUPANCY)
        if value is None:
            return self.conn.lane.getLastStepOccupancy(lane_id)

        return float(value)

    def _lane_speed(self, lane_id: str) -> float:
        value = self._lane_subscription_value(lane_id, tc.LAST_STEP_MEAN_SPEED)
        if value is None:
            return self.conn.lane.getLastStepMeanSpeed(lane_id)

        return float(value)

    def _lane_wait(self, lane_id: str) -> float:
        value = self._lane_subscription_value(lane_id, tc.VAR_WAITING_TIME)
        if value is None:
            return self.conn.lane.getWaitingTime(lane_id)

        return float(value)

    def _lane_vehicle_count(self, lane_id: str) -> int:
        value = self._lane_subscription_value(lane_id, tc.LAST_STEP_VEHICLE_NUMBER)
        if value is None:
            return self.conn.lane.getLastStepVehicleNumber(lane_id)

        return int(value)


    def close(self):
        if self.conn != None:
            self.conn.close()
        
        self.conn = None

    def gather_statistics(self) -> tuple[float, int, float, float]:
        assert self.conn
        total_speed: float = 0
        total_waiting: float = 0
        vehicle_ids = self.conn.vehicle.getIDList()
        n = cast(int, len(vehicle_ids))

        for veh in vehicle_ids:
            total_speed += cast(float, self.conn.vehicle.getSpeed(veh))
            total_waiting += cast(float, traci.vehicle.get)

        time = cast(float, self.conn.simulation.getTime())
        avg_speed = total_speed / n if n > 0 else 0
        avg_waiting = total_waiting / n if n > 0 else 0
        return time, n, avg_speed, avg_waiting
