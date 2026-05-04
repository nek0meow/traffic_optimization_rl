import gymnasium as gym
import numpy as np
import traci

from tls_map_data import TLSInfo, TlsMapData

SPEED_SCALE = 15
SWITCH_COST = 2
class SumoTLSControlEnv(gym.Env):
    def __init__(self, 
                 sumo_cfg: list[str],
                 tls_data: TlsMapData,
                 port: int,
                 delta_time: int = 5, 
                 yellow_time: int = 3, 
                 num_steps: int = 75,
                 max_lanes=12, 
                 max_phases=8):
        
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.port = port
        self.delta_time = delta_time
        self.yellow_time = delta_time # !!!!! for now
        self.num_steps = num_steps
        self.max_lanes = max_lanes
        self.max_phases = max_phases
        self.cur_step: int = 1
        self.conn: traci.connection.Connection | None = None
        
        # agent-specific data
        self.tls_data = tls_data
        self.tls_ids = list(self.tls_data.tls.keys())
        self.num_agents = len(self.tls_ids)
        self.current_phases = np.zeros(self.num_agents, dtype=int)
        self.time_in_phase = np.zeros(self.num_agents, dtype=int)
        self.last_waiting_times = np.zeros(self.num_agents, dtype=float)

        phase_counts = [self.tls_data.tls[tls_id].num_phases for tls_id in self.tls_ids]
        self.action_space = gym.spaces.MultiDiscrete(phase_counts)

        one_obs_size = (self.max_lanes * 3) + self.max_phases + 1
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.num_agents, one_obs_size), dtype=np.float32
        )


    def reset(self, seed = None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if self.conn is not None:
            self.conn.close()

        traci.start(self.sumo_cfg, label=f"sim_multi_{self.cur_step}", port=self.port)
        self.conn = traci.getConnection(f"sim_multi_{self.cur_step}")

        for _ in range(10):
            self.conn.simulationStep()

        self.cur_step = 1
        self.current_phases.fill(0)
        self.time_in_phase.fill(0)
        self.last_waiting_times.fill(0)

        info = {'step': self.cur_step, 
                'conn': self.conn.getLabel()}
        
        return self._get_obs(), info


    def _get_obs(self) -> np.ndarray:
        all_obs = []
        for i, tls_id in enumerate(self.tls_ids):
            lanes = self.tls_data.tls[tls_id].lanes
            occupancy = [self.conn.lane.getLastStepOccupancy(l) for l in lanes]
            speed = [self.conn.lane.getLastStepMeanSpeed(l) / SPEED_SCALE for l in lanes]
            wait = [min(self.conn.lane.getWaitingTime(l) / 100.0, 1.0) for l in lanes]

            padding_size = self.max_lanes - len(lanes)
            obs_parts = [
                [0.0] * padding_size + occupancy,
                [0.0] * padding_size + speed,
                [0.0] * padding_size + wait
            ]

            phase_one_hot = np.zeros(self.max_phases, dtype=np.float32)
            phase_one_hot[self.current_phases[i]] = 1
            # TODO: add information about phase content to the observation

            norm_time_in_phase = [min(self.time_in_phase[i] / 60.0, 1.0)]

            agent_obs = np.concatenate([*obs_parts, phase_one_hot, norm_time_in_phase])
            all_obs.append(agent_obs)
            
        return np.array(all_obs, dtype=np.float32)


    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, str]]:
        reward: float = 0.0

        for i, tls_id in enumerate(self.tls_ids):
            action = actions[i]

            if action != self.current_phases[i]:
                reward -= SWITCH_COST
                self._set_yellow_phase(tls_id, action)
                self.current_phases[i] = action
                self.time_in_phase[i] = 0
            else:
                self.time_in_phase[i] += self.delta_time
                
        for _ in range(self.delta_time):
            self.conn.simulationStep()

        for i, tid in enumerate(self.tls_ids):
            # change from yellow, or stay in the same phase
            self.conn.trafficlight.setRedYellowGreenState(tid, self.tls_data.tls[tid].green_states[actions[i]])

            lanes = self.tls_data.tls[tid].lanes
            out_lanes = self.tls_data.tls[tid].out_lanes

            curr_wait = sum([self.conn.lane.getWaitingTime(l) for l in lanes])
            reward_waiting = self.last_waiting_times[i] - curr_wait
            
            flow_reward = abs(sum([self.conn.lane.getLastStepVehicleNumber(l) for l in lanes]) - 
                           sum([self.conn.lane.getLastStepVehicleNumber(l) for l in out_lanes]))
            
            reward += (reward_waiting * 0.25) - (flow_reward * 0.5)
            
            self.last_waiting_times[i] = curr_wait

        
        obs = self._get_obs()

        terminated: bool = False
        truncated: bool = self.cur_step >= self.num_steps
        info = {'step': self.cur_step, 
                'conn': self.conn.getLabel()}

        self.cur_step += 1
        return obs, reward, terminated, truncated, info


    def _set_yellow_phase(self, tls_id: str, new_phase: int) -> None:
        # e.g. "GGggrrrr"
        current_state: str = self.conn.trafficlight.getRedYellowGreenState(tls_id)
        target_state: str = self.tls_data.tls[tls_id].green_states[new_phase]

        # G, g -> y
        yellow_state = ''.join(['y' if (cur in ['G', 'g'] and new == 'r') else cur for cur, new in zip(current_state, target_state)])
        self.conn.trafficlight.setRedYellowGreenState(tls_id, yellow_state)


    def close(self):
        if self.conn != None:
            self.conn.close()
        
        self.active = False
        self.conn = None


