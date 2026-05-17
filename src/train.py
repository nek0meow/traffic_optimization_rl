
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from envs.env import SumoTLSControlEnv
from util.history import OverallHistory
from rl.shared_ppo import SharedTLSPolicy
from rl.nn_utils import masked_dist, tensor, compute_gae_with_next_values
from util.traci_utils import calc_cur_stats
from config import CPU_THREADS
from util.stat_dataclasses import StepStat, aggregate_eq_episode_steps
from util.config_dataclasses import TrainConfig


@dataclass
class RolloutTask:
    env: SumoTLSControlEnv
    action: np.ndarray
    episode: int

class RolloutBuffer:
    def __init__(self):
        self.obs: list[torch.Tensor] = []
        self.mask: list[torch.Tensor] = []
        self.action: list[torch.Tensor] = []
        self.log_prob: list[torch.Tensor] = []
        self.value: list[torch.Tensor] = []
        self.next_value: list[torch.Tensor] = []
        self.reward: list[torch.Tensor] = []
        self.done: list[torch.Tensor] = []
    
    def update(self, obs, mask, action, log_prob, value, next_value, reward, done):
        self.obs.append(obs)
        self.mask.append(mask)
        self.action.append(action)
        self.log_prob.append(log_prob)
        self.value.append(value)
        self.next_value.append(next_value)
        self.reward.append(reward)
        self.done.append(done)

def step_env(env: SumoTLSControlEnv, actions: np.ndarray):
    return env.step(actions)

class PPOTrainer:
    def __init__(
        self,
        envs: list[SumoTLSControlEnv],
        policy: SharedTLSPolicy,
        device: torch.device,
        history: OverallHistory,
        config: TrainConfig = TrainConfig(),
    ):

        self.config = config
        self.device = device
        self.envs = envs
        self.policy = policy
        self.history = history

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )

        self.n_envs = self.config.n_envs
        self.global_step = 1
        self.global_episode = 1
        self.epoch = 1
        self.best_metric_result = float('-inf')
        self.conseq_no_improve = 0

        self.stat_infos: list[list[StepStat]] = [[] for _ in range(self.n_envs)]



    def train(self, model_dir_path: str, eval_cb=None, early_stopping=True):
        obs_list = []
        info_list = []
        eval_periods = 0

        for env_idx, env in enumerate(self.envs):
            obs, info = env.reset(
                seed=self.config.base_seed + env_idx
            )
            obs_list.append(obs)
            info_list.append(info)

        print("TRAINING START")
        while self.global_step < self.config.total_env_steps:
            buffer, obs_list, info_list = self.collect_rollout(
                obs_list,
                info_list
            )
            self.update_policy(buffer)
            print(f"step={self.global_step}")

            cur_eval_period = self.global_step // self.config.eval_every
            if eval_cb and cur_eval_period > eval_periods:
                eval_periods = cur_eval_period
                metric = eval_cb()
                if metric > self.best_metric_result:
                    self.best_metric_result = metric
                    self.conseq_no_improve = 0
                    print(f"new best. Saving...")
                    torch.save(
                        self.policy.state_dict(),
                        os.path.join(model_dir_path, 'best_shared_policy.pt')
                    )
                else:
                    self.conseq_no_improve += 1
                    if self.conseq_no_improve >= self.config.early_stop_evals and early_stopping:
                        print(f'Ended learning on step {self.global_step}: no imporovements for {self.conseq_no_improve} epochs')
                        break
        print("TRAINING END")


    def collect_rollout(
        self,
        obs_list,
        info_list
    ):
        done = False
        buffer = RolloutBuffer()
        with ThreadPoolExecutor(max_workers=self.n_envs) as executor:
            for _ in range(self.config.rollout_steps):
                obs_arr = np.stack(obs_list)
                mask_arr = np.stack([info["action_mask"] for info in info_list])

                env_count, agent_count, obs_dim = obs_arr.shape

                obs_t = tensor(obs_arr, self.device).reshape(-1, obs_dim)
                mask_t = tensor(mask_arr, self.device).reshape(-1, mask_arr.shape[-1])

                with torch.no_grad():
                    logits, values = self.policy(obs_t)
                    values = values.squeeze(-1)
                    dist = masked_dist(logits, mask_t)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

                action_arr = actions.cpu().numpy().reshape(env_count, agent_count)

                tasks = [
                    RolloutTask(env, action_arr[env_idx], self.global_episode + env_idx)
                    for env_idx, env in enumerate(self.envs)
                ]
                results = list(
                    executor.map(
                        self.rollout_worker,
                        tasks
                    )
                )

                next_obs_list = []
                next_info_list = []

                reward_rows = []
                done_rows = []
                ended_eps = 0

                for env_idx, result in enumerate(results):
                    next_obs, step_info, rewards, done, step_stat = [
                        result[key] for key in ['next_obs', 'step_info', 'rewards', 'done', 'step_stat']
                    ]
                    
                    reward_rows.append(rewards)
                    done_rows.append(np.full(agent_count, float(done), dtype=np.float32))

                    if step_stat is not None:
                        self.stat_infos[env_idx].append(step_stat)
                    
                    if done:
                        self.history.add_info(aggregate_eq_episode_steps(self.stat_infos[env_idx], self.global_episode + env_idx))
                        self.stat_infos[env_idx].clear()

                        next_obs, step_info = self.envs[env_idx].reset(
                            seed=self.config.base_seed + self.global_step + env_idx
                        )
                        ended_eps += 1

                    next_obs_list.append(next_obs)
                    next_info_list.append(step_info)

                next_obs_arr = np.stack(next_obs_list)
                next_obs_t = tensor(next_obs_arr, self.device).reshape(-1, obs_dim)

                with torch.no_grad():
                    _, next_values = self.policy(next_obs_t)
                    next_values = next_values.squeeze(-1)

                buffer.update(
                    obs_t,
                    mask_t,
                    actions,
                    log_probs,
                    values,
                    next_values,
                    tensor(np.concatenate(reward_rows), self.device) * self.config.reward_scale,
                    tensor(np.concatenate(done_rows), self.device)
                )

                obs_list = next_obs_list
                info_list = next_info_list

                self.global_step += self.n_envs
                self.global_episode += ended_eps

                if self.global_step >= self.config.total_env_steps:
                    break
            self.epoch += 1
        
        return buffer, obs_list, info_list
    
    def rollout_worker(self, task: RolloutTask):
        env, action, episode = task.env, task.action, task.episode
        next_obs, mean_step_reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        step_stat = None

        if env.cur_step % self.config.stat_every == 0:
            _, n_vehicles, avg_speed, avg_wait = calc_cur_stats(env)
            step_stat = StepStat(
                episode=episode, # not used
                avg_speed=avg_speed,
                n_vehicles=n_vehicles,
                avg_wait=avg_wait,
                reward=mean_step_reward
            )

        rewards = np.asarray(
            step_info["agent_rewards"],
            dtype=np.float32
        )
        if done:
            next_obs, step_info = env.reset(
                seed=self.config.base_seed + episode
            )

        return {
            "next_obs": next_obs,
            "step_info": step_info,
            "rewards": rewards,
            "done": done,
            "step_stat": step_stat
        }


    def update_policy(self, buffer: RolloutBuffer):
        obs = torch.stack(buffer.obs).detach()
        masks = torch.stack(buffer.mask).detach()
        actions = torch.stack(buffer.action).detach()
        old_log_probs = torch.stack(buffer.log_prob).detach()
        values = torch.stack(buffer.value).detach()
        next_values = torch.stack(buffer.next_value).detach()
        rewards = torch.stack(buffer.reward).detach()
        dones = torch.stack(buffer.done).detach()

        returns, advantages = compute_gae_with_next_values(
            rewards,
            dones,
            values,
            next_values,
            self.config.gamma,
            self.config.gae_lambda
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_masks = masks.reshape(-1, masks.shape[-1])
        flat_actions = actions.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)

        batch_size = flat_obs.shape[0]

        for _ in range(self.config.update_steps):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, self.config.minibatch_size):
                mb_idx = indices[start:start + self.config.minibatch_size]

                logits, new_values = self.policy(flat_obs[mb_idx])
                new_values = new_values.squeeze(-1)
                dist = masked_dist(logits, flat_masks[mb_idx])

                new_log_probs = dist.log_prob(flat_actions[mb_idx])
                entropy = dist.entropy().mean()
                ratio = (new_log_probs - flat_old_log_probs[mb_idx]).exp()

                pg_loss_1 = (
                    -flat_advantages[mb_idx]
                    * ratio
                )

                pg_loss_2 = (
                    -flat_advantages[mb_idx]
                    * torch.clamp(
                        ratio,
                        1.0 - self.config.clip_coef,
                        1.0 + self.config.clip_coef
                    )
                )

                policy_loss = torch.max(
                    pg_loss_1,
                    pg_loss_2
                ).mean()
                old_values_mb = values.reshape(-1)[mb_idx]

                value_pred_clipped = old_values_mb + (
                    new_values - old_values_mb
                ).clamp(
                    -self.config.clip_coef,
                    self.config.clip_coef
                )

                value_loss_unclipped = (
                    new_values - flat_returns[mb_idx]
                ).pow(2)
                value_loss_clipped = (
                    value_pred_clipped - flat_returns[mb_idx]
                ).pow(2)

                value_loss = 0.5 * torch.max(
                    value_loss_unclipped,
                    value_loss_clipped
                ).mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
