import torch
from torch.distributions import Categorical

def masked_dist(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
    masked_logits = logits.masked_fill(action_mask <= 0, -1e9)
    return Categorical(logits=masked_logits)


def tensor(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)

    for step in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae
        next_value = values[step]

    returns = advantages + values
    return returns, advantages


def compute_gae_with_next_values(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_values[-1])

    for step in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[step]
        delta = (
            rewards[step]
            + gamma * next_values[step] * next_non_terminal
            - values[step]
        )
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae

    returns = advantages + values
    return returns, advantages
