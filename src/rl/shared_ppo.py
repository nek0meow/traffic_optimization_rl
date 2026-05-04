import torch
from torch import nn

class SharedTLSPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        hidden_dim = 512
        self.input_norm = nn.LayerNorm(obs_dim)
        self.input = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
        )
        self.block_1 = ResidualBlock(hidden_dim)
        self.block_2 = ResidualBlock(hidden_dim)
        self.actor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(self._init_weights)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input(self.input_norm(obs))
        hidden = self.block_1(hidden)
        hidden = self.block_2(hidden)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
