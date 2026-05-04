import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv


class TrafficGNN(nn.Module):

    def __init__(
        self,
        obs_dim,
        hidden_dim,
        num_actions
    ):
        super().__init__()

        self.gat1 = GATv2Conv(
            in_channels=obs_dim,
            out_channels=hidden_dim,
            heads=4,
            edge_dim=1
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * 4,
            out_channels=hidden_dim,
            heads=1,
            edge_dim=1
        )

        self.policy_head = nn.Linear(
            hidden_dim,
            num_actions
        )

        self.value_head = nn.Linear(
            hidden_dim,
            1
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        action_mask=None
    ):
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = F.relu(x)

        logits = self.policy_head(x)

        if action_mask is not None:
            logits = logits.masked_fill(
                action_mask == 0,
                -1e9
            )

        values = self.value_head(x)

        return logits, values.squeeze(-1)