import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int = 32, cutoff: float = 6.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class CrystalGNN(nn.Module):
    def __init__(self, hidden: int = 128, n_layers: int = 3, n_rbf: int = 32, cutoff: float = 6.0):
        super().__init__()
        self.embed = nn.Embedding(100, hidden)  # atomic number -> vector
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.convs = nn.ModuleList(
            [CGConv(hidden, dim=n_rbf, batch_norm=True) for _ in range(n_layers)]
        )
        self.energy_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.stab_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, batch) -> dict:
        x = self.embed(batch.z)
        edge_attr = self.rbf(batch.edge_dist)
        for conv in self.convs:
            x = conv(x, batch.edge_index, edge_attr)
        pooled = global_mean_pool(x, batch.batch)
        return {
            "energy": self.energy_head(pooled).squeeze(-1),
            "stab_logit": self.stab_head(pooled).squeeze(-1),
        }
