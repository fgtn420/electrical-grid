import torch
import torch.nn as nn


class GridMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.Softplus(),
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # scalar logit per sample


def compute_triple_norm(model: GridMLP) -> list[float]:
    """Per Linear layer: max_j sum_i |w_ij|  (induced L1 operator norm)."""
    norms = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # sum over rows (i), max over columns (j)
            norms.append(module.weight.abs().sum(dim=0).max().item())
    return norms
