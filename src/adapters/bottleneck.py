import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Houlsby 2019 bottleneck adapter: down -> ReLU -> up, with residual.

    Up-projection zero-init so the adapter starts as identity.
    """
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))
