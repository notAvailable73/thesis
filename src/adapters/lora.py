import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """Hu 2021 LoRA-style adapter: x + (alpha/rank) * up(down(x)), no bias, no non-linearity.

    Kaiming-uniform init on down, zero init on up so the adapter starts as identity.
    """
    def __init__(self, dim: int, rank: int, alpha: float | None = None):
        super().__init__()
        self.rank = rank
        self.alpha = float(alpha) if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scaling * self.up(self.down(x))
