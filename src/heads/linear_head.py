import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Plain linear classifier producing raw logits."""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class EvidentialHead(nn.Module):
    """Linear classifier followed by an evidence activation that yields non-negative outputs.

    activation:
      - softplus:  log(1 + exp(z))                — smooth, bounded growth
      - exp_clamp: exp(clamp(z, max=10))          — sharp, multiplicative growth
      - relu:      max(0, z)                      — original Sensoy 2018 activation
    """
    SUPPORTED = ("softplus", "exp_clamp", "relu")

    def __init__(self, in_dim: int, num_classes: int, activation: str = "softplus"):
        super().__init__()
        if activation not in self.SUPPORTED:
            raise ValueError(f"activation must be one of {self.SUPPORTED}, got {activation!r}")
        self.activation = activation
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc(x)
        if self.activation == "softplus":
            return F.softplus(z)
        if self.activation == "exp_clamp":
            return torch.exp(torch.clamp(z, max=10.0))
        return F.relu(z)
