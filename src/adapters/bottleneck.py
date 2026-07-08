import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Houlsby 2019 bottleneck adapter: down -> ReLU -> up, with residual.

    Up-projection zero-init so the adapter starts as identity.

    Post-pool adapter: it operates on the frozen backbone's pooled feature
    vector, so the backbone stays frozen (backbone_trainable = False) and is
    run under no_grad in BPEFTModel.adapter_features.
    """

    #: the backbone is frozen; only this post-pool module trains.
    backbone_trainable = False

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))
