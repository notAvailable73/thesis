import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """
    Bottleneck adapter: Down-project to rank r, ReLU, Up-project back to dim.
    Adds a residual connection: output = x + up(act(down(x))).

    Parameter count for dim=512, r=16:
      down: 512*16 + 16 = 8,208
      up:   16*512 + 512 = 8,704
      total = 16,912

    Up-projection initialized to zero so adapter starts as identity (Houlsby 2019).
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


if __name__ == "__main__":
    adapter = BottleneckAdapter(512, 16)
    x = torch.randn(4, 512)
    out = adapter(x)
    assert out.shape == (4, 512), f"Bad shape: {out.shape}"
    assert torch.allclose(out, x, atol=1e-6), "Adapter should be identity at init!"
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"Output shape: {out.shape}")
    print(f"Identity at init: True")
    print(f"Param count: {n_params:,}  (expect 16,912)")
    print("OK")
