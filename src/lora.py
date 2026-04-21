import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """
    LoRA-style adapter (Hu et al. 2021, "LoRA: Low-Rank Adaptation of LLMs").

    output = x + (alpha / rank) * up(down(x))

    Differences from BottleneckAdapter:
      - No non-linearity between down and up
      - No bias terms
      - Kaiming-uniform init on `down`, zero init on `up` (identity at start)
      - Scaling factor (alpha / rank) controls update magnitude

    Parameter count for dim=512, r=16:
      down: 512 * 16 = 8,192
      up:   16 * 512 = 8,192
      total = 16,384  (slightly fewer than bottleneck's 16,912)
    """
    def __init__(self, dim: int, rank: int, alpha: float = None):
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


if __name__ == "__main__":
    adapter = LoRAAdapter(512, 16)
    x = torch.randn(4, 512)
    out = adapter(x)
    assert out.shape == (4, 512), f"Bad shape: {out.shape}"
    assert torch.allclose(out, x, atol=1e-6), "LoRA should be identity at init!"
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"Output shape: {out.shape}")
    print(f"Identity at init: True")
    print(f"Param count: {n_params:,}  (expect 16,384)")
    print("OK")
