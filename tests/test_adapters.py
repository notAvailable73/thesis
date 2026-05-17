import torch
from src.adapters import BottleneckAdapter, LoRAAdapter, build_adapter


def test_bottleneck_is_identity_at_init():
    adapter = BottleneckAdapter(dim=512, rank=16)
    x = torch.randn(4, 512)
    out = adapter(x)
    assert out.shape == (4, 512)
    assert torch.allclose(out, x, atol=1e-6)


def test_lora_is_identity_at_init():
    adapter = LoRAAdapter(dim=512, rank=4, alpha=8)
    x = torch.randn(4, 512)
    out = adapter(x)
    assert out.shape == (4, 512)
    assert torch.allclose(out, x, atol=1e-6)


def test_build_adapter_factory():
    a1 = build_adapter({"type": "bottleneck", "rank": 8}, dim=64)
    a2 = build_adapter({"type": "lora", "rank": 4, "alpha": 4}, dim=64)
    assert isinstance(a1, BottleneckAdapter)
    assert isinstance(a2, LoRAAdapter)


def test_bottleneck_param_count():
    a = BottleneckAdapter(dim=512, rank=16)
    n = sum(p.numel() for p in a.parameters())
    # 512*16 + 16 + 16*512 + 512 = 16912
    assert n == 16912
