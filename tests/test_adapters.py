import torch
from src.adapters import BottleneckAdapter, LoRAAdapter, build_adapter


def test_bottleneck_is_identity_at_init():
    adapter = BottleneckAdapter(dim=512, rank=16)
    x = torch.randn(4, 512)
    out = adapter(x)
    assert out.shape == (4, 512)
    assert torch.allclose(out, x, atol=1e-6)


def test_build_adapter_factory_bottleneck():
    a1 = build_adapter({"type": "bottleneck", "rank": 8}, dim=64)
    assert isinstance(a1, BottleneckAdapter)
    # Post-pool adapter -> backbone stays frozen.
    assert a1.backbone_trainable is False


def test_build_adapter_lora_requires_backbone():
    # Step 5: LoRA is no longer a post-pool stub; it must inject into a
    # backbone, so the factory rejects a call without one.
    import pytest
    with pytest.raises(ValueError):
        build_adapter({"type": "lora", "rank": 4, "alpha": 4}, dim=64)


def test_bottleneck_param_count():
    a = BottleneckAdapter(dim=512, rank=16)
    n = sum(p.numel() for p in a.parameters())
    # 512*16 + 16 + 16*512 + 512 = 16912
    assert n == 16912
