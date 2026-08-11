import importlib.util

import pytest


torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(torch_spec is None, reason="torch not installed locally")


def test_vgpu_trainable_only_checkpoint_helpers_roundtrip():
    import torch
    from src.vgpu.runtime import _vgpu_cpu_state, _vgpu_load_partial

    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    state = _vgpu_cpu_state(model)
    assert set(state) == {"1.weight", "1.bias"}
    with torch.no_grad():
        model[1].weight.add_(5)
    _vgpu_load_partial(model, state)
    assert torch.equal(model[1].weight.cpu(), state["1.weight"])


def test_vgpu_partial_checkpoint_rejects_unknown_key():
    import torch
    from src.vgpu.runtime import _vgpu_load_partial

    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="unknown model keys"):
        _vgpu_load_partial(model, {"missing": torch.tensor(1)})
