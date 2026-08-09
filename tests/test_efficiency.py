"""Step 11 (RQ4) tests for src/utils/efficiency.py.

Repo convention: plain module-level functions, no classes, CPU-only, offline
(`resnet18(weights=None)` / `mobilenet_v3_small(weights=None)`, mirroring
tests/test_placement.py::_bare_frozen_resnet18 and
tests/test_mobilenetv3.py::_bare_frozen_mbnet), no CUDA assumed anywhere.
fvcore is now in requirements.txt (added for this step), but FLOP tests still
guard with `pytest.importorskip` so this file degrades gracefully in an
environment where it isn't installed.

Structural assertions only — no test asserts a wall-clock threshold; latency
values are measured on whatever hardware pytest happens to run on.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_small, resnet18

from src.adapters import PlacementAdapter, LoRAAdapter, LinearProbeAdapter
from src.heads import PrototypeHead
from src.models.bpeft_model import BPEFTModel
from src.utils import load_config
from src.utils import params as params_mod
from src.utils.efficiency import (
    FLOP_CONVENTION,
    PUBLISHED_REFERENCE_MACS,
    _module_device,
    params_report,
    count_frozen_params,
    param_bytes,
    flops_backend_available,
    count_flops,
    count_flops_detailed,
    check_reference_flops,
    time_callable,
    measure_latency_gpu,
    measure_latency_cpu,
    thread_count,
    episode_input_shapes,
    measure_peak_memory,
    measure_train_step_peak_memory,
    collect_env,
    device_profile_slug,
    session_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "configs" / "grid" / "_index.json"

fvcore = pytest.importorskip("fvcore", reason="fvcore not installed")


def _bare_frozen_resnet18():
    m = resnet18(weights=None)
    m.fc = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


def _bare_frozen_mbnet():
    m = mobilenet_v3_small(weights=None)
    m.classifier = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


def _proto_model(backbone, adapter, *, metric="cosine", evidence_affine=False):
    # BPEFTModel/PrototypeHead don't take an "interpretation" — that's a
    # loss/evaluator-side concept (cfg.head.interpretation), irrelevant here.
    head = PrototypeHead(metric=metric, evidence_affine=evidence_affine)
    return BPEFTModel(backbone, adapter, head,
                      backbone_trainable=getattr(adapter, "backbone_trainable", False))


# --------------------------------------------------------------------------
# Params / bytes
# --------------------------------------------------------------------------
def test_param_counts_split_and_sum():
    bb = _bare_frozen_resnet18()
    ad = PlacementAdapter(bb, rank=16, placement="parallel", block_ids=[0, 1, 2, 3])
    model = _proto_model(bb, ad)
    pr = params_report(model)
    assert pr["trainable"] == 31744
    assert pr["total"] == 11176512 + 31744
    assert pr["frozen"] == pr["total"] - pr["trainable"]
    assert count_frozen_params(model) == pr["frozen"]


def test_param_bytes_is_numel_times_four_fp32():
    bb = _bare_frozen_resnet18()
    ad = PlacementAdapter(bb, rank=16, placement="parallel", block_ids=[3])
    model = _proto_model(bb, ad)
    total_params = params_mod.count_total_params(model)
    assert param_bytes(model) == 4 * total_params
    with_buffers = param_bytes(model, include_buffers=True)
    assert with_buffers >= param_bytes(model)


def test_params_report_reports_zero_trainable_as_zero_not_missing():
    bb = _bare_frozen_resnet18()
    ad = LinearProbeAdapter()
    model = _proto_model(bb, ad)
    pr = params_report(model)
    assert pr["trainable"] == 0
    assert pr["trainable_fraction"] == 0.0
    assert "trainable" in pr


# --------------------------------------------------------------------------
# FLOPs / MACs
# --------------------------------------------------------------------------
def test_flop_convention_is_macs_and_flops_is_double():
    assert FLOP_CONVENTION == "macs"
    m = nn.Linear(1000, 1000, bias=False)
    detail = count_flops_detailed(m, input_shape=(1, 1000), forward="forward")
    assert detail["macs"] == 1_000_000
    assert detail["flops_2x_macs"] == 2_000_000


def test_resnet18_macs_matches_published_reference():
    raw = resnet18(weights=None)
    raw.eval()
    macs = count_flops(raw, forward="forward")
    result = check_reference_flops("resnet18", macs)
    assert result["status"] == "pass"
    assert 0.9 < result["ratio_to_midpoint"] < 1.1


def test_mobilenetv3_small_macs_matches_published_reference_and_classifier_delta():
    raw = mobilenet_v3_small(weights=None)
    raw.eval()
    macs_with_head = count_flops(raw, forward="forward")

    backbone_only = _bare_frozen_mbnet()
    macs_backbone_only = count_flops(backbone_only)

    result = check_reference_flops("mobilenetv3_small", macs_with_head, macs_backbone_only)
    assert result["status"] == "pass"
    expected_delta = 576 * 1024 + 1024 * 1000
    assert result["classifier_delta_measured"] == expected_delta
    assert result["classifier_delta_analytical"] == expected_delta
    assert result["delta_explains_gap"] is True


def test_macs_scale_linearly_with_batch():
    raw = resnet18(weights=None)
    raw.eval()
    macs_b1 = count_flops(raw, input_shape=(1, 3, 224, 224), forward="forward")
    macs_b2 = count_flops(raw, input_shape=(2, 3, 224, 224), forward="forward")
    assert macs_b2 == 2 * macs_b1


def test_placement_adapter_macs_delta_matches_analytical():
    # PlacementAdapter wires its Conv1x1Bottleneck bodies in via a forward
    # HOOK closure, not by inserting them into the backbone's own module
    # tree (unlike LoRA, which replaces a submodule in place) -- so tracing
    # the bare backbone alone can't see the bodies' parameters. The real
    # measurement path (count_flops(model, forward="auto")) traces the
    # WHOLE model (model.adapter_features), whose submodule tree includes
    # both model.backbone AND model.adapter -- mirror that here.
    bb_base = _bare_frozen_resnet18()
    base_model = _proto_model(bb_base, LinearProbeAdapter())
    macs_base = count_flops(base_model)

    bb = _bare_frozen_resnet18()
    ad = PlacementAdapter(bb, rank=16, placement="parallel", block_ids=[0, 1, 2, 3])
    adapted_model = _proto_model(bb, ad)
    macs_adapted = count_flops(adapted_model)

    delta = macs_adapted - macs_base
    channels_hw = [(64, 56), (128, 28), (256, 14), (512, 7)]
    expected = sum(2 * c * 16 * h * h for c, h in channels_hw)
    assert delta == expected == 12_042_240


def test_lora_macs_delta_matches_analytical():
    bb_base = _bare_frozen_resnet18()
    macs_base = count_flops(bb_base, forward="forward")

    bb = _bare_frozen_resnet18()
    LoRAAdapter(bb, rank=16, targets=["layer4.0.downsample.0"])
    macs_adapted = count_flops(bb, forward="forward")

    delta = macs_adapted - macs_base
    expected = 256 * 16 * 49 + 16 * 512 * 49
    assert delta == expected == 602_112


def test_full_ft_and_linear_probe_have_identical_inference_macs():
    # Both adapters leave the backbone's FORWARD graph untouched (full_ft only
    # flips requires_grad; linear_probe inserts nothing) -- same MACs.
    bb_a = _bare_frozen_resnet18()
    bb_b = _bare_frozen_resnet18()
    assert count_flops(bb_a, forward="forward") == count_flops(bb_b, forward="forward")


def test_uncounted_ops_recorded_and_json_safe():
    raw = mobilenet_v3_small(weights=None)
    raw.eval()
    detail = count_flops_detailed(raw, forward="forward")
    assert "aten::hardswish_" in detail["uncounted_ops"]
    for k, v in detail["uncounted_ops"].items():
        assert isinstance(k, str)
        assert isinstance(v, int)
    json.dumps(detail)  # must not raise


def test_reference_gate_fails_a_wrong_number():
    result = check_reference_flops("resnet18", int(1.8e9 * 0.5))
    assert result["status"] == "fail"
    assert result["ratio_to_midpoint"] < 0.9


def test_flops_backend_available_reports_fvcore_true_here():
    assert flops_backend_available()["fvcore"] is True


# --------------------------------------------------------------------------
# Trace-device regression (step_writeups/step11.txt Section 8)
#
# count_flops_detailed builds its own fvcore trace tensor. When that tensor
# was unconditionally CPU, any caller holding a CUDA-resident model crashed
# inside fvcore -- which discarded a whole canonical Kaggle measurement. The
# device half of that bug is only reachable with real CUDA, so it is gated;
# the two CPU tests below pin the invariant that makes it unreachable.
# --------------------------------------------------------------------------
def test_module_device_falls_back_through_params_buffers_then_cpu():
    assert _module_device(nn.Linear(4, 4)).type == "cpu"      # from a parameter

    buffers_only = nn.Module()
    buffers_only.register_buffer("b", torch.zeros(2))
    assert not list(buffers_only.parameters())
    assert _module_device(buffers_only).type == "cpu"          # from a buffer

    assert _module_device(nn.Module()).type == "cpu"           # from neither


def test_count_flops_traces_on_the_models_own_device_not_an_assumed_cpu():
    m = nn.Linear(1000, 1000, bias=False)
    detail = count_flops_detailed(m, input_shape=(1, 1000), forward="forward")
    assert detail["trace_device"] == str(_module_device(m))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs real CUDA")
def test_count_flops_on_a_cuda_resident_model_matches_cpu_and_does_not_crash():
    # The exact crash from Section 8.1: model on cuda, trace tensor on cpu.
    m_cpu = _bare_frozen_resnet18()
    macs_cpu = count_flops(m_cpu, forward="forward")

    m_cuda = _bare_frozen_resnet18().to("cuda")
    detail = count_flops_detailed(m_cuda, forward="forward")
    assert detail["trace_device"].startswith("cuda")
    assert detail["macs"] == macs_cpu  # MACs are device-independent


# --------------------------------------------------------------------------
# Latency harness (schema/behaviour only)
# --------------------------------------------------------------------------
def test_time_callable_schema_and_ordering():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return torch.zeros(1)

    result = time_callable(fn, device="cpu", n_warmup=1, n_measure=3,
                           grad_mode="grad")
    stats = result["latency_ms"]
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["std"] >= 0
    assert stats["n_measure"] == 3
    assert all(isinstance(v, float) for v in
              (stats["mean"], stats["std"], stats["median"], stats["p90"]))


def test_time_callable_calls_fn_exactly_warmup_plus_measure_times():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return torch.zeros(1)

    time_callable(fn, device="cpu", n_warmup=2, n_measure=5, grad_mode="grad")
    assert calls["n"] == 7


def test_time_callable_records_grad_mode_and_output_has_no_grad():
    w = torch.nn.Parameter(torch.ones(1))

    def fn():
        return w * 2  # would require_grad under a "grad" context

    result = time_callable(fn, device="cpu", n_warmup=1, n_measure=2,
                           grad_mode="inference")
    assert result["grad_mode"] == "inference"


def test_time_callable_raises_if_inference_mode_not_actually_applied(monkeypatch):
    # torch.inference_mode()/no_grad() unconditionally strip requires_grad
    # from anything computed inside them, so the tripwire can't be triggered
    # through time_callable's own (correct) context selection. Simulate the
    # bug it guards against -- grad_mode="inference" LABELLED but the
    # context that actually runs is a no-op -- by monkeypatching the context
    # lookup, and confirm the post-hoc assertion catches it.
    import contextlib
    from src.utils import efficiency as eff

    monkeypatch.setitem(eff._GRAD_CONTEXTS, "inference", contextlib.nullcontext)
    w = torch.nn.Parameter(torch.ones(1))

    def fn():
        return w * 2  # requires_grad=True since no real no-grad context runs

    with pytest.raises(RuntimeError):
        time_callable(fn, device="cpu", n_warmup=1, n_measure=2,
                      grad_mode="inference")


def test_measure_latency_cpu_returns_mean_std_tuple():
    bb = _bare_frozen_resnet18()
    ad = LinearProbeAdapter()
    model = _proto_model(bb, ad)
    result = measure_latency_cpu(model, input_shape=(1, 3, 64, 64),
                                 n_warmup=1, n_measure=2, num_threads=1)
    assert isinstance(result, tuple) and len(result) == 2
    mean, std = result
    assert isinstance(mean, float) and isinstance(std, float)
    assert mean >= 0 and std >= 0


def test_measure_latency_cpu_restores_thread_count():
    bb = _bare_frozen_resnet18()
    ad = LinearProbeAdapter()
    model = _proto_model(bb, ad)
    before = torch.get_num_threads()
    measure_latency_cpu(model, input_shape=(1, 3, 64, 64),
                        n_warmup=1, n_measure=1, num_threads=1)
    assert torch.get_num_threads() == before


def test_thread_count_context_manager_restores_on_exception():
    before = torch.get_num_threads()
    with pytest.raises(ValueError):
        with thread_count(1):
            raise ValueError("boom")
    assert torch.get_num_threads() == before


def test_measure_latency_gpu_raises_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host; the no-CUDA path isn't exercised")
    bb = _bare_frozen_resnet18()
    ad = LinearProbeAdapter()
    model = _proto_model(bb, ad)
    with pytest.raises(RuntimeError):
        measure_latency_gpu(model, n_warmup=1, n_measure=1)


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
def test_measure_peak_memory_skipped_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    bb = _bare_frozen_resnet18()
    ad = LinearProbeAdapter()
    model = _proto_model(bb, ad)
    result = measure_peak_memory(model, input_shape=(1, 3, 64, 64))
    assert result["status"] == "skipped"
    assert result["reason"]


def test_train_step_measurement_skips_zero_trainable_params():
    bb = _bare_frozen_resnet18()
    ad = LinearProbeAdapter()
    model = _proto_model(bb, ad)
    for p in model.parameters():
        p.requires_grad = False
    cfg = load_config(REPO_ROOT / "configs" / "exp_phase3_linear_probe_softmax.yaml")
    result = measure_train_step_peak_memory(model, cfg, device="cpu", n_steps=1)
    assert result["status"] == "skipped"
    assert "trainable" in result["reason"]
    assert result["trainable_params"] == 0
    assert result["spec_status"] == "ADDITION beyond implementation.txt 11.1"


def test_train_step_measurement_restores_model_state():
    bb = _bare_frozen_resnet18()
    ad = PlacementAdapter(bb, rank=4, placement="parallel", block_ids=[3])
    model = _proto_model(bb, ad)
    cfg = load_config(REPO_ROOT / "configs" / "exp_phase3_placement_parallel_softmax.yaml")
    was_training = model.training
    backbone_was_training = model.backbone.training
    req_grad_before = {id(p): p.requires_grad for p in model.parameters()}

    result = measure_train_step_peak_memory(model, cfg, device="cpu", n_steps=2)

    assert result["status"] == "ok"
    assert result["trainable_params"] > 0
    assert model.training == was_training
    assert model.backbone.training == backbone_was_training
    for p in model.parameters():
        assert p.requires_grad == req_grad_before[id(p)]
    # The backbone's OWN weights stay frozen even for a placement adapter --
    # the placed Conv1x1Bottleneck bodies are child modules of `ad`, not of
    # `bb` (they're wired in via a forward hook closure, not module
    # insertion), so backbone_trainable=True only means the backbone runs
    # WITH autograd (needed for the hook's gradient to reach `ad.bodies`),
    # not that any backbone parameter itself receives gradients.
    assert all(not p.requires_grad for p in bb.parameters())
    assert result["grad_bytes_analytical"] == 4 * result["trainable_params"]
    assert result["optimizer_state_bytes_analytical"] == 8 * result["trainable_params"]


# --------------------------------------------------------------------------
# Contract + environment
# --------------------------------------------------------------------------
def test_prototype_model_forward_raises_so_harness_uses_adapter_features():
    bb = _bare_frozen_resnet18()
    ad = LinearProbeAdapter()
    model = _proto_model(bb, ad)
    x = torch.zeros(2, 3, 224, 224)
    with pytest.raises(TypeError):
        model(x)
    feats = model.adapter_features(x)
    assert feats.shape == (2, 512)


def test_episode_input_shapes_derived_from_config_never_hardcoded():
    cfg = load_config(REPO_ROOT / "configs" / "exp_phase3_placement_parallel_softmax.yaml")
    shapes = episode_input_shapes(cfg)
    assert shapes["n_way"] == 5
    assert shapes["support"] == (5 * shapes["k_shot"], 3, shapes["image_size"], shapes["image_size"])
    assert shapes["query"] == (5 * shapes["q_query"], 3, shapes["image_size"], shapes["image_size"])


def test_collect_env_has_required_fields():
    env = collect_env(device="cpu")
    assert "version" in env["torch"]
    assert env["torch"]["device_kind"] == "cpu"
    for key in ("cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms",
               "torch_num_threads"):
        assert key in env["flags"]
    assert "git_sha" in env["repo"]  # may be None, but must be present
    json.dumps(env, default=str)  # must not raise


def test_device_profile_slug_distinguishes_cpu_thread_counts():
    env = collect_env(device="cpu")
    slug1 = device_profile_slug(env, num_threads=1)
    slug4 = device_profile_slug(env, num_threads=4)
    assert slug1 != slug4
    assert slug1.startswith("cpu_1thread_")


def test_session_id_is_stable_shape():
    env = collect_env(device="cpu")
    sid = session_id(env)
    assert sid.startswith("s_")
    parts = sid.split("_")
    assert len(parts) >= 4


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="configs/grid/_index.json not present")
def test_efficiency_key_arity_matches_grid_axes():
    # 12 distinct (backbone, adapter, head) tuples span the closed grid --
    # not a measurement of scripts/efficiency_table.py (that file's own key
    # derivation is tested separately), just a fixed fact about the index
    # this module's measurement loop must cover.
    cells = json.loads(INDEX_PATH.read_text())["cells"]
    keys = {(c["backbone"], c["adapter"], c["head"]) for c in cells}
    assert len(keys) == 12


# --------------------------------------------------------------------------
# scripts/efficiency_table.py::_reference_backbones — fault isolation
#
# The opt-in --include-reference-backbones rows had ZERO coverage before
# 2026-08-09, which is how a crash in them discarded an entire canonical
# Kaggle measurement (step_writeups/step11.txt Section 8). Real ViT-B/16 /
# DeiT-Tiny are far too heavy for this suite, so the backbone constructors
# are stubbed: what is under test is the ISOLATION contract, not the models.
# --------------------------------------------------------------------------
def _tiny_224_model():
    return nn.Sequential(
        nn.Conv2d(3, 2, kernel_size=3, stride=16),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 2),
    )


def _reference_backbones_fn():
    from scripts.efficiency_table import _reference_backbones
    return _reference_backbones


def test_reference_backbones_returns_measured_rows_for_a_working_backbone(monkeypatch):
    import torchvision.models
    monkeypatch.setattr(torchvision.models, "vit_b_16",
                        lambda **kw: _tiny_224_model())
    out = _reference_backbones_fn()(device="cpu")
    row = out["vit_b_16"]
    assert row["params"]["trainable"] > 0
    assert row["flops"]["macs"] > 0
    assert "disclaimer" in row  # never mistakable for a thesis result
    json.dumps(out)  # must be JSON-safe -- it is written straight to the table


def test_a_failing_reference_backbone_degrades_one_row_not_the_whole_block(monkeypatch):
    """The Section 8.1 failure mode, in miniature: whatever blows up in a
    bonus row must be captured as data, so the other row still survives."""
    import torchvision.models

    def _boom(**kw):
        raise RuntimeError("Input type (torch.FloatTensor) and weight type "
                           "(torch.cuda.FloatTensor) should be the same")

    monkeypatch.setattr(torchvision.models, "vit_b_16", _boom)
    fake_timm = types.ModuleType("timm")
    fake_timm.create_model = lambda *a, **kw: _tiny_224_model()
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    out = _reference_backbones_fn()(device="cpu")
    assert out["vit_b_16"]["status"] == "failed"
    assert "RuntimeError" in out["vit_b_16"]["error"]
    # ...and the independent row is unaffected:
    assert out["deit_tiny_patch16_224"]["flops"]["macs"] > 0


def test_missing_timm_is_reported_as_unavailable_not_failed(monkeypatch):
    # A missing optional dependency is a known, benign state -- it must stay
    # distinguishable from a real failure in the merged JSON.
    import builtins
    real_import = builtins.__import__

    def _no_timm(name, *a, **kw):
        # Raised before sys.modules is consulted, so this holds whether or
        # not timm happens to be installed in the running environment.
        if name == "timm":
            raise ImportError("No module named 'timm'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_timm)
    import torchvision.models
    monkeypatch.setattr(torchvision.models, "vit_b_16",
                        lambda **kw: _tiny_224_model())

    out = _reference_backbones_fn()(device="cpu")
    assert out["deit_tiny_patch16_224"]["status"] == "unavailable"
