"""Efficiency instrumentation for Step 11 (RQ4).

TWO CLASSES OF NUMBER, kept apart everywhere in this module and in
results/efficiency_table.json:

  DETERMINISTIC  params, FLOPs/MACs. Device-independent and byte-identically
                 reproducible, in the same spirit as the rest of this repo's
                 reproducibility invariant (CLAUDE.md / src/utils/seed.py) --
                 scripts/efficiency_table.py recomputes and diffs this block
                 across sessions.
  MEASURED       latency, CUDA peak memory. Depend on host load, GPU
                 clocks/thermals, driver, cuDNN kernel selection and
                 allocator history. EXEMPT from the byte-identical invariant
                 by construction; every recorded value names the session and
                 hardware profile that produced it.

FLOP CONVENTION. This module reports MACs (multiply-accumulate operations),
matching the "MAdds" convention of Howard 2019 (PAPER SUMMARIES/
CNN_paper_summaries.txt Section 6: "V3-Small 1.0 = ~56-57 MAdds, 2.5M params")
and the "1.8 GFLOPs" figure quoted for ResNet-18 in
src/backbones/mobilenetv3.py's docstring -- both are MAC counts despite the
naming. fvcore.nn.FlopCountAnalysis already returns MACs directly (verified:
nn.Linear(1000, 1000, bias=False) at B=1 -> 1_000_000, matching in*out
exactly, NOT 2x); `flops_2x_macs` is also emitted for readers who use the
"FLOPs = 2*MACs" convention, so a table can never be off by a factor of 2.

WHAT MAC COUNTERS DO NOT SEE. Only multiply-accumulate ops are counted
(convolution, mm/addmm/bmm). BatchNorm, ReLU, h-swish, hard-sigmoid,
global mean-pool and residual adds are NOT. This systematically understates
depthwise / h-swish backbones -- report a MAC ratio next to its matching
latency ratio, never alone (see check_reference_flops / the module tests for
the measured ResNet-18:MobileNetV3-Small MAC-vs-latency gap).

CONTRACT WITH src.models.bpeft_model.BPEFTModel. The MVT grid uses
head.type=prototype, so `model(x)` RAISES TypeError (PrototypeHead.forward
takes (support_features, support_labels, query_features), not a bare image
batch). Every function here that takes a `model` therefore resolves its
per-image forward path to `model.adapter_features(x)` by default
(`forward="auto"`) rather than `model(x)` -- see `_resolve_forward`.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from .params import count_total_params, count_trainable_params

__all__ = [
    "FLOP_CONVENTION",
    "DEFAULT_INPUT_SHAPE",
    "PUBLISHED_REFERENCE_MACS",
    "params_report",
    "count_frozen_params",
    "param_bytes",
    "flops_backend_available",
    "count_flops",
    "count_flops_detailed",
    "check_reference_flops",
    "time_callable",
    "measure_latency_gpu",
    "measure_latency_cpu",
    "thread_count",
    "episode_input_shapes",
    "measure_peak_memory",
    "measure_train_step_peak_memory",
    "collect_env",
    "device_profile_slug",
    "session_id",
    "gpu_clock_snapshot",
]

#: fvcore.nn.FlopCountAnalysis.total() already returns MACs (verified against
#: nn.Linear and nn.Conv2d in tests/test_efficiency.py), not 2*MACs.
FLOP_CONVENTION = "macs"

#: (B, C, H, W) — matches configs/base.yaml's dataset.image_size: 224.
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)

#: Published reference figures, used by check_reference_flops(). NOTHING here
#: is general knowledge -- each entry names the summary file/line it came
#: from, per thesis_implementation_instructions.txt Section 6.
PUBLISHED_REFERENCE_MACS: dict = {
    "resnet18": {
        "macs": 1.8e9,
        "tolerance_frac": 0.05,
        "variant": "torchvision resnet18, 224x224 input, WITH the fc layer",
        "source": (
            "src/backbones/mobilenetv3.py docstring, citing PAPER SUMMARIES/"
            "CNN_paper_summaries.txt Section 8: 'ResNet-18 is 11.7M params / "
            "1.8 GFLOPs'"
        ),
        "convention_note": "published as 'GFLOPs' but is a MAC count",
    },
    "mobilenetv3_small": {
        "macs_range": (56e6, 57e6),
        "tolerance_frac": 0.05,
        "variant": "torchvision mobilenet_v3_small, 224x224 input, WITH the classifier",
        "source": (
            "PAPER SUMMARIES/CNN_paper_summaries.txt Section 6 (Howard 2019): "
            "'V3-Small 1.0 = ~67.4-67.5% top-1, ~56-57 MAdds, 2.5M params'"
        ),
        "classifier_macs_analytical": 576 * 1024 + 1024 * 1000,
        "note": (
            "our backbone sets classifier=Identity (576-d pooled feature, "
            "implementation.txt Section 8.1), so the comparable published "
            "figure is the WITH-classifier measurement; the classifier-free "
            "backbone must read ~1.61M MACs lower"
        ),
    },
}


# --------------------------------------------------------------------------
# Forward-path resolution (the PrototypeHead contract)
# --------------------------------------------------------------------------
def _resolve_forward(model: nn.Module, forward: str | Callable) -> Callable:
    """Resolve `forward` to a callable(x) -> tensor, for TIMING/MEMORY
    measurement (no JIT tracing involved, so a bare bound method is fine).

    "auto" prefers `model.adapter_features` (the backbone->adapter path used
    by every grid config, since head.type=prototype makes `model(x)` raise)
    and falls back to `model` itself for models that don't have it (e.g. a
    bare torchvision backbone passed in directly for a reference row).
    "forward" explicitly requests the model's own `__call__` — the path for
    bare reference backbones (e.g. a raw torchvision model) that have no
    `adapter_features`/`forward_proto`.
    """
    if forward == "auto":
        return model.adapter_features if hasattr(model, "adapter_features") else model
    if forward == "adapter_features":
        return model.adapter_features
    if forward == "forward_proto":
        return model.forward_proto
    if forward == "forward":
        return model
    if callable(forward):
        return forward
    raise ValueError(f"Unknown forward spec: {forward!r}")


class _TracedForward(nn.Module):
    """Registers `model` as a genuine CHILD MODULE (via `self.inner = model`)
    so tracing-based FLOP counters (fvcore, which traces via
    torch.jit.trace) see its parameters as part of the traced graph.

    Storing only a BOUND METHOD (e.g. `model.adapter_features`, a plain
    python attribute on a closure) does NOT register `model` as a submodule
    of the traced wrapper — torch.jit.trace then treats its weights as
    untracked free variables and crashes with "Cannot insert a Tensor that
    requires grad as a constant" the moment it hits the first conv whose
    weight requires_grad. Verified empirically against
    torchvision.models.mobilenet_v3_small (fails via a closure wrapper,
    passes once the model is a registered submodule).
    """

    def __init__(self, model: nn.Module, method_name: str):
        super().__init__()
        self.inner = model
        self._method_name = method_name

    def forward(self, x):
        return getattr(self.inner, self._method_name)(x)


class _CallableModule(nn.Module):
    """Best-effort nn.Module wrapper for an arbitrary callable passed to
    `forward=`. Fine for timing (no tracing); FLOP counting through this
    path only traces correctly if the callable's closed-over model is
    ALREADY a registered submodule somewhere (e.g. a real nn.Module method
    bound elsewhere) — prefer `forward="auto"` / a named method wherever
    possible so `_TracedForward` above is used instead.
    """

    def __init__(self, fn: Callable):
        super().__init__()
        self._fn = fn

    def forward(self, x):
        return self._fn(x)


def _resolve_forward_module(model: nn.Module, forward: str | Callable) -> nn.Module:
    """Like `_resolve_forward`, but returns an nn.Module suitable for
    tracing-based FLOP counting (see `_TracedForward`)."""
    if isinstance(model, nn.Module) and forward in (
            "auto", "adapter_features", "forward_proto", "forward"):
        if forward == "auto":
            method = "adapter_features" if hasattr(model, "adapter_features") else "forward"
        else:
            method = forward
        return _TracedForward(model, method)
    return _CallableModule(_resolve_forward(model, forward))


# --------------------------------------------------------------------------
# Parameters (deterministic)
# --------------------------------------------------------------------------
def count_frozen_params(model: nn.Module) -> int:
    return count_total_params(model) - count_trainable_params(model)


def param_bytes(model: nn.Module, *, trainable_only: bool = False,
                include_buffers: bool = False) -> int:
    total = sum(p.numel() * p.element_size() for p in model.parameters()
               if (p.requires_grad or not trainable_only))
    if include_buffers:
        total += sum(b.numel() * b.element_size() for b in model.buffers())
    return int(total)


def params_report(model: nn.Module) -> dict:
    """Deterministic parameter/buffer census.

    `trainable_fraction` is 0.0 for linear_probe+softmax (0 trainable
    params, the training-free nearest-prototype baseline) -- a real value,
    not a missing one.
    """
    trainable = count_trainable_params(model)
    total = count_total_params(model)
    n_buffers = sum(1 for _ in model.buffers())
    buffer_numel = sum(b.numel() for b in model.buffers())
    return {
        "trainable": int(trainable),
        "total": int(total),
        "frozen": int(total - trainable),
        "trainable_fraction": (trainable / total) if total else 0.0,
        "n_buffers": int(n_buffers),
        "buffer_numel": int(buffer_numel),
        "bytes_fp32": {
            "trainable": int(trainable * 4),
            "total": int(total * 4),
            "buffers": int(buffer_numel * 4),
        },
    }


# --------------------------------------------------------------------------
# FLOPs / MACs (deterministic) — fvcore backend, per implementation.txt 11.1
# --------------------------------------------------------------------------
def flops_backend_available() -> dict:
    try:
        import fvcore  # noqa: F401
        return {"fvcore": True}
    except ImportError:
        return {"fvcore": False}


def _module_device(model: nn.Module) -> torch.device:
    """The device a trace input for `model` must live on — taken from the
    model's first parameter, then its first buffer, else CPU.

    Exists because `count_flops_detailed` builds its OWN fvcore trace tensor.
    Until 2026-08-09 that tensor was unconditionally CPU, so ANY caller that
    had already moved its model to CUDA crashed inside fvcore's trace with
    "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
    should be the same". That is exactly what killed the first canonical
    Step-11 Kaggle session (step_writeups/step11.txt Section 8), and it was
    structurally unreachable from this CPU-only test module — so the fix
    belongs HERE, in the function that owns the tensor, rather than in each
    caller's `.to(device)` ordering. MAC counts are device-independent, so
    following the model's device changes no reported number.
    """
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def count_flops(model: nn.Module, input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
                *, forward: str | Callable = "auto") -> int:
    """SPEC SIGNATURE (implementation.txt 11.1). Returns MACs (see module
    docstring for the convention)."""
    return count_flops_detailed(model, input_shape, forward=forward)["macs"]


def count_flops_detailed(model: nn.Module,
                         input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
                         *, forward: str | Callable = "auto") -> dict:
    """MAC count + provenance. Raises RuntimeError with an actionable message
    if fvcore is not importable (callers that want a soft failure should
    catch this and record `status: "unavailable"` for the FLOPs field only,
    per implementation.txt: fvcore is not preinstalled on Kaggle/Colab and is
    deliberately not vendored into this always-imported module)."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as e:
        raise RuntimeError(
            "count_flops_detailed requires fvcore (implementation.txt "
            "Section 11.1). Install with `pip install fvcore` (now in "
            "requirements.txt) or `uv sync`."
        ) from e

    wrapper = _resolve_forward_module(model, forward)
    # Device taken from the model, never assumed to be CPU -- see
    # _module_device. The MAC count itself is unaffected by the device.
    trace_device = _module_device(model)
    x = torch.zeros(*input_shape, device=trace_device)
    wrapper.eval()
    fca = FlopCountAnalysis(wrapper, (x,))
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    macs = int(fca.total())
    unsupported = {str(k): int(v) for k, v in fca.unsupported_ops().items()}
    try:
        uncalled = sorted(str(m) for m in fca.uncalled_modules())
    except Exception:
        uncalled = []
    try:
        import fvcore
        backend_version = getattr(fvcore, "__version__", "unknown")
    except Exception:
        backend_version = "unknown"
    return {
        "macs": macs,
        "flops_2x_macs": macs * 2,
        "backend": "fvcore.nn.FlopCountAnalysis",
        "backend_version": str(backend_version),
        "input_shape": list(input_shape),
        "trace_device": str(trace_device),
        "uncounted_ops": unsupported,
        "uncounted_note": (
            "BatchNorm / ReLU / h-swish / pooling / residual-add are not "
            "multiply-accumulate ops and are counted by NO MAC counter "
            "(fvcore included). Their omission understates depthwise + "
            "h-swish backbones."
        ),
        "uncalled_modules": uncalled,
    }


def check_reference_flops(backbone_name: str, measured_macs_with_head: int,
                          measured_macs_backbone_only: int | None = None) -> dict:
    """Validation gate against PUBLISHED_REFERENCE_MACS. A gate whose
    failure is silently tolerated is not a gate — callers should abort (or
    require an explicit override flag) on `status == "fail"`."""
    ref = PUBLISHED_REFERENCE_MACS.get(backbone_name)
    if ref is None:
        return {"status": "unknown", "reason": f"no published reference for {backbone_name!r}"}

    tol = float(ref.get("tolerance_frac", 0.05))
    if "macs" in ref:
        point = float(ref["macs"])
        lo, hi = point * (1 - tol), point * (1 + tol)
        published_repr = point
    else:
        lo_pub, hi_pub = ref["macs_range"]
        lo, hi = lo_pub * (1 - tol), hi_pub * (1 + tol)
        published_repr = list(ref["macs_range"])

    status = "pass" if lo <= measured_macs_with_head <= hi else "fail"
    midpoint = (lo + hi) / 2.0
    result = {
        "status": status,
        "measured_macs": int(measured_macs_with_head),
        "published_macs": published_repr,
        "tolerance_frac": tol,
        "ratio_to_midpoint": measured_macs_with_head / midpoint if midpoint else None,
        "source": ref.get("source"),
        "convention_note": ref.get("convention_note"),
        "variant": ref.get("variant"),
    }
    if (measured_macs_backbone_only is not None
            and "classifier_macs_analytical" in ref):
        delta = int(measured_macs_with_head) - int(measured_macs_backbone_only)
        analytical = int(ref["classifier_macs_analytical"])
        result["classifier_delta_measured"] = delta
        result["classifier_delta_analytical"] = analytical
        result["delta_explains_gap"] = (delta == analytical)
    return result


# --------------------------------------------------------------------------
# Latency (measured)
# --------------------------------------------------------------------------
def _stats(values: list[float]) -> dict:
    n = len(values)
    s = sorted(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    p90_idx = min(n - 1, int(round(0.9 * (n - 1))))
    q1_idx = int(round(0.25 * (n - 1)))
    q3_idx = int(round(0.75 * (n - 1)))
    return {
        "mean": mean,
        "std": var ** 0.5,
        "median": statistics.median(values),
        "p90": s[p90_idx],
        "min": s[0],
        "max": s[-1],
        "iqr": s[q3_idx] - s[q1_idx],
    }


_GRAD_CONTEXTS = {
    "inference": torch.inference_mode,
    "no_grad": torch.no_grad,
    "grad": nullcontext,
}


def time_callable(fn: Callable[[], Any], *, device: torch.device | str,
                  n_warmup: int, n_measure: int,
                  grad_mode: str = "inference",
                  timer: str = "auto",
                  keep_raw: bool = False) -> dict:
    """Time `fn()` (a zero-arg closure). Returns a dict with `latency_ms`
    (median/mean/std/p90/min/max/iqr) and, on CUDA, `latency_ms_wall`
    (host-observed perf_counter time alongside the device-side cuda.Event
    time — kernel-launch overhead is a real fraction of batch-1 latency).

    `grad_mode="inference"` is entered ONCE around warmup+loop (not per-rep:
    for a ~0.2ms score-stage unit, per-rep context enter/exit is a
    measurable fraction) and the timed output's `requires_grad` is asserted
    False — the tripwire for the "PEFT/full_ft cells run WITH autograd
    inside adapter_features()" asymmetry documented in
    src/models/bpeft_model.py (forgetting the outer inference context biases
    latency measurements in favour of the frozen-backbone baselines).
    """
    device = torch.device(device)
    if timer == "auto":
        timer = "cuda_event" if device.type == "cuda" else "perf_counter"
    ctx = _GRAD_CONTEXTS[grad_mode]

    times: list[float] = []
    wall_times: list[float] | None = [] if device.type == "cuda" else None
    out = None
    with ctx():
        for _ in range(n_warmup):
            out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(n_measure):
            if device.type == "cuda" and timer == "cuda_event":
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                out = fn()
                ev1.record()
                torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                times.append(float(ev0.elapsed_time(ev1)))
                wall_times.append((t1 - t0) * 1000.0)
            else:
                t0 = time.perf_counter()
                out = fn()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

    if grad_mode == "inference" and bool(getattr(out, "requires_grad", False)):
        raise RuntimeError(
            "time_callable: grad_mode='inference' but the timed output has "
            "requires_grad=True — the inference context was not applied "
            "correctly around the timed call."
        )

    latency_stats = _stats(times)
    latency_stats["n_warmup"] = int(n_warmup)
    latency_stats["n_measure"] = int(n_measure)
    latency_stats["timer"] = ("torch.cuda.Event(enable_timing=True)"
                              if (device.type == "cuda" and timer == "cuda_event")
                              else "time.perf_counter")
    latency_stats["sync"] = ("torch.cuda.synchronize() before and after each rep"
                             if device.type == "cuda" else "n/a (CPU is synchronous)")
    if keep_raw:
        latency_stats["per_rep_ms"] = times

    result = {
        "latency_ms": latency_stats,
        "n_calls": int(n_warmup + n_measure),
        "device": str(device),
        "grad_mode": grad_mode,
        "model_mode": "eval",
    }
    if wall_times is not None:
        wall_stats = _stats(wall_times)
        wall_stats["timer"] = "time.perf_counter"
        wall_stats["note"] = (
            "host-observed; includes kernel-launch overhead, a real fraction "
            "of batch-1 latency on a many-small-kernel CNN"
        )
        result["latency_ms_wall"] = wall_stats
    return result


@contextmanager
def thread_count(n: int | None):
    """Save/restore torch's intra-op thread count. `torch.set_num_threads`
    is process-global — leaking a change here would silently single-thread
    the rest of the process (and, in tests, the rest of the pytest session).
    """
    if n is None or n <= 0:
        yield
        return
    old = torch.get_num_threads()
    torch.set_num_threads(int(n))
    try:
        yield
    finally:
        torch.set_num_threads(old)


def measure_latency_gpu(model: nn.Module,
                        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
                        n_warmup: int = 10, n_measure: int = 100,
                        *, forward: str | Callable = "auto",
                        device: torch.device | str | None = None
                        ) -> tuple[float, float]:
    """SPEC SIGNATURE (implementation.txt 11.1) -> (mean_ms, std_ms).

    Raises RuntimeError if CUDA is unavailable — never silently falls back
    to CPU and labels the result "gpu" (that mislabelling is worse than a
    loud failure)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "measure_latency_gpu requires CUDA; none is available on this "
            "host. Use measure_latency_cpu for a CPU measurement instead."
        )
    device = torch.device(device) if device is not None else torch.device("cuda")
    model = model.to(device)
    model.eval()
    fn0 = _resolve_forward(model, forward)
    x = torch.zeros(*input_shape, device=device)
    result = time_callable(lambda: fn0(x), device=device,
                           n_warmup=n_warmup, n_measure=n_measure,
                           grad_mode="inference")
    return result["latency_ms"]["mean"], result["latency_ms"]["std"]


def measure_latency_cpu(model: nn.Module,
                        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
                        n_warmup: int = 5, n_measure: int = 50,
                        *, num_threads: int = 1,
                        forward: str | Callable = "auto"
                        ) -> tuple[float, float]:
    """SPEC SIGNATURE (implementation.txt 11.1) -> (mean_ms, std_ms).

    `num_threads=1` by default: the analogue of Howard 2019's "single large
    core" phone-latency methodology (PAPER SUMMARIES/CNN_paper_summaries.txt
    Section 5) — the edge proxy this repo has no Jetson/phone to measure on."""
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    fn0 = _resolve_forward(model, forward)
    x = torch.zeros(*input_shape, device=device)
    with thread_count(num_threads):
        result = time_callable(lambda: fn0(x), device=device,
                               n_warmup=n_warmup, n_measure=n_measure,
                               grad_mode="inference")
    return result["latency_ms"]["mean"], result["latency_ms"]["std"]


def episode_input_shapes(cfg) -> dict:
    """Derive episode tensor shapes from a resolved config — never
    hardcoded. Matches src/evaluators/episodic.py's batching (whole support
    tensor, whole query tensor; the 64-image OOD-pool chunking in
    scripts/evaluate.py:_extract_features does not apply here)."""
    n_way = int(cfg["dataset"]["n_way"])
    k_shot = int(cfg["dataset"]["k_shot"])
    q_query = int(cfg["dataset"]["q_query"])
    size = int(cfg["dataset"].get("image_size", 224))
    return {
        "n_way": n_way, "k_shot": k_shot, "q_query": q_query, "image_size": size,
        "support": (n_way * k_shot, 3, size, size),
        "support_labels": (n_way * k_shot,),
        "query": (n_way * q_query, 3, size, size),
    }


# --------------------------------------------------------------------------
# Memory (measured)
# --------------------------------------------------------------------------
def measure_peak_memory(model: nn.Module,
                        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
                        *, forward: str | Callable = "auto",
                        device: torch.device | str | None = None) -> dict:
    """SPEC FUNCTION (implementation.txt 11.1), via
    torch.cuda.max_memory_allocated. Returns status='skipped' with a reason
    on CPU — never a fabricated 0."""
    device = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if device.type != "cuda":
        return {
            "status": "skipped",
            "reason": (
                "no CUDA device; torch.cuda.max_memory_allocated is "
                "unavailable on CPU. resource.getrusage(ru_maxrss) was "
                "considered and rejected: it is a process-lifetime "
                "high-water mark, not resettable per measurement, so it "
                "cannot attribute memory to a specific unit."
            ),
        }
    model = model.to(device)
    model.eval()
    fn0 = _resolve_forward(model, forward)
    x = torch.zeros(*input_shape, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    with torch.inference_mode():
        for _ in range(3):
            _ = fn0(x)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        "status": "ok",
        "weights_bytes": int(weights_bytes),
        "allocated_before_bytes": int(allocated_before),
        "peak_allocated_bytes": int(peak),
        "activation_peak_bytes": int(peak - allocated_before),
        "peak_reserved_bytes": int(peak_reserved),
        "api": "torch.cuda.max_memory_allocated",
        "reset": "torch.cuda.reset_peak_memory_stats() before this measurement",
    }


def measure_train_step_peak_memory(model: nn.Module, cfg, *, device,
                                   n_steps: int = 3) -> dict:
    """ADDITION beyond implementation.txt 11.1 (see step_writeups/step11.txt
    Section 0). Mirrors src/trainers/episodic_trainer.py's train step
    exactly: model.train() then model.backbone.eval(), Adam over the
    trainable params at the config's own lr/weight_decay, cross-entropy
    (softmax) or evidential_mse_loss (evidential) on a synthetic episode.

    n_steps>=2 because Adam allocates its exp_avg/exp_avg_sq state lazily
    inside the FIRST step() call — the peak must be taken over more than one
    step to see the real optimiser-state memory.

    Returns status='skipped' (not a raised exception) when the trainable
    list is empty (linear_probe + softmax: torch.optim.Adam([]) raises
    ValueError; scripts/train.py already short-circuits this case the same
    way) — mirrors that rather than crashing the measurement loop.
    """
    device = torch.device(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = int(sum(p.numel() for p in trainable))
    if n_trainable == 0:
        return {
            "status": "skipped",
            "reason": (
                "0 trainable parameters (e.g. linear_probe+softmax); "
                "torch.optim.Adam([]) raises ValueError on an empty "
                "parameter list, mirroring scripts/train.py's short-circuit."
            ),
            "trainable_params": 0,
            "spec_status": "ADDITION beyond implementation.txt 11.1",
        }

    was_training = model.training
    backbone_was_training = (model.backbone.training
                             if hasattr(model, "backbone") else None)
    req_grad_snapshot = {id(p): p.requires_grad for p in model.parameters()}
    try:
        model.to(device)
        model.train()
        if hasattr(model, "backbone"):
            model.backbone.eval()

        shapes = episode_input_shapes(cfg)
        lr = float(cfg["train"]["lr"])
        wd = float(cfg["train"].get("weight_decay", 0.0))
        opt = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)

        sx = torch.zeros(*shapes["support"], device=device)
        sy = torch.arange(shapes["n_way"], device=device).repeat_interleave(shapes["k_shot"])
        qx = torch.zeros(*shapes["query"], device=device)
        qy = torch.arange(shapes["n_way"], device=device).repeat_interleave(
            shapes["query"][0] // shapes["n_way"])

        interpretation = str(cfg["head"].get("interpretation", "softmax"))

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            opt.zero_grad()
            logits = model.forward_proto(sx, sy, qx)
            if interpretation == "softmax":
                import torch.nn.functional as F
                loss = F.cross_entropy(logits, qy)
            else:
                from ..losses.evidential import evidential_mse_loss
                evidence = model.head.to_evidence(logits)
                target_oh = torch.eye(shapes["n_way"], device=device)[qy]
                kl_weight_max = float(cfg["loss"]["kl_weight_max"])
                loss = evidential_mse_loss(
                    evidence, target_oh, num_classes=shapes["n_way"],
                    kl_weight=kl_weight_max,
                    prior_per_class=float(cfg["loss"].get("prior_per_class", 1.0)),
                    use_variance=bool(cfg["loss"].get("use_variance", True)),
                )
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = {
            "status": "ok",
            "n_steps": int(n_steps),
            "trainable_params": n_trainable,
            "elapsed_ms_total": elapsed_ms,
            "elapsed_ms_per_step": elapsed_ms / n_steps,
            "grad_bytes_analytical": 4 * n_trainable,
            "optimizer_state_bytes_analytical": 8 * n_trainable,
            "spec_status": "ADDITION beyond implementation.txt 11.1",
        }
        if device.type == "cuda":
            result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        else:
            result["peak_allocated_bytes"] = None
            result["memory_note"] = (
                "CPU: no resettable peak-memory API; only elapsed time and "
                "the analytical grad/optimizer-state byte counts are reported."
            )
        return result
    finally:
        model.train(was_training)
        if backbone_was_training is not None:
            model.backbone.train(backbone_was_training)
        for p in model.parameters():
            p.requires_grad = req_grad_snapshot[id(p)]


# --------------------------------------------------------------------------
# Environment / session metadata
# --------------------------------------------------------------------------
def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _git_sha() -> str | None:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(repo_root), capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def collect_env(*, device: torch.device | str | None = None) -> dict:
    """Everything needed to interpret a measured number later — host, torch,
    GPU (if any), determinism flags, and repo provenance."""
    device = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    host = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
    }
    torch_info: dict = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "device_kind": device.type,
    }
    try:
        torch_info["cudnn"] = (torch.backends.cudnn.version()
                               if torch.backends.cudnn.is_available() else None)
    except Exception:
        torch_info["cudnn"] = None
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        torch_info["gpu_name"] = torch.cuda.get_device_name(idx)
        torch_info["gpu_capability"] = list(torch.cuda.get_device_capability(idx))
        torch_info["gpu_total_memory_bytes"] = int(
            torch.cuda.get_device_properties(idx).total_memory)

    flags = {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_num_threads": int(torch.get_num_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        parallel_info = torch.__config__.parallel_info()
    except Exception:
        parallel_info = None

    return {
        "host": host,
        "torch": torch_info,
        "flags": flags,
        "parallel_info": parallel_info,
        "repo": {"git_sha": _git_sha(), "argv": list(sys.argv)},
    }


def device_profile_slug(env: dict, *, num_threads: int | None = None) -> str:
    """Canonical, hardware-bearing profile key so a CPU number can NEVER
    land in a CUDA leaf. Two different T4 hosts still share a slug (both are
    'cuda_tesla-t4'); the session id (see session_id()) is what
    disambiguates individual runs."""
    kind = env["torch"]["device_kind"]
    if kind == "cuda":
        name = env["torch"].get("gpu_name", "unknown-gpu")
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return f"cuda_{slug}"
    cpu_model = env["host"].get("cpu_model", "unknown-cpu")
    slug = re.sub(r"[^a-z0-9]+", "-", cpu_model.lower()).strip("-")
    thread_label = f"{int(num_threads)}thread" if num_threads else "allthreads"
    return f"cpu_{thread_label}_{slug}"


def session_id(env: dict) -> str:
    """s_<UTC compact>_<profile slug>_<sha1(env json)[:6]>."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = device_profile_slug(env)
    digest = hashlib.sha1(
        json.dumps(env, sort_keys=True, default=str).encode()
    ).hexdigest()[:6]
    return f"s_{ts}_{slug}_{digest}"


def gpu_clock_snapshot() -> dict:
    """Best-effort nvidia-smi clock/temperature/power snapshot. Never fatal
    — nvidia-smi is absent on CPU-only dev machines by design."""
    if shutil.which("nvidia-smi") is None:
        return {"status": "unavailable", "reason": "nvidia-smi not found"}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return {"status": "unavailable",
                    "reason": out.stderr.strip() or "nvidia-smi exited non-zero"}
        return {"status": "ok", "raw": out.stdout.strip()}
    except Exception as e:  # noqa: BLE001 — this must never be fatal
        return {"status": "unavailable", "reason": repr(e)}
