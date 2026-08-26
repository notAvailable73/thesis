"""RQ1/RQ2/RQ5 core — factorised objective x score evaluation + VAL-only
evidence-affine refit.

This is the code that is inlined into notebooks/new_rqs.ipynb. It is kept as a
standalone file ONLY so it can be unit-tested on CPU before the Kaggle run; the
notebook carries a verbatim copy.

Design decisions (docs/NEW_RQS_TASK_PLAN.md T1.0a/b/c), all deliberate:

T1.0a  ENERGY on an evidential-trained model is computed on the RAW PROTOTYPE
       LOGITS (pre-`to_evidence`), identically to a softmax-trained model.
       Energy is then the same function of the same quantity in both arms,
       which is exactly what makes the score axis a clean contrast. Computing
       it on alpha would re-confound the two axes.

T1.0b  VACUITY on a softmax-trained model needs an evidence affine, and a
       softmax-trained cell never trained one. We report BOTH:
         vacuity_native  — the checkpoint's own affine. Evidential cells: the
                           TRAINED (scale, bias). Softmax cells: whatever the
                           config left there (base.yaml -> 1.0/0.0). This is
                           the key that must reproduce Step 10 bit-for-bit.
         vacuity_valfit  — (scale, bias) refit on the FROZEN VAL seeds, by the
                           SAME procedure in BOTH arms. This is the headline
                           cross-term, because handing the softmax arm an
                           untuned mapping would stack the comparison exactly
                           the way RQ2 exists to un-stack it.

T1.0c  TS-MSP is well-defined on an evidential-trained model: verified by
       reading scripts/evaluate.py:_fit_val_temperature — it consumes
       `model.forward_proto_from_features` logits and calls `fit_temperature`,
       and never branches on `interpretation`. evaluate.py merely declined to
       CALL it for evidential cells; the function itself is logit-level and
       interpretation-agnostic.

T2.2   The evidence map is NEVER reimplemented here. Every softplus goes
       through `PrototypeHead.to_evidence`, and every evidence->prob/vacuity
       step through `src.evaluators.ood.evidence_to_probs_and_vacuity`. The
       Step 4 collapse happened because train and eval evidence maps drifted.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.datasets import (
    EpisodicIterableDataset, get_id_split, get_svhn_ood, get_heldout_near_ood,
    get_tinyimagenet_ood, get_gaussian_ood,
)
from src.datasets.mini_imagenet import MINI_IMAGENET_ALL_WNIDS
from src.evaluators import (
    accuracy, f1_macro, expected_calibration_error, brier_score,
    ood_auroc, fpr_at_95_tpr, energy_score, fit_temperature,
    evidence_to_probs_and_vacuity,
)
from src.evaluators.temperature import apply_temperature
from src.heads.prototype_head import PrototypeHead
from src.models import build_model

# Reused verbatim from the repo so the OOD-feature path cannot drift.
from scripts.evaluate import _extract_features

#: The four scores of RQ1's score axis, plus the native-affine vacuity kept
#: alongside it as the Step-10 regression anchor / sensitivity row.
FACTORIAL_SCORES = ("msp", "energy", "ts_msp", "vacuity_valfit", "vacuity_native")

# ---------------------------------------------------------------------------
# Process-lifetime caches. A grid sweep evaluates dozens of cells that share
# the same in-distribution splits and the same OOD image pools; only the
# BACKBONE differs, so the images can be loaded once and re-featurised per
# cell. The ID splits are lazy torchvision Datasets (cheap). The OOD pools are
# materialised (N, 3, 224, 224) float tensors -- ~300 MB each at n=500 -- so
# caching all four costs ~1.2 GB of host RAM; set CACHE_OOD_IMAGES=False if a
# session is memory-constrained.
# ---------------------------------------------------------------------------
_SPLIT_CACHE: dict = {}
_OOD_IMG_CACHE: dict = {}
CACHE_OOD_IMAGES = True


def cached_id_split(dataset_cfg, split: str):
    key = (str(dataset_cfg.get("name", "cifar_fs")), split,
           int(dataset_cfg.get("image_size", 224)),
           str(dataset_cfg.get("data_root", "data")),
           json.dumps(list(dataset_cfg.get("class_ids") or []), sort_keys=True))
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = get_id_split(dataset_cfg, split=split)
    return _SPLIT_CACHE[key]


def _cached_ood_images(key, builder):
    if not CACHE_OOD_IMAGES:
        return builder()
    if key not in _OOD_IMG_CACHE:
        _OOD_IMG_CACHE[key] = builder()
    return _OOD_IMG_CACHE[key]


def clear_caches():
    _SPLIT_CACHE.clear()
    _OOD_IMG_CACHE.clear()


# =====================================================================
# Evidence affine: read, override, refit
# =====================================================================
def read_evidence_affine(head: PrototypeHead) -> tuple[float, float]:
    """The (scale, bias) the checkpoint actually carries.

    `scale` is read through the head's own parameterisation (softplus of the
    raw parameter when learnable) rather than recomputed, so it matches what
    `to_evidence` will use.
    """
    with torch.no_grad():
        if head.evidence_affine:
            scale = float(F.softplus(head._evidence_raw_scale).item())
            bias = float(head._evidence_bias.item())
        else:
            scale = float(head._evidence_scale_const.item())
            bias = float(head._evidence_bias_const.item())
    return scale, bias


@contextmanager
def evidence_affine_override(head: PrototypeHead, scale: float, bias: float):
    """Temporarily install (scale, bias) on the head itself.

    Deliberately mutates the head's own parameters/buffers instead of adding a
    second evidence path, so `head.to_evidence` remains the single source of
    truth (T2.2). Restores the originals on exit, including on exception.
    """
    with torch.no_grad():
        if head.evidence_affine:
            old = (head._evidence_raw_scale.detach().clone(),
                   head._evidence_bias.detach().clone())
            raw = torch.log(torch.expm1(torch.tensor(
                max(float(scale), 1e-4), dtype=head._evidence_raw_scale.dtype)))
            head._evidence_raw_scale.copy_(raw.to(head._evidence_raw_scale.device))
            head._evidence_bias.copy_(torch.tensor(
                float(bias), dtype=head._evidence_bias.dtype,
                device=head._evidence_bias.device))
        else:
            old = (head._evidence_scale_const.detach().clone(),
                   head._evidence_bias_const.detach().clone())
            head._evidence_scale_const.copy_(torch.tensor(
                float(scale), dtype=head._evidence_scale_const.dtype,
                device=head._evidence_scale_const.device))
            head._evidence_bias_const.copy_(torch.tensor(
                float(bias), dtype=head._evidence_bias_const.dtype,
                device=head._evidence_bias_const.device))
    try:
        yield
    finally:
        with torch.no_grad():
            if head.evidence_affine:
                head._evidence_raw_scale.copy_(old[0])
                head._evidence_bias.copy_(old[1])
            else:
                head._evidence_scale_const.copy_(old[0])
                head._evidence_bias_const.copy_(old[1])


def fit_evidence_affine(val_logits: torch.Tensor, val_targets: torch.Tensor,
                        *, num_classes: int, prior_per_class: float,
                        scale_init: float, bias_init: float,
                        max_iter: int = 500, lr: float = 0.05) -> tuple[float, float]:
    """RQ2 / T2.1 — refit the two evidence-affine scalars on VAL logits only.

    Mirrors scripts/evaluate.py:_fit_val_temperature's protocol exactly: one
    global fit, on the frozen VAL episodes, minimising NLL (Guo et al. 2017),
    fixed iteration count, no randomness -> deterministic.

    The optimisation runs on a THROWAWAY PrototypeHead whose only role is to
    own two learnable scalars, so `to_evidence` is still the one softplus in
    the codebase. The caller never has to touch the real head's gradients.
    """
    logits = val_logits.detach().float().cpu()
    targets = val_targets.detach().long().cpu()

    fit_head = PrototypeHead(
        metric="l2",  # unused: only to_evidence() is called on this head
        evidence_affine=True,
        evidence_scale_init=max(float(scale_init), 1e-4),
        evidence_bias_init=float(bias_init),
    )
    params = [fit_head._evidence_raw_scale, fit_head._evidence_bias]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        evidence = fit_head.to_evidence(logits)
        probs, _vac = evidence_to_probs_and_vacuity(
            evidence, num_classes, prior_per_class)
        loss = F.nll_loss(torch.log(probs.clamp_min(1e-12)), targets)
        loss.backward()
        opt.step()
    return read_evidence_affine(fit_head)


# =====================================================================
# The factorised score set (T1.1)
# =====================================================================
def all_id_scores(logits: torch.Tensor, head: PrototypeHead, *,
                  num_classes: int, temperature: float | None,
                  prior_per_class: float,
                  affine_valfit: tuple[float, float]) -> dict[str, torch.Tensor]:
    """Every score for BOTH objectives, driven by an explicit list rather than
    an `if interpretation ==` branch (T1.1). Higher => more in-distribution.

    Replaces src/evaluators/episodic.py:_id_score_set's diagonal-only behaviour
    WITHOUT changing it: on a softmax cell msp/energy/ts_msp are computed from
    the identical expressions, and on an evidential cell vacuity_native goes
    through the identical `head.to_evidence` call, so Step 10's numbers are
    reproduced exactly (verified by the T1.6 regression guard).
    """
    scores: dict[str, torch.Tensor] = {}

    probs = torch.softmax(logits, dim=-1)
    scores["msp"] = probs.max(dim=-1).values
    # T1.0a: raw prototype logits in BOTH arms.
    scores["energy"] = energy_score(logits)
    if temperature is not None:
        scores["ts_msp"] = apply_temperature(logits, temperature).max(dim=-1).values

    # Native affine: whatever the checkpoint carries.
    evidence = head.to_evidence(logits)
    _p, vac = evidence_to_probs_and_vacuity(evidence, num_classes, prior_per_class)
    scores["vacuity_native"] = 1.0 - vac

    # VAL-refit affine, same procedure in both arms (T1.0b option ii).
    with evidence_affine_override(head, *affine_valfit):
        evidence_v = head.to_evidence(logits)
    _pv, vac_v = evidence_to_probs_and_vacuity(evidence_v, num_classes, prior_per_class)
    scores["vacuity_valfit"] = 1.0 - vac_v

    # The head's evidence affine is a learnable Parameter on evidential cells,
    # so these carry grad when called outside torch.no_grad(). Scoring is a
    # read-only operation; detaching makes the function safe to call anywhere
    # (e.g. post-hoc re-scoring of a persisted logit dump).
    return {k: v.detach() for k, v in scores.items()}


def all_prob_sets(logits: torch.Tensor, head: PrototypeHead, *,
                  num_classes: int, temperature: float | None,
                  prior_per_class: float,
                  affine_valfit: tuple[float, float]) -> dict[str, torch.Tensor]:
    """Probability vectors under every objective, for ECE / Brier / F1.

    `evidential_native` on an evidential cell is Step 10's `ece_pooled`;
    `evidential_valfit` is RQ2's "after". `softmax` / `ts` are the softmax arm.
    """
    out = {"softmax": torch.softmax(logits, dim=-1)}
    if temperature is not None:
        out["ts"] = apply_temperature(logits, temperature)

    evidence = head.to_evidence(logits)
    out["evidential_native"], _ = evidence_to_probs_and_vacuity(
        evidence, num_classes, prior_per_class)

    with evidence_affine_override(head, *affine_valfit):
        evidence_v = head.to_evidence(logits)
    out["evidential_valfit"], _ = evidence_to_probs_and_vacuity(
        evidence_v, num_classes, prior_per_class)
    return {k: v.detach() for k, v in out.items()}


# =====================================================================
# Data paths (replicated from scripts/evaluate.py, not re-invented)
# =====================================================================
def load_val_logits(model, cfg, device, repo_root: Path):
    """Pooled VAL query logits + targets.

    Byte-for-byte the same data path as scripts/evaluate.py:_fit_val_temperature
    (same file, same seeds, same iterable construction), so a temperature fit
    on top of this reproduces the one Step 10 recorded.

    T2.5 guard: returns the val seed list it actually used so the caller can
    assert it is [10000..10099] and disjoint from the 600 test seeds. The test
    split is never constructed in this function.
    """
    with open(repo_root / "configs" / "val_episodes.yaml") as f:
        val_spec = yaml.safe_load(f)
    val_seeds = list(val_spec["seeds"])
    val_split = cached_id_split(cfg.dataset, "val")
    val_iter = EpisodicIterableDataset(
        val_split, n_way=int(cfg.dataset.n_way), k_shot=int(cfg.dataset.k_shot),
        q_query=int(cfg.dataset.q_query), num_episodes=len(val_seeds),
        seed_offset=int(val_seeds[0]),
    )
    backbone = model.backbone
    logits_all, targets_all = [], []
    model.eval()
    with torch.no_grad():
        for sx, sy, qx, qy in val_iter:
            sf = backbone(sx.to(device))
            qf = backbone(qx.to(device))
            ql = model.forward_proto_from_features(sf, sy.to(device), qf)
            logits_all.append(ql.cpu())
            targets_all.append(qy.cpu())
    return torch.cat(logits_all), torch.cat(targets_all), val_seeds


def build_ood_pools(model, cfg, device, *, use_tinyimagenet=True, use_gaussian=True):
    """The same four pools, in the same insertion order, as the Step 10 grid
    invocation (`--use-tinyimagenet --use-gaussian`). Order matters: the first
    pool is the legacy `primary_ood_pool` (svhn_far)."""
    img_size = int(cfg.dataset.image_size)
    n_ood = int(cfg.ood.num_samples)
    ood_seed = int(cfg.ood.seed)
    pools = {}

    ds_name = str(cfg.dataset.get("name", "cifar_fs"))
    base_key = (ds_name, img_size, n_ood, ood_seed)

    svhn_x = _cached_ood_images(
        ("svhn", img_size, n_ood, ood_seed),
        lambda: get_svhn_ood(data_root=cfg.ood.data_root, image_size=img_size,
                             num_samples=n_ood, seed=ood_seed))
    pools["svhn_far"] = _extract_features(model.backbone, svhn_x, device)

    near_name, heldout_x = _cached_ood_images(
        ("heldout",) + base_key,
        lambda: get_heldout_near_ood(cfg.dataset, num_samples=n_ood,
                                     seed=ood_seed, heldout_split="val"))
    pools[near_name] = _extract_features(model.backbone, heldout_x, device)

    if use_tinyimagenet:
        # Step 9: TinyImageNet-200 shares 25 wnids with MiniImageNet's 100
        # classes, so an "OOD" pool could otherwise contain literal ID images.
        exclude = (MINI_IMAGENET_ALL_WNIDS if ds_name == "mini_imagenet" else None)
        tin_x = _cached_ood_images(
            ("tin",) + base_key,
            lambda: get_tinyimagenet_ood(data_root=cfg.dataset.data_root,
                                         image_size=img_size, num_samples=n_ood,
                                         seed=ood_seed, exclude_wnids=exclude))
        pools["tin_near"] = _extract_features(model.backbone, tin_x, device)

    if use_gaussian:
        gauss_x = _cached_ood_images(
            ("gauss", img_size, n_ood, ood_seed),
            lambda: get_gaussian_ood(image_size=img_size, num_samples=n_ood,
                                     seed=ood_seed))
        pools["gaussian_far"] = _extract_features(model.backbone, gauss_x, device)

    return pools


# =====================================================================
# The factorial evaluation
# =====================================================================
def factorial_evaluate(model, cfg, *, test_seeds, ood_pools, device,
                       temperature, affine_valfit, prior_per_class,
                       ece_bins=15, logits_out: Path | None = None,
                       log_every=100, logger_print=print) -> dict:
    """Run the 600 test episodes once and score them under EVERY
    (objective, score) combination.

    Returns a summary dict whose native-interpretation keys are numerically
    identical to Step 10's, plus the cross-terms. Optionally persists the raw
    per-episode logits (T1.2) so every FUTURE post-hoc scoring question is a
    re-analysis rather than a retrain.
    """
    K = int(cfg.dataset.n_way)
    head = model.head
    n_eval = len(test_seeds)
    seed_offset = int(test_seeds[0])
    if test_seeds != list(range(seed_offset, seed_offset + n_eval)):
        raise ValueError("test seeds must be a contiguous range (see "
                         "scripts/evaluate.py:_evaluate_episodic)")

    test_split = cached_id_split(cfg.dataset, "test")
    test_iter = EpisodicIterableDataset(
        test_split, n_way=K, k_shot=int(cfg.dataset.k_shot),
        q_query=int(cfg.dataset.q_query), num_episodes=n_eval,
        seed_offset=seed_offset,
    )

    pool_names = list(ood_pools.keys())
    ood_pools = {k: v.to(device) for k, v in ood_pools.items()}
    prob_sets = ("softmax", "ts", "evidential_native", "evidential_valfit")

    per_ep = {f"acc__{p}": [] for p in prob_sets}
    per_ep.update({f"f1__{p}": [] for p in prob_sets})
    per_ep.update({f"ece__{p}": [] for p in prob_sets})
    per_ep.update({f"brier__{p}": [] for p in prob_sets})
    auroc_acc = {p: {s: [] for s in FACTORIAL_SCORES} for p in pool_names}
    fpr_acc = {p: {s: [] for s in FACTORIAL_SCORES} for p in pool_names}
    pooled = {p: [] for p in prob_sets}
    pooled_targets = []
    pooled_logits = []   # needed to reproduce ece_ts EXACTLY -- see below

    dump_id, dump_ood, dump_tgt = [], {p: [] for p in pool_names}, []
    # T2.8: keep both vacuity variants so "does the OOD ranking survive the
    # refit?" is answered with a measured number per pool, not an assertion.
    track_id = {"vacuity_native": [], "vacuity_valfit": []}
    track_ood = {p: {"vacuity_native": [], "vacuity_valfit": []} for p in pool_names}

    model.eval()
    backbone = model.backbone
    with torch.no_grad():
        for i, (sx, sy, qx, qy) in enumerate(test_iter):
            sx, sy = sx.to(device), sy.to(device)
            qx, qy = qx.to(device), qy.to(device)
            sf = backbone(sx)
            qf = backbone(qx)
            q_logits = model.forward_proto_from_features(sf, sy, qf)

            probs = all_prob_sets(q_logits, head, num_classes=K,
                                  temperature=temperature,
                                  prior_per_class=prior_per_class,
                                  affine_valfit=affine_valfit)
            for name, p in probs.items():
                per_ep[f"acc__{name}"].append(accuracy(p, qy))
                per_ep[f"f1__{name}"].append(f1_macro(p, qy, num_classes=K))
                per_ep[f"ece__{name}"].append(
                    expected_calibration_error(p, qy, num_bins=ece_bins))
                per_ep[f"brier__{name}"].append(brier_score(p, qy, K))
                pooled[name].append(p.cpu())
            pooled_targets.append(qy.cpu())
            pooled_logits.append(q_logits.cpu())

            id_scores = all_id_scores(q_logits, head, num_classes=K,
                                      temperature=temperature,
                                      prior_per_class=prior_per_class,
                                      affine_valfit=affine_valfit)
            for k in track_id:
                track_id[k].append(id_scores[k].cpu().numpy())
            if logits_out is not None:
                dump_id.append(q_logits.cpu().to(torch.float16).numpy())
                dump_tgt.append(qy.cpu().to(torch.int8).numpy())

            for pname in pool_names:
                ood_logits = model.forward_proto_from_features(
                    sf, sy, ood_pools[pname])
                ood_scores = all_id_scores(ood_logits, head, num_classes=K,
                                           temperature=temperature,
                                           prior_per_class=prior_per_class,
                                           affine_valfit=affine_valfit)
                for k in track_ood[pname]:
                    track_ood[pname][k].append(ood_scores[k].cpu().numpy())
                if logits_out is not None:
                    dump_ood[pname].append(
                        ood_logits.cpu().to(torch.float16).numpy())
                for sname in FACTORIAL_SCORES:
                    if sname not in id_scores:
                        continue
                    id_np = id_scores[sname].cpu().numpy()
                    ood_np = ood_scores[sname].cpu().numpy()
                    auroc_acc[pname][sname].append(ood_auroc(id_np, ood_np))
                    fpr_acc[pname][sname].append(fpr_at_95_tpr(id_np, ood_np))

            if log_every and (i + 1) % log_every == 0:
                logger_print(f"    ep {i + 1}/{n_eval}  "
                             f"acc={np.mean(per_ep['acc__softmax']):.4f}")

    summary: dict = {"num_episodes": n_eval, "n_way": K}
    for name in prob_sets:
        if not per_ep[f"acc__{name}"]:
            continue
        pooled_p = torch.cat(pooled[name], dim=0)
        tgt = torch.cat(pooled_targets, dim=0)
        summary[f"accuracy_mean__{name}"] = float(np.mean(per_ep[f"acc__{name}"]))
        summary[f"accuracy_std__{name}"] = float(np.std(per_ep[f"acc__{name}"]))
        summary[f"f1_macro_mean__{name}"] = float(np.mean(per_ep[f"f1__{name}"]))
        summary[f"ece_per_episode_mean__{name}"] = float(np.mean(per_ep[f"ece__{name}"]))
        summary[f"ece_pooled__{name}"] = float(
            expected_calibration_error(pooled_p, tgt, num_bins=ece_bins))
        summary[f"brier_mean__{name}"] = float(np.mean(per_ep[f"brier__{name}"]))

    # `ece_ts` is the ONE key src/evaluators/episodic.py computes from the
    # CONCATENATED logits (`apply_temperature(pooled_logits_t, T)`) instead of
    # per-episode-then-concatenate like every other pooled metric there.
    # Softmax is row-wise, so the two are mathematically identical -- but
    # torch.softmax picks a different CUDA kernel for (75, K) than for
    # (E*75, K), and the differing reduction order surfaces at ~1e-8. Invisible
    # on CPU, real on a T4. Reproduce the repo's exact expression so the T1.6
    # guard can still report `exact`, rather than widening a tolerance until it
    # would also hide a genuine regression.
    if temperature is not None and pooled_logits:
        pooled_logits_t = torch.cat(pooled_logits, dim=0)
        tgt_all = torch.cat(pooled_targets, dim=0)
        ts_pooled = apply_temperature(pooled_logits_t, temperature)
        summary["ece_pooled__ts"] = float(expected_calibration_error(
            ts_pooled, tgt_all, num_bins=ece_bins))
        # The repo's `brier_ts` is likewise a single POOLED Brier, not the mean
        # of per-episode Briers `brier_mean__ts` reports. Keep both: they are
        # different quantities and only this one is comparable to Step 10.
        summary["brier_pooled__ts"] = float(brier_score(ts_pooled, tgt_all, K))

    for pname in pool_names:
        for sname in FACTORIAL_SCORES:
            vals = auroc_acc[pname][sname]
            if not vals:
                continue
            summary[f"ood_auroc__{pname}__{sname}"] = float(np.mean(vals))
            summary[f"ood_auroc_std__{pname}__{sname}"] = float(np.std(vals))
            summary[f"fpr_at_95_tpr__{pname}__{sname}"] = float(
                np.mean(fpr_acc[pname][sname]))

    # T2.8 — measured, per pool, over the exact ID+OOD vector AUROC ranks.
    id_nat = np.concatenate(track_id["vacuity_native"])
    id_fit = np.concatenate(track_id["vacuity_valfit"])
    for pname in pool_names:
        a = np.concatenate([id_nat, np.concatenate(track_ood[pname]["vacuity_native"])])
        b = np.concatenate([id_fit, np.concatenate(track_ood[pname]["vacuity_valfit"])])
        summary[f"ranking_shift__{pname}"] = ranking_shift(a, b)

    if logits_out is not None:
        logits_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            logits_out,
            id_logits=np.stack(dump_id),          # (E, Q, K) float16
            id_targets=np.stack(dump_tgt),        # (E, Q)    int8
            pool_names=np.array(pool_names),
            **{f"ood_logits__{p}": np.stack(dump_ood[p]) for p in pool_names},
        )
        summary["logits_dump"] = str(logits_out)
        summary["logits_dump_bytes"] = int(logits_out.stat().st_size)

    return summary


# =====================================================================
# T1.6 regression guard
# =====================================================================
def native_score_name(interpretation: str) -> str:
    return "vacuity_native" if interpretation == "evidential" else "msp"


def regression_guard(summary: dict, committed_path: Path, interpretation: str,
                     *, tol: float = 1e-6) -> dict:
    """T1.6 — prove the refactor added cross-terms without perturbing the
    diagonal, by diffing against the COMMITTED Step 10 metrics JSON.

    Checks every key Step 10 wrote for this cell's native interpretation, and
    grades the result in three tiers rather than pass/fail:

      exact       every key bit-identical. Expected when re-running on the SAME
                  hardware the committed numbers came from. Measured: 12/12 and
                  13/13 exact when the old and new evaluators run in one process.
      within_tol  max |diff| <= tol. This is a PASS. Re-running the committed
                  cifar_1shot/mbnet/lora cell on CPU against numbers produced on
                  a Kaggle T4 gives max |diff| = 2.2e-7 -- float32 accumulation
                  differing across devices, which flips a handful of near-tied
                  ID/OOD pairs and moves a 600-episode mean AUROC in the 7th
                  decimal. Nothing about the logic changed.
      MISMATCH    max |diff| > tol: a real difference, investigate before
                  quoting anything.

    Why 1e-6 and not something tighter: a genuine logic error here -- wrong
    score, wrong affine, wrong pool -- moves AUROC by 1e-2 to 1e-1, four to five
    orders of magnitude above the float32 noise floor. A threshold in between
    separates them cleanly without ever calling hardware noise a defect.
    """
    if not committed_path.exists():
        return {"status": "no_committed_file", "path": str(committed_path)}
    old = json.load(open(committed_path))
    native = native_score_name(interpretation)
    new_prob_set = "evidential_native" if interpretation == "evidential" else "softmax"

    checks: dict[str, tuple[float, float]] = {}
    for key, val in old.items():
        if key.startswith("ood_auroc__") or key.startswith("fpr_at_95_tpr__"):
            head, pool, score = key.split("__")
            if score != ("vacuity" if interpretation == "evidential" else "msp"):
                continue
            checks[key] = (float(val), summary.get(f"{head}__{pool}__{native}", float("nan")))
    for old_key, new_key in (
        ("accuracy_mean", f"accuracy_mean__{new_prob_set}"),
        ("ece_pooled", f"ece_pooled__{new_prob_set}"),
        ("brier_mean", f"brier_mean__{new_prob_set}"),
        ("f1_macro_mean", f"f1_macro_mean__{new_prob_set}"),
        ("ece_ts", "ece_pooled__ts"),
    ):
        if old_key in old and new_key in summary:
            checks[old_key] = (float(old[old_key]), float(summary[new_key]))

    diffs = {k: abs(a - b) for k, (a, b) in checks.items()}
    if not diffs:
        return {"status": "no_comparable_keys", "path": str(committed_path)}
    max_key = max(diffs, key=diffs.get)
    n_exact = sum(1 for d in diffs.values() if d == 0.0)
    return {
        "status": ("exact" if n_exact == len(diffs)
                   else "within_tol" if diffs[max_key] <= tol else "MISMATCH"),
        "tol": tol,
        "n_keys": len(diffs),
        "n_exact": n_exact,
        "max_abs_diff": float(diffs[max_key]),
        "max_abs_diff_key": max_key,
        "path": str(committed_path),
    }


# =====================================================================
# T2.8 — is vacuity reordering even possible under an affine change?
# =====================================================================
def ranking_shift(score_a: np.ndarray, score_b: np.ndarray,
                  *, max_pairs: int = 2_000_000, seed: int = 0) -> dict:
    """Turn RQ2's central question into a measurement.

    `softplus(a*x + b)` is monotone in each logit separately, but vacuity is
    K / sum_k(alpha_k) — a function of the SUM. A monotone per-logit map is not
    monotone in the sum, so reordering is possible in principle. This reports
    whether it happens in practice: Spearman rho and the exact fraction of
    sample pairs whose relative order flips.
    """
    a = np.asarray(score_a, dtype=np.float64).ravel()
    b = np.asarray(score_b, dtype=np.float64).ravel()
    n = len(a)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    rho = float(np.corrcoef(ra, rb)[0, 1]) if n > 1 else 1.0

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        da = a[:, None] - a[None, :]
        db = b[:, None] - b[None, :]
        iu = np.triu_indices(n, k=1)
        disc = int(np.sum(np.sign(da[iu]) * np.sign(db[iu]) < 0))
        sampled = total_pairs
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        disc = int(np.sum(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) < 0))
        sampled = len(i)
    return {
        "spearman_rho": rho,
        "discordant_pairs": disc,
        "pairs_compared": int(sampled),
        "discordant_fraction": float(disc / sampled) if sampled else 0.0,
        "exhaustive": total_pairs <= max_pairs,
        "reordering_observed": disc > 0,
    }
