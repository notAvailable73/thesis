"""Evaluate a trained B-PEFT checkpoint over a fixed list of test episodes.

Two modes, dispatched on cfg.trainer.type:

  trainer.type: single_episode    (Step 1-3 legacy)
      Per-episode fine-tune protocol: build a fresh model each episode,
      train on the support set for 200 inner steps, evaluate on the
      query set. The 600 test seeds come from configs/test_episodes.yaml.

  trainer.type: episodic          (Step 4 / Phase 2)
      Episodic test protocol: load the meta-trained adapter checkpoint
      (frozen), prototype-classify each episode's query set, score OOD
      samples against THIS episode's prototypes. Same 600 seeds, no
      inner fitting.

Both modes:
  - read the canonical seed list from configs/test_episodes.yaml,
  - support --num-episodes K to truncate to the first K seeds,
  - dump a metrics JSON with sort_keys=True for byte-identical reruns,
  - log to W&B (online | offline | disabled).

Usage:
    python scripts/evaluate.py --config configs/exp_step1.yaml          # legacy
    python scripts/evaluate.py --config configs/exp_phase2_evidential.yaml
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from src.utils import (
    set_seed, get_device, count_trainable_params, load_config, get_logger,
    WandbRun, make_run_name, reliability_diagram, ood_histogram, confusion_matrix,
)
from src.datasets import (
    build_dataset, sample_episode, get_svhn_ood, get_cifar_fs,
    get_cifar_fs_heldout_ood, get_tinyimagenet_ood, get_gaussian_ood,
    EpisodicIterableDataset,
)
from src.models import build_model
from src.losses import build_loss
from src.trainers import train_one_episode
from src.evaluators import (
    accuracy, f1_macro, expected_calibration_error, brier_score,
    ood_auroc, fpr_at_95_tpr, energy_score, fit_temperature,
    evidence_to_probs_and_vacuity, logits_to_probs_and_uncertainty,
    evaluate_episodic,
)


def _head_descriptor(cfg) -> str:
    """Same as scripts/train.py:_head_descriptor — kept in sync.

    Step 1-3: head.type is sufficient (softmax / evidential).
    Step 4+:  prototype configs need head.interpretation appended,
              otherwise the two Phase 2 configs would collide on disk.
    """
    head_type = cfg.head.type
    if head_type == "prototype":
        return f"prototype-{cfg.head.get('interpretation', 'evidential')}"
    return head_type


# =====================================================================
# Shared helpers
# =====================================================================
def _extract_features(backbone, x, device, batch_size=64):
    """Run the frozen backbone once over `x` and return (N, D) features on CPU."""
    backbone.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(device)
            chunks.append(backbone(batch).cpu())
    return torch.cat(chunks, dim=0)


def _predict_from_features(model, feats, head_type, device,
                           batch_size=256, num_classes=5):
    """Step 1-3 single-episode path: model with LinearHead / EvidentialHead."""
    model.eval()
    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            batch = feats[i:i + batch_size].to(device)
            out_chunks.append(model.forward_from_features(batch).cpu())
    output = torch.cat(out_chunks, dim=0)
    if head_type == "evidential":
        return evidence_to_probs_and_vacuity(output, num_classes)
    return logits_to_probs_and_uncertainty(output)


def _load_test_seeds(repo_root: Path, cfg) -> list[int]:
    eps_path = repo_root / cfg.eval.episodes_file
    with open(eps_path) as f:
        spec = yaml.safe_load(f)
    seeds = list(spec["seeds"])
    assert len(seeds) == int(spec["num_episodes"]), \
        f"{eps_path}: num_episodes ({spec['num_episodes']}) != len(seeds) ({len(seeds)})"
    return seeds


def _build_wandb_run(cfg, args, head_type, extra_tags=None) -> WandbRun:
    wcfg = cfg.get("wandb", None) if isinstance(cfg, dict) else None
    project = (wcfg or {}).get("project", "bpeft-thesis") if wcfg else "bpeft-thesis"
    base_mode = (wcfg or {}).get("mode", "online") if wcfg else "online"
    disabled = bool((wcfg or {}).get("disabled", False)) if wcfg else False
    group = (wcfg or {}).get("group", None) if wcfg else None
    tags = list((wcfg or {}).get("tags", []) or []) if wcfg else []
    if extra_tags:
        tags = tags + list(extra_tags)

    mode = args.wandb_mode or base_mode
    if args.wandb_mode == "disabled":
        disabled = True
    return WandbRun(
        project=project,
        run_name=make_run_name(cfg) + "_eval",
        config={
            "config_path": str(Path(args.config).resolve()),
            "seed": int(cfg.seed),
            "head": dict(cfg.head),
            "adapter": dict(cfg.adapter),
            "loss": dict(cfg.loss),
            "trainer": dict(cfg.trainer),
            "eval": dict(cfg.eval),
            "ood": dict(cfg.ood),
        },
        mode=mode, disabled=disabled, group=group,
        tags=tags + [f"head:{head_type}", f"adapter:{cfg.adapter.type}"],
        job_type="evaluate",
    )


# =====================================================================
# Step 1-3 legacy path (per-episode fine-tune)
# =====================================================================
def _evaluate_finetune(cfg, args, logger, device, wb, seeds, repo_root) -> dict:
    """Existing Step 3 per-episode fine-tune evaluator. Verbatim from the
    pre-Step-4 version of this script."""
    n_eval = len(seeds)
    set_seed(int(cfg.seed))
    head_type = cfg.head.type
    K = int(cfg.dataset.n_way)

    dataset = build_dataset(dict(cfg.dataset))
    svhn_x = get_svhn_ood(
        data_root=cfg.ood.data_root,
        image_size=int(cfg.dataset.image_size),
        num_samples=int(cfg.ood.num_samples),
        seed=int(cfg.ood.seed),
    )
    shared_model = build_model(cfg).to(device)
    svhn_feats = _extract_features(shared_model.backbone, svhn_x, device)
    logger.info(f"cached SVHN features: {tuple(svhn_feats.shape)}")

    per_ep_acc, per_ep_ece, per_ep_brier, per_ep_auroc = [], [], [], []
    pooled_probs, pooled_targets = [], []
    last_id_scores = None
    last_ood_scores = None

    for i, ep_seed in enumerate(seeds):
        set_seed(int(cfg.seed) + int(ep_seed))
        model = build_model(cfg).to(device)
        loss_spec = dict(cfg.loss)
        loss_spec["type"] = "evidential" if head_type == "evidential" else "cross_entropy"
        loss_fn = build_loss(loss_spec)

        support_x, support_y, query_x, query_y = sample_episode(
            dataset=dataset,
            class_ids=list(cfg.dataset.class_ids) if cfg.dataset.get("class_ids") else None,
            n_way=K, k_shot=int(cfg.dataset.k_shot),
            q_query=int(cfg.dataset.q_query),
            seed=int(ep_seed),
        )
        support_feats = _extract_features(model.backbone, support_x, device).to(device)
        query_feats   = _extract_features(model.backbone, query_x, device)
        support_y_d = support_y.to(device)

        train_one_episode(
            model=model,
            support_x=support_feats, support_y=support_y_d,
            loss_fn=loss_fn, num_classes=K, head_type=head_type,
            lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
            num_steps=int(cfg.train.num_steps),
            log_every=10**9, logger=None, wandb_run=None,
        )

        probs, vac = _predict_from_features(model, query_feats, head_type, device, num_classes=K)
        acc = accuracy(probs, query_y)
        ece = expected_calibration_error(probs, query_y, num_bins=int(cfg.eval.ece_bins))
        bri = brier_score(probs, query_y, K)

        _, ood_vac = _predict_from_features(model, svhn_feats, head_type, device, num_classes=K)
        id_scores  = (1.0 - vac).numpy()
        ood_scores = (1.0 - ood_vac).numpy()
        auroc = ood_auroc(id_scores, ood_scores)
        last_id_scores, last_ood_scores = id_scores, ood_scores

        per_ep_acc.append(acc); per_ep_ece.append(ece)
        per_ep_brier.append(bri); per_ep_auroc.append(auroc)
        pooled_probs.append(probs); pooled_targets.append(query_y)
        logger.info(
            f"ep {i:3d} (seed={ep_seed:3d})  acc={acc:.3f}  ECE={ece:.3f}  "
            f"Brier={bri:.3f}  AUROC={auroc:.3f}"
        )
        wb.log({
            "eval/episode": i,
            "eval/episode_seed": int(ep_seed),
            "eval/accuracy": float(acc),
            "eval/ECE": float(ece),
            "eval/Brier": float(bri),
            "eval/OOD_AUROC": float(auroc),
            "eval/accuracy_running_mean": float(np.mean(per_ep_acc)),
            "eval/ECE_running_mean": float(np.mean(per_ep_ece)),
            "eval/AUROC_running_mean": float(np.mean(per_ep_auroc)),
        })

    pooled_probs = torch.cat(pooled_probs, dim=0)
    pooled_targets = torch.cat(pooled_targets, dim=0)
    pooled_ece = expected_calibration_error(
        pooled_probs, pooled_targets, num_bins=int(cfg.eval.ece_bins),
    )

    summary = {
        "accuracy_mean": float(np.mean(per_ep_acc)),
        "accuracy_std":  float(np.std(per_ep_acc)),
        "accuracy_ci95": float(1.96 * np.std(per_ep_acc) / np.sqrt(n_eval)),
        "adapter_type": cfg.adapter.type,
        "brier_mean":          float(np.mean(per_ep_brier)),
        "brier_std":           float(np.std(per_ep_brier)),
        "config_path": str(Path(args.config).resolve()),
        "ece_per_episode_mean": float(np.mean(per_ep_ece)),
        "ece_per_episode_std":  float(np.std(per_ep_ece)),
        "ece_pooled":          float(pooled_ece),
        "episodes_file": str(cfg.eval.episodes_file),
        "head_type": head_type,
        "n_params": int(count_trainable_params(build_model(cfg))),
        "num_episodes": int(n_eval),
        "ood_auroc_mean":      float(np.mean(per_ep_auroc)),
        "ood_auroc_std":       float(np.std(per_ep_auroc)),
        "seed": int(cfg.seed),
        "seeds_first10": [int(s) for s in seeds[:10]],
        "seeds_last10":  [int(s) for s in seeds[-10:]],
        "trainer_type": "single_episode",
    }
    return {
        "summary": summary,
        "pooled_probs": pooled_probs,
        "pooled_targets": pooled_targets,
        "last_id_scores": last_id_scores,
        "last_ood_scores": last_ood_scores,
    }


# =====================================================================
# Step 4 / Phase 2 path (prototype head, frozen meta-trained adapter)
# =====================================================================
def _fit_val_temperature(model, cfg, device, repo_root, logger) -> float:
    """Fit one global temperature T on the FROZEN val episodes (seeds from
    configs/val_episodes.yaml), Guo et al. 2017. Softmax interpretation only.
    The val seeds ([10000..10099]) are disjoint from the 600 test seeds
    ([0..599]) -> no test leakage. T is then applied unchanged to every test
    episode to produce the ts_msp OOD score + ece_ts/brier_ts calibration."""
    with open(repo_root / "configs" / "val_episodes.yaml") as f:
        val_spec = yaml.safe_load(f)
    val_seeds = list(val_spec["seeds"])
    val_split = get_cifar_fs(
        data_root=cfg.dataset.data_root,
        image_size=int(cfg.dataset.image_size), split="val",
    )
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
    T = fit_temperature(torch.cat(logits_all), torch.cat(targets_all))
    logger.info(f"fit temperature on {len(val_seeds)} val episodes: T={T:.4f}")
    return T


def _evaluate_episodic(cfg, args, logger, device, wb, seeds, repo_root) -> dict:
    """Phase 2 evaluation: load the meta-trained adapter, freeze it, run
    prototype-head over the 600 test episodes."""
    n_eval = len(seeds)
    head_type = cfg.head.type
    interp = cfg.head.get("interpretation", "evidential")
    K = int(cfg.dataset.n_way)

    # --- Test split (Bertinetto 20 test classes) ----------------------
    test_split = get_cifar_fs(
        data_root=cfg.dataset.data_root,
        image_size=int(cfg.dataset.image_size),
        split="test",
    )

    # Build EpisodicIterableDataset with seed_offset = seeds[0] so the
    # 600 episodes match configs/test_episodes.yaml's seeds exactly.
    # (test_episodes.yaml uses seeds [0..599], so seed_offset=0.)
    seed_offset = int(seeds[0])
    # Sanity check: seeds must be a contiguous range starting from
    # seed_offset, since EpisodicIterableDataset increments by 1.
    expected = list(range(seed_offset, seed_offset + n_eval))
    if seeds != expected:
        raise ValueError(
            "configs/test_episodes.yaml seeds are not a contiguous "
            "range starting from seeds[0]; episodic evaluator requires "
            "contiguous seeds. Got non-contiguous seeds."
        )

    test_iter = EpisodicIterableDataset(
        test_split, n_way=K, k_shot=int(cfg.dataset.k_shot),
        q_query=int(cfg.dataset.q_query),
        num_episodes=n_eval, seed_offset=seed_offset,
    )

    # --- Load checkpoint ----------------------------------------------
    model = build_model(cfg).to(device)
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_dir = Path(cfg.output.checkpoint_dir)
        tag = f"{cfg.adapter.type}_{_head_descriptor(cfg)}_seed{cfg.seed}"
        ckpt_path = ckpt_dir / f"model_phase2_{tag}.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    best_val_epoch = int(ckpt.get("best_val_epoch", -1))
    logger.info(
        f"loaded checkpoint: {ckpt_path}  "
        f"best_val_epoch={best_val_epoch}  "
        f"best_val_acc={ckpt.get('best_val_acc', float('nan')):.3f}"
    )

    # --- OOD pools (Step 4.5 / W3): far SVHN + near CIFAR-100-heldout
    #     + optional near TinyImageNet. Each is precomputed backbone features. -
    img_size = int(cfg.dataset.image_size)
    n_ood = int(cfg.ood.num_samples)
    ood_seed = int(cfg.ood.seed)
    pools = {}
    svhn_x = get_svhn_ood(data_root=cfg.ood.data_root, image_size=img_size,
                          num_samples=n_ood, seed=ood_seed)
    pools["svhn_far"] = _extract_features(model.backbone, svhn_x, device)
    heldout_x = get_cifar_fs_heldout_ood(
        data_root=cfg.dataset.data_root, image_size=img_size,
        num_samples=n_ood, seed=ood_seed, heldout_split="val")
    pools["cifar100_near"] = _extract_features(model.backbone, heldout_x, device)
    # TinyImageNet near-OOD: from the config OR the --use-tinyimagenet CLI flag
    # (the flag lets us add it to an ALREADY-TRAINED checkpoint, eval-only, with
    # no retrain).
    use_tin = bool(cfg.ood.get("use_tinyimagenet", False)) or bool(
        getattr(args, "use_tinyimagenet", False))
    if use_tin:
        tin_x = get_tinyimagenet_ood(data_root=cfg.dataset.data_root,
                                     image_size=img_size, num_samples=n_ood,
                                     seed=ood_seed)
        pools["tin_near"] = _extract_features(model.backbone, tin_x, device)
    # Gaussian far-OOD (Step 7): pure-noise sanity/ablation pool. Same config OR
    # --use-gaussian CLI flag pattern as TinyImageNet (eval-only, no retrain).
    use_gauss = bool(cfg.ood.get("use_gaussian", False)) or bool(
        getattr(args, "use_gaussian", False))
    if use_gauss:
        gauss_x = get_gaussian_ood(image_size=img_size, num_samples=n_ood,
                                   seed=ood_seed)
        pools["gaussian_far"] = _extract_features(model.backbone, gauss_x, device)
    logger.info("OOD pools: " + ", ".join(
        f"{k}={tuple(v.shape)}" for k, v in pools.items()))

    # Temperature scaling baseline (softmax only), fit on the val episodes.
    T = None
    if interp == "softmax":
        T = _fit_val_temperature(model, cfg, device, repo_root, logger)

    prior_pc = float(cfg.loss.get("prior_per_class", 1.0))

    # --- Run the episodic evaluator (score x OOD-pool matrix) ---------
    result = evaluate_episodic(
        model=model,
        test_iterable=test_iter,
        ood_pools=pools,
        num_classes=K,
        interpretation=interp,
        ece_bins=int(cfg.eval.ece_bins),
        temperature=T,
        prior_per_class=prior_pc,
        device=device,
        logger=logger,
        wandb_run=wb,
    )
    base_summary = result["summary"]

    # Augment with config-level metadata (so the JSON schema matches
    # Step 3's + the new Phase 2 fields).
    base_summary.update({
        "adapter_type": cfg.adapter.type,
        "config_path":  str(Path(args.config).resolve()),
        "episodes_file": str(cfg.eval.episodes_file),
        "head_type": head_type,
        "interpretation": interp,
        "n_params": int(count_trainable_params(build_model(cfg))),
        "seed": int(cfg.seed),
        "seeds_first10": [int(s) for s in seeds[:10]],
        "seeds_last10":  [int(s) for s in seeds[-10:]],
        "trainer_type": "episodic",
        "best_val_epoch": int(best_val_epoch),
        "temperature": (float(T) if T is not None else 0.0),
        "prior_per_class": prior_pc,
    })
    return {
        "summary": base_summary,
        "pooled_probs": result["pooled_probs"],
        "pooled_targets": result["pooled_targets"],
        "last_id_scores": result["last_id_scores"],
        "last_ood_scores": result["last_ood_scores"],
    }


# =====================================================================
# main
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="If omitted, derived from config "
                             "(adapter+head+seed).")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Truncate the canonical seed list to the "
                             "first K seeds.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        default=None, help="Override cfg.wandb.mode")
    parser.add_argument("--results-suffix", default="step3",
                        help="Prefix for output JSON / PNG files in "
                             "results_dir.")
    parser.add_argument("--use-tinyimagenet", action="store_true",
                        help="Add the TinyImageNet near-OOD pool at eval time "
                             "(no retrain needed); ORs with cfg.ood.use_tinyimagenet.")
    parser.add_argument("--use-gaussian", action="store_true",
                        help="Add the Gaussian-noise far-OOD pool at eval time "
                             "(sanity/ablation); ORs with cfg.ood.use_gaussian.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("bpeft.evaluate")
    repo_root = Path(__file__).resolve().parents[1]

    seeds = _load_test_seeds(repo_root, cfg)
    if args.num_episodes is not None:
        seeds = seeds[: int(args.num_episodes)]
    n_eval = len(seeds)
    trainer_type = cfg.get("trainer", {}).get("type", "single_episode") if isinstance(cfg, dict) else "single_episode"
    logger.info(
        f"config={args.config}  num_episodes={n_eval}  "
        f"trainer.type={trainer_type}  "
        f"(seeds from {cfg.eval.episodes_file})"
    )

    set_seed(int(cfg.seed))
    device = get_device()
    head_type = cfg.head.type
    K = int(cfg.dataset.n_way)

    extra_tags = [f"trainer:{trainer_type}",
                  f"phase:{'2' if trainer_type == 'episodic' else '1'}"]
    wb = _build_wandb_run(cfg, args, head_type, extra_tags=extra_tags)
    if not wb.disabled:
        logger.info(f"wandb run: {wb.run_name}  url={wb.url}")

    if trainer_type == "single_episode":
        bundle = _evaluate_finetune(cfg, args, logger, device, wb, seeds, repo_root)
    elif trainer_type == "episodic":
        bundle = _evaluate_episodic(cfg, args, logger, device, wb, seeds, repo_root)
    else:
        raise ValueError(
            f"Unknown trainer.type: {trainer_type!r}; expected "
            f"'single_episode' or 'episodic'."
        )

    summary = bundle["summary"]
    pooled_probs   = bundle["pooled_probs"]
    pooled_targets = bundle["pooled_targets"]
    last_id_scores = bundle["last_id_scores"]
    last_ood_scores = bundle["last_ood_scores"]

    # --- Final artifacts ----------------------------------------------
    out_dir = Path(cfg.output.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.results_suffix}_{cfg.adapter.type}_{_head_descriptor(cfg)}"
    metrics_path = out_dir / f"{tag}_metrics.json"
    rel_path     = out_dir / f"{tag}_reliability.png"
    hist_path    = out_dir / f"{tag}_ood_histogram.png"
    cm_path      = out_dir / f"{tag}_confusion_matrix.png"

    reliability_diagram(
        pooled_probs, pooled_targets, rel_path,
        num_bins=int(cfg.eval.ece_bins),
        title=f"Reliability ({head_type}, {cfg.adapter.type})  "
              f"ECE={summary['ece_pooled']:.3f}",
    )
    if last_id_scores is not None and last_ood_scores is not None:
        ood_histogram(
            last_id_scores, last_ood_scores, hist_path,
            title=f"ID vs SVHN ({head_type})  "
                  f"AUROC={summary['ood_auroc_mean']:.3f}",
        )
    confusion_matrix(
        pooled_probs, pooled_targets, cm_path, num_classes=K,
        title=f"Confusion ({head_type})  "
              f"acc={summary['accuracy_mean']:.3f}",
    )

    # sort_keys=True so a re-run on a fixed seed yields a byte-identical
    # metrics.json (Step 3 exit criterion still applies in Phase 2).
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logger.info(f"saved metrics: {metrics_path}")

    wb.update_summary({
        "final/accuracy_mean":  summary["accuracy_mean"],
        "final/accuracy_ci95":  summary.get("accuracy_ci95", 0.0),
        "final/f1_macro_mean":  summary.get("f1_macro_mean", 0.0),
        "final/ece_pooled":     summary["ece_pooled"],
        "final/ece_per_episode_mean": summary["ece_per_episode_mean"],
        "final/brier_mean":     summary["brier_mean"],
        "final/ood_auroc_mean": summary["ood_auroc_mean"],
        "final/fpr_at_95_tpr_mean": summary.get("fpr_at_95_tpr_mean", 1.0),
        "final/num_episodes":   summary["num_episodes"],
    })
    wb.log_image("plots/reliability_diagram", str(rel_path),
                 caption=f"{head_type} reliability")
    wb.log_image("plots/ood_histogram", str(hist_path),
                 caption=f"{head_type} ID vs OOD scores (last episode)")
    wb.log_image("plots/confusion_matrix", str(cm_path),
                 caption=f"{head_type} confusion matrix (pooled)")
    wb.log_artifact(str(metrics_path), artifact_name=f"metrics_{tag}",
                    artifact_type="metrics")
    wb.finish()

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
