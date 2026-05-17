"""Evaluate a trained B-PEFT checkpoint over a fixed list of test episodes.

Usage:
    python scripts/evaluate.py --config configs/exp_step1.yaml
    python scripts/evaluate.py --config configs/exp_step1.yaml --num-episodes 50

Step 3 changes (post-defence plan, action 3.1 + 3.3 + 3.4 + 3.7):
  - episode seeds come from configs/test_episodes.yaml (committed list of
    600 seeds). --num-episodes K truncates to the first K of that list, so
    quick checks and full runs use the *same* seed prefix.
  - per-episode acc / ECE / Brier / OOD-AUROC are logged to W&B.
  - final reliability diagram, OOD histogram, confusion matrix are saved to
    cfg.output.results_dir and logged as W&B images.
  - the summary JSON is sorted (sort_keys=True) so a re-run on a fixed seed
    yields a byte-identical metrics.json (Step 3 exit criterion).
"""
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
from src.datasets import build_dataset, sample_episode, get_svhn_ood
from src.models import build_model
from src.losses import build_loss
from src.trainers import train_one_episode
from src.evaluators import (
    accuracy, expected_calibration_error, brier_score, ood_auroc,
    evidence_to_probs_and_vacuity, logits_to_probs_and_uncertainty,
)


def _extract_features(backbone, x, device, batch_size=64):
    """Run the frozen backbone once over `x` and return (N, D) features on CPU."""
    backbone.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(device)
            chunks.append(backbone(batch).cpu())
    return torch.cat(chunks, dim=0)


def _predict_from_features(model, feats, head_type, device, batch_size=256, num_classes=5):
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
    """Step 3 action 3.7: load the 600 canonical seeds from disk."""
    eps_path = repo_root / cfg.eval.episodes_file
    with open(eps_path) as f:
        spec = yaml.safe_load(f)
    seeds = list(spec["seeds"])
    assert len(seeds) == int(spec["num_episodes"]), \
        f"{eps_path}: num_episodes ({spec['num_episodes']}) != len(seeds) ({len(seeds)})"
    return seeds


def _build_wandb_run(cfg, args, head_type) -> WandbRun:
    wcfg = cfg.get("wandb", None) if isinstance(cfg, dict) else None
    project = (wcfg or {}).get("project", "bpeft-thesis") if wcfg else "bpeft-thesis"
    base_mode = (wcfg or {}).get("mode", "online") if wcfg else "online"
    disabled = bool((wcfg or {}).get("disabled", False)) if wcfg else False
    group = (wcfg or {}).get("group", None) if wcfg else None
    tags = list((wcfg or {}).get("tags", []) or []) if wcfg else []

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
            "eval": dict(cfg.eval),
            "ood": dict(cfg.ood),
        },
        mode=mode, disabled=disabled, group=group,
        tags=tags + [f"head:{head_type}", f"adapter:{cfg.adapter.type}", "step3-eval"],
        job_type="evaluate",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="If omitted, derived from config (adapter+head+seed).")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Truncate the canonical seed list to the first K seeds.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        default=None, help="Override cfg.wandb.mode")
    parser.add_argument("--results-suffix", default="step3",
                        help="Prefix for output JSON / PNG files in results_dir.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("bpeft.evaluate")
    repo_root = Path(__file__).resolve().parents[1]

    # --- Episode seed list (Step 3, action 3.7) ---------------------------
    seeds = _load_test_seeds(repo_root, cfg)
    if args.num_episodes is not None:
        seeds = seeds[: int(args.num_episodes)]
    n_eval = len(seeds)
    logger.info(f"config={args.config}  num_episodes={n_eval}  "
                f"(from {cfg.eval.episodes_file})")

    set_seed(int(cfg.seed))
    device = get_device()
    head_type = cfg.head.type
    K = int(cfg.dataset.n_way)

    # --- W&B run (Step 3) -------------------------------------------------
    wb = _build_wandb_run(cfg, args, head_type)
    if not wb.disabled:
        logger.info(f"wandb run: {wb.run_name}  url={wb.url}")

    # --- Load shared data once --------------------------------------------
    dataset = build_dataset(dict(cfg.dataset))
    svhn_x = get_svhn_ood(
        data_root=cfg.ood.data_root,
        image_size=int(cfg.dataset.image_size),
        num_samples=int(cfg.ood.num_samples),
        seed=int(cfg.ood.seed),
    )

    # Build a single backbone instance to extract features. The trained
    # adapter+head live in a per-episode model; the backbone is frozen and
    # identical across episodes so feature extraction is amortised here.
    shared_model = build_model(cfg).to(device)
    svhn_feats = _extract_features(shared_model.backbone, svhn_x, device)
    logger.info(f"cached SVHN features: {tuple(svhn_feats.shape)}")

    # --- Per-episode loop --------------------------------------------------
    per_ep_acc, per_ep_ece, per_ep_brier, per_ep_auroc = [], [], [], []
    pooled_probs, pooled_targets = [], []
    last_id_scores: np.ndarray | None = None
    last_ood_scores: np.ndarray | None = None

    for i, ep_seed in enumerate(seeds):
        set_seed(int(cfg.seed) + int(ep_seed))
        # Fresh model per episode -> trained on this episode's support set.
        model = build_model(cfg).to(device)
        loss_spec = dict(cfg.loss)
        loss_spec["type"] = "evidential" if head_type == "evidential" else "cross_entropy"
        loss_fn = build_loss(loss_spec)

        support_x, support_y, query_x, query_y = sample_episode(
            dataset=dataset,
            class_ids=list(cfg.dataset.class_ids),
            n_way=K, k_shot=int(cfg.dataset.k_shot),
            q_query=int(cfg.dataset.q_query),
            seed=int(ep_seed),
        )
        # Cache features once per episode; adapter+head train on (B, D) tensors.
        support_feats = _extract_features(model.backbone, support_x, device).to(device)
        query_feats = _extract_features(model.backbone, query_x, device)
        support_y_d = support_y.to(device)

        train_one_episode(
            model=model,
            support_x=support_feats, support_y=support_y_d,
            loss_fn=loss_fn, num_classes=K, head_type=head_type,
            lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
            num_steps=int(cfg.train.num_steps),
            log_every=10**9, logger=None,
            wandb_run=None,  # don't pollute the eval run with per-step training logs
        )

        # ID metrics
        probs, vac = _predict_from_features(model, query_feats, head_type, device, num_classes=K)
        acc = accuracy(probs, query_y)
        ece = expected_calibration_error(probs, query_y, num_bins=int(cfg.eval.ece_bins))
        bri = brier_score(probs, query_y, K)

        # OOD AUROC: ID score = 1 - vacuity
        _, ood_vac = _predict_from_features(model, svhn_feats, head_type, device, num_classes=K)
        id_scores = (1.0 - vac).numpy()
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
        # Step 3 action 3.3: per-episode W&B log
        wb.log({
            "eval/episode": i,
            "eval/episode_seed": int(ep_seed),
            "eval/accuracy": float(acc),
            "eval/ECE": float(ece),
            "eval/Brier": float(bri),
            "eval/OOD_AUROC": float(auroc),
            # rolling means make the W&B dashboard sortable mid-run
            "eval/accuracy_running_mean": float(np.mean(per_ep_acc)),
            "eval/ECE_running_mean": float(np.mean(per_ep_ece)),
            "eval/AUROC_running_mean": float(np.mean(per_ep_auroc)),
        })

    pooled_probs = torch.cat(pooled_probs, dim=0)
    pooled_targets = torch.cat(pooled_targets, dim=0)
    pooled_ece = expected_calibration_error(
        pooled_probs, pooled_targets, num_bins=int(cfg.eval.ece_bins),
    )

    # --- Summary (sort_keys -> byte-identical on repeated runs) -----------
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
    }

    # --- Final artifacts (Step 3 action 3.4) -------------------------------
    out_dir = Path(cfg.output.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.results_suffix}_{cfg.adapter.type}_{head_type}"

    metrics_path = out_dir / f"{tag}_metrics.json"
    rel_path     = out_dir / f"{tag}_reliability.png"
    hist_path    = out_dir / f"{tag}_ood_histogram.png"
    cm_path      = out_dir / f"{tag}_confusion_matrix.png"

    reliability_diagram(
        pooled_probs, pooled_targets, rel_path,
        num_bins=int(cfg.eval.ece_bins),
        title=f"Reliability ({head_type}, {cfg.adapter.type})  ECE={pooled_ece:.3f}",
    )
    if last_id_scores is not None and last_ood_scores is not None:
        ood_histogram(
            last_id_scores, last_ood_scores, hist_path,
            title=f"ID vs SVHN ({head_type})  AUROC={np.mean(per_ep_auroc):.3f}",
        )
    confusion_matrix(
        pooled_probs, pooled_targets, cm_path, num_classes=K,
        title=f"Confusion ({head_type})  acc={np.mean(per_ep_acc):.3f}",
    )

    # Step 3 exit criterion: byte-identical metrics.json on repeated runs.
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logger.info(f"saved metrics: {metrics_path}")

    # Log final artifacts to W&B
    wb.update_summary({
        "final/accuracy_mean": summary["accuracy_mean"],
        "final/accuracy_ci95": summary["accuracy_ci95"],
        "final/ece_pooled":    summary["ece_pooled"],
        "final/ece_per_episode_mean": summary["ece_per_episode_mean"],
        "final/brier_mean":    summary["brier_mean"],
        "final/ood_auroc_mean": summary["ood_auroc_mean"],
        "final/num_episodes":  summary["num_episodes"],
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
