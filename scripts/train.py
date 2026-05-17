"""Train a single B-PEFT few-shot episode from a YAML config.

Usage:
    python scripts/train.py --config configs/exp_step1.yaml

Step 3 (action 3.1 + 3.6):
  - set_seed() is the first thing the script does after parsing args.
  - W&B run is opened (project from cfg.wandb.project, name from
    make_run_name(cfg)) and passed into the trainer so per-step
    loss/support_acc/lr/kl_weight are logged.
  - --wandb-mode CLI override lets the notebook switch online/offline/disabled
    per run without editing YAML.

Produces a checkpoint under cfg.output.checkpoint_dir keyed by adapter+head.
"""
import argparse
import os
import sys
from pathlib import Path

# Make `src` importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils import (
    set_seed, get_device, count_trainable_params, load_config, get_logger,
    WandbRun, make_run_name,
)
from src.datasets import build_dataset, sample_episode
from src.models import build_model
from src.losses import build_loss
from src.trainers import train_one_episode


def _build_wandb_run(cfg, args, head_type) -> WandbRun:
    """Open a W&B run for this training job. Honours --wandb-mode CLI flag."""
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
        run_name=make_run_name(cfg),
        config={
            "config_path": str(Path(args.config).resolve()),
            "seed": int(cfg.seed),
            "head": dict(cfg.head),
            "adapter": dict(cfg.adapter),
            "loss": dict(cfg.loss),
            "dataset": {k: v for k, v in dict(cfg.dataset).items()
                        if k != "class_ids"},
            "train": dict(cfg.train),
        },
        mode=mode, disabled=disabled, group=group,
        tags=tags + [f"head:{head_type}", f"adapter:{cfg.adapter.type}", "step3"],
        job_type="train",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--seed-override", type=int, default=None,
                        help="Override cfg.seed (used by run_grid.sh)")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                        default=None, help="Override cfg.wandb.mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed_override is not None:
        cfg.seed = args.seed_override
    logger = get_logger("bpeft.train")
    logger.info(f"config: {args.config}  seed: {cfg.seed}")

    set_seed(int(cfg.seed))
    device = get_device()
    head_type = cfg.head.type

    # --- W&B run (Step 3) -------------------------------------------------
    wb = _build_wandb_run(cfg, args, head_type)
    if not wb.disabled:
        logger.info(f"wandb run: {wb.run_name}  ({wb.mode})  url={wb.url}")
    else:
        logger.info("wandb: disabled (no-op logger)")

    # --- Data --------------------------------------------------------------
    dataset = build_dataset(dict(cfg.dataset))
    support_x, support_y, query_x, query_y = sample_episode(
        dataset=dataset,
        class_ids=list(cfg.dataset.class_ids),
        n_way=int(cfg.dataset.n_way),
        k_shot=int(cfg.dataset.k_shot),
        q_query=int(cfg.dataset.q_query),
        seed=int(cfg.seed),
    )
    support_x = support_x.to(device)
    support_y = support_y.to(device)

    # --- Model -------------------------------------------------------------
    model = build_model(cfg).to(device)
    n_params = count_trainable_params(model)
    logger.info(f"trainable params: {n_params:,}")
    wb.update_summary({"n_params": int(n_params)})

    # Pre-compute backbone features once. The backbone is frozen so this is
    # equivalent to calling model(support_x) every step but ~50x cheaper.
    with torch.no_grad():
        support_feats = model.backbone(support_x)

    # --- Loss --------------------------------------------------------------
    loss_spec = dict(cfg.loss)
    loss_spec["type"] = "evidential" if head_type == "evidential" else "cross_entropy"
    loss_fn = build_loss(loss_spec)

    # --- Train -------------------------------------------------------------
    history = train_one_episode(
        model=model,
        support_x=support_feats,
        support_y=support_y,
        loss_fn=loss_fn,
        num_classes=int(cfg.dataset.n_way),
        head_type=head_type,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        num_steps=int(cfg.train.num_steps),
        log_every=int(cfg.train.log_every),
        logger=logger,
        wandb_run=wb,
        kl_weight_max=float(cfg.loss.kl_weight_max) if head_type == "evidential" else None,
        kl_anneal_steps=int(cfg.loss.kl_anneal_steps) if head_type == "evidential" else None,
    )

    # --- Save --------------------------------------------------------------
    ckpt_dir = Path(cfg.output.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.adapter.type}_{head_type}_seed{cfg.seed}"
    ckpt_path = ckpt_dir / f"model_{tag}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config_path": str(Path(args.config).resolve()),
        "head_type": head_type,
        "adapter_type": cfg.adapter.type,
        "train_history": history,
        "episode": {
            "support_x": support_x.cpu(),
            "support_y": support_y.cpu(),
            "query_x": query_x,
            "query_y": query_y,
        },
    }, ckpt_path)
    logger.info(f"saved checkpoint: {ckpt_path}")

    final_acc = history["support_acc"][-1]
    final_loss = history["loss"][-1]
    logger.info(f"final support accuracy: {final_acc:.3f}")
    wb.update_summary({
        "train/final_support_acc": float(final_acc),
        "train/final_loss": float(final_loss),
    })
    wb.finish()


if __name__ == "__main__":
    main()
