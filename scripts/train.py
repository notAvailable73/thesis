"""Train a B-PEFT model from a YAML config.

Two modes, dispatched on cfg.trainer.type:

  trainer.type: single_episode    (Step 1-3 legacy)
      One episode of (n_way, k_shot, q_query), 200 inner training steps,
      LinearHead / EvidentialHead. Used by configs/exp_step1.yaml +
      configs/exp_step1_softmax.yaml — the Step 3 reproduction tests
      depend on this path being identical.

  trainer.type: episodic          (Step 4 / Phase 2)
      Episodic meta-training: per epoch we sample `episodes_per_epoch`
      independent episodes from the Bertinetto TRAIN split (64 classes),
      apply one outer gradient step per episode, then validate on the
      VAL split (16 classes) using configs/val_episodes.yaml. Early stop
      on val accuracy plateau. The adapter is the trainable PEFT
      module; the head is a parameter-free PrototypeHead.

Usage:
    python scripts/train.py --config configs/exp_step1.yaml          # legacy
    python scripts/train.py --config configs/exp_phase2_evidential.yaml
    python scripts/train.py --config configs/exp_phase2_softmax.yaml
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.utils import (
    set_seed, get_device, count_trainable_params, load_config, get_logger,
    WandbRun, make_run_name,
)
from src.datasets import (
    build_dataset, sample_episode, get_cifar_fs, EpisodicIterableDataset,
)
from src.models import build_model
from src.losses import build_loss
from src.trainers import train_one_episode, EpisodicTrainer


# =====================================================================
# Shared helpers
# =====================================================================
def _build_wandb_run(cfg, args, head_type, job_type: str = "train",
                    extra_tags: list[str] | None = None) -> WandbRun:
    """Open a W&B run for this training job. Honours --wandb-mode CLI flag."""
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
        run_name=make_run_name(cfg),
        config={
            "config_path": str(Path(args.config).resolve()),
            "seed": int(cfg.seed),
            "head": dict(cfg.head),
            "adapter": dict(cfg.adapter),
            "loss": dict(cfg.loss),
            "trainer": dict(cfg.trainer),
            "dataset": {k: v for k, v in dict(cfg.dataset).items()
                        if k != "class_ids"},
            "train": dict(cfg.train),
        },
        mode=mode, disabled=disabled, group=group,
        tags=tags + [f"head:{head_type}", f"adapter:{cfg.adapter.type}"],
        job_type=job_type,
    )


def _head_descriptor(cfg) -> str:
    """Filesystem-safe head identifier.

    Step 1-3: head.type is sufficient (softmax / evidential).
    Step 4+:  prototype configs need head.interpretation appended,
              otherwise the two Phase 2 configs collide on disk.
    """
    head_type = cfg.head.type
    if head_type == "prototype":
        return f"prototype-{cfg.head.get('interpretation', 'evidential')}"
    return head_type


def _checkpoint_tag(cfg) -> str:
    """Filesystem-safe tag identifying this training run."""
    return f"{cfg.adapter.type}_{_head_descriptor(cfg)}_seed{cfg.seed}"


# =====================================================================
# Step 1-3 legacy path (trainer.type: single_episode)
# =====================================================================
def _train_single_episode(cfg, args, logger, device, wb) -> None:
    """Existing Step 1-3 single-episode training flow. Unchanged on
    purpose so the Step 3 reproduction tests stay byte-identical."""
    head_type = cfg.head.type

    dataset = build_dataset(dict(cfg.dataset))
    support_x, support_y, query_x, query_y = sample_episode(
        dataset=dataset,
        class_ids=list(cfg.dataset.class_ids) if cfg.dataset.get("class_ids") else None,
        n_way=int(cfg.dataset.n_way),
        k_shot=int(cfg.dataset.k_shot),
        q_query=int(cfg.dataset.q_query),
        seed=int(cfg.seed),
    )
    support_x = support_x.to(device)
    support_y = support_y.to(device)

    model = build_model(cfg).to(device)
    n_params = count_trainable_params(model)
    logger.info(f"trainable params: {n_params:,}")
    wb.update_summary({"n_params": int(n_params)})

    with torch.no_grad():
        support_feats = model.backbone(support_x)

    loss_spec = dict(cfg.loss)
    loss_spec["type"] = "evidential" if head_type == "evidential" else "cross_entropy"
    loss_fn = build_loss(loss_spec)

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

    ckpt_dir = Path(cfg.output.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = _checkpoint_tag(cfg)
    ckpt_path = ckpt_dir / f"model_{tag}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config_path": str(Path(args.config).resolve()),
        "head_type": head_type,
        "adapter_type": cfg.adapter.type,
        "trainer_type": "single_episode",
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


# =====================================================================
# Step 4 / Phase 2 path (trainer.type: episodic)
# =====================================================================
def _train_episodic(cfg, args, logger, device, wb) -> None:
    """Episodic meta-training with prototype head."""
    interp = cfg.head.get("interpretation", "evidential")
    if interp not in ("softmax", "evidential"):
        raise ValueError(
            f"head.interpretation must be 'softmax' or 'evidential' for "
            f"trainer.type=episodic; got {interp!r}"
        )
    if cfg.head.type != "prototype":
        raise ValueError(
            f"trainer.type=episodic requires head.type=prototype; "
            f"got head.type={cfg.head.type!r}"
        )

    # --- Datasets: train + val splits ---------------------------------
    train_split = get_cifar_fs(
        data_root=cfg.dataset.data_root,
        image_size=int(cfg.dataset.image_size),
        split="train",
    )
    val_split = get_cifar_fs(
        data_root=cfg.dataset.data_root,
        image_size=int(cfg.dataset.image_size),
        split="val",
    )
    n_way   = int(cfg.dataset.n_way)
    k_shot  = int(cfg.dataset.k_shot)
    q_query = int(cfg.dataset.q_query)
    eps_per_epoch    = int(cfg.trainer.episodes_per_epoch)
    val_eps_per_epoch = int(cfg.trainer.val_episodes_per_epoch)
    train_seed_offset = int(cfg.trainer.train_seed_offset)

    # Frozen val seeds (configs/val_episodes.yaml).
    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / cfg.eval.val_episodes_file) as f:
        val_eps_spec = yaml.safe_load(f)
    val_seeds = list(val_eps_spec["seeds"])
    if len(val_seeds) < val_eps_per_epoch:
        raise ValueError(
            f"val_episodes.yaml has {len(val_seeds)} seeds but "
            f"val_episodes_per_epoch={val_eps_per_epoch}"
        )
    val_seed_offset = int(val_seeds[0])

    def train_iter_factory(epoch: int):
        # Different seed_offset each epoch so train episodes don't repeat
        # across epochs; deterministic given (TRAIN_BASE, epoch, eps_per_epoch).
        offset = train_seed_offset + (epoch - 1) * eps_per_epoch
        return EpisodicIterableDataset(
            train_split, n_way=n_way, k_shot=k_shot, q_query=q_query,
            num_episodes=eps_per_epoch, seed_offset=offset,
        )

    def val_iter_factory(epoch: int):
        # Fixed seed offset so the same val episodes are used every epoch.
        # That makes the early-stop comparison paired.
        return EpisodicIterableDataset(
            val_split, n_way=n_way, k_shot=k_shot, q_query=q_query,
            num_episodes=val_eps_per_epoch, seed_offset=val_seed_offset,
        )

    # --- Model + optimiser --------------------------------------------
    model = build_model(cfg).to(device)
    n_params = count_trainable_params(model)
    logger.info(f"trainable params: {n_params:,}")
    wb.update_summary({"n_params": int(n_params)})

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    trainer = EpisodicTrainer(
        model=model,
        optimizer=optimizer,
        num_classes=n_way,
        num_epochs=int(cfg.trainer.num_epochs),
        episodes_per_epoch=eps_per_epoch,
        val_episodes_per_epoch=val_eps_per_epoch,
        early_stop_patience=int(cfg.trainer.early_stop_patience),
        interpretation=interp,
        kl_weight_max=(float(cfg.loss.kl_weight_max)
                        if interp == "evidential" else None),
        kl_anneal_steps=(int(cfg.loss.kl_anneal_steps)
                         if interp == "evidential" else None),
        ece_bins=int(cfg.eval.ece_bins),
        logger=logger,
        wandb_run=wb,
        device=device,
        collapse_threshold=float(cfg.trainer.collapse_threshold),
        # R-EDL knobs (Step 4.5 / W2); defaults reproduce the Sensoy loss.
        evid_prior_per_class=float(cfg.loss.get("prior_per_class", 1.0)),
        evid_use_variance=bool(cfg.loss.get("use_variance", True)),
    )
    result = trainer.fit(train_iter_factory, val_iter_factory)

    # --- Save ----------------------------------------------------------
    ckpt_dir = Path(cfg.output.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = _checkpoint_tag(cfg)
    ckpt_path = ckpt_dir / f"model_phase2_{tag}.pt"
    history = result["history"]
    torch.save({
        "state_dict":   model.state_dict(),
        "config_path":  str(Path(args.config).resolve()),
        "head_type":    cfg.head.type,
        "adapter_type": cfg.adapter.type,
        "trainer_type": "episodic",
        "interpretation": interp,
        "best_val_acc":   result["best_val_acc"],
        "best_val_epoch": result["best_val_epoch"],
        "train_history": {
            "epoch":             list(history.epoch),
            "train_loss":        list(history.train_loss),
            "train_acc":         list(history.train_acc),
            "val_loss":          list(history.val_loss),
            "val_acc":           list(history.val_acc),
            "kl_weight_at_end":  list(history.kl_weight_at_end),
            "mean_evidence":     list(history.mean_evidence),
            "adapter_grad_norm": list(history.adapter_grad_norm),
        },
    }, ckpt_path)
    logger.info(
        f"saved checkpoint: {ckpt_path}  "
        f"best_val_acc={result['best_val_acc']:.3f} at "
        f"epoch {result['best_val_epoch']}"
    )
    wb.update_summary({
        "train/best_val_acc":   float(result["best_val_acc"]),
        "train/best_val_epoch": int(result["best_val_epoch"]),
        "train/total_epochs":   int(history.epoch[-1] if history.epoch else 0),
    })


# =====================================================================
# main
# =====================================================================
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
    trainer_type = cfg.get("trainer", {}).get("type", "single_episode") if isinstance(cfg, dict) else "single_episode"
    logger.info(f"config: {args.config}  seed: {cfg.seed}  trainer.type: {trainer_type}")

    set_seed(int(cfg.seed))
    device = get_device()
    head_type = cfg.head.type

    extra_tags = [f"trainer:{trainer_type}",
                  f"phase:{'2' if trainer_type == 'episodic' else '1'}"]
    wb = _build_wandb_run(cfg, args, head_type, extra_tags=extra_tags)
    if not wb.disabled:
        logger.info(f"wandb run: {wb.run_name}  ({wb.mode})  url={wb.url}")
    else:
        logger.info("wandb: disabled (no-op logger)")

    if trainer_type == "single_episode":
        _train_single_episode(cfg, args, logger, device, wb)
    elif trainer_type == "episodic":
        _train_episodic(cfg, args, logger, device, wb)
    else:
        raise ValueError(
            f"Unknown trainer.type: {trainer_type!r}; expected "
            f"'single_episode' or 'episodic'."
        )

    wb.finish()


if __name__ == "__main__":
    main()
