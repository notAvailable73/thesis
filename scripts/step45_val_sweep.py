"""Step 4.5 / W2 — VAL-only R-EDL hyperparameter sweep (in-process).

Trains a SHORT evidential model per (kl_weight_max, use_variance, prior_per_class)
combo, evaluates on the FROZEN VAL episodes (val split + configs/val_episodes.yaml
seeds), and ranks by VAL calibration (pooled ECE). Writes the winning combo into
a swept config so the subsequent FULL train+test run uses a VAL-selected operating
point instead of hand-picked defaults.

Selection is on VAL only — the 600 test seeds (configs/test_episodes.yaml) are
never touched here, so this cannot leak into the reported result.

Runs everything IN-PROCESS (no subprocess) so Colab cannot swallow the per-combo
output the way the old inline sweep cell did.

Reasoning (thesis instructions): closes the Step 4.5 §5 caveat that the retuned
R-EDL hyperparameters were reasoned defaults, not an empirically-swept choice.
The short-epoch ranking is a proxy for the full run; the winner is then trained
to convergence separately (standard sweep-then-refit protocol) — a documented
approximation, not a claim that 8-epoch ranking == 30-epoch ranking.
"""
from __future__ import annotations
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.utils import set_seed, get_device, load_config, get_logger
from src.datasets import get_cifar_fs, get_svhn_ood, EpisodicIterableDataset
from src.models import build_model
from src.trainers import EpisodicTrainer
from src.evaluators import evaluate_episodic


def _extract_features(backbone, x, device, batch_size=64):
    backbone.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            chunks.append(backbone(x[i:i + batch_size].to(device)).cpu())
    return torch.cat(chunks, dim=0)


def _train_and_val(cfg, epochs, device, svhn_feats, logger=None):
    """Train `epochs` epochs, then evaluate on the VAL episodes. Returns a
    dict of VAL metrics (ECE / acc / far-OOD AUROC). No test data touched.

    `logger` (if given) is passed to the trainer so each combo prints per-epoch
    progress — otherwise a long combo looks like a hang on Colab."""
    interp = cfg.head.get("interpretation", "evidential")
    n_way = int(cfg.dataset.n_way)
    k_shot = int(cfg.dataset.k_shot)
    q_query = int(cfg.dataset.q_query)
    eps = int(cfg.trainer.episodes_per_epoch)
    veps = int(cfg.trainer.val_episodes_per_epoch)
    tso = int(cfg.trainer.train_seed_offset)

    train_split = get_cifar_fs(data_root=cfg.dataset.data_root,
                               image_size=int(cfg.dataset.image_size), split="train")
    val_split = get_cifar_fs(data_root=cfg.dataset.data_root,
                             image_size=int(cfg.dataset.image_size), split="val")
    repo = Path(__file__).resolve().parents[1]
    val_seeds = yaml.safe_load(open(repo / cfg.eval.val_episodes_file))["seeds"]
    vso = int(val_seeds[0])

    def train_factory(epoch):
        return EpisodicIterableDataset(
            train_split, n_way=n_way, k_shot=k_shot, q_query=q_query,
            num_episodes=eps, seed_offset=tso + (epoch - 1) * eps)

    def val_factory(epoch):
        return EpisodicIterableDataset(
            val_split, n_way=n_way, k_shot=k_shot, q_query=q_query,
            num_episodes=veps, seed_offset=vso)

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=float(cfg.train.lr),
                           weight_decay=float(cfg.train.weight_decay))
    trainer = EpisodicTrainer(
        model=model, optimizer=opt, num_classes=n_way, num_epochs=epochs,
        episodes_per_epoch=eps, val_episodes_per_epoch=veps,
        early_stop_patience=int(cfg.trainer.early_stop_patience),
        interpretation=interp,
        kl_weight_max=float(cfg.loss.kl_weight_max),
        kl_anneal_steps=int(cfg.loss.kl_anneal_steps),
        ece_bins=int(cfg.eval.ece_bins), logger=logger, wandb_run=None,
        device=device, collapse_threshold=float(cfg.trainer.collapse_threshold),
        evid_prior_per_class=float(cfg.loss.get("prior_per_class", 1.0)),
        evid_use_variance=bool(cfg.loss.get("use_variance", True)))
    trainer.fit(train_factory, val_factory)

    val_iter = EpisodicIterableDataset(
        val_split, n_way=n_way, k_shot=k_shot, q_query=q_query,
        num_episodes=veps, seed_offset=vso)
    res = evaluate_episodic(
        model, val_iter, {"svhn_far": svhn_feats}, num_classes=n_way,
        interpretation=interp, ece_bins=int(cfg.eval.ece_bins),
        temperature=None, prior_per_class=float(cfg.loss.get("prior_per_class", 1.0)),
        device=device)
    s = res["summary"]
    return {"val_ece": float(s["ece_pooled"]),
            "val_acc": float(s["accuracy_mean"]),
            "val_svhn_auroc": float(s.get("ood_auroc__svhn_far__vacuity", float("nan")))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/exp_phase2_evidential_retuned.yaml")
    ap.add_argument("--out-config", default="configs/exp_phase2_evidential_swept.yaml")
    ap.add_argument("--epochs", type=int, default=8,
                    help="Short training budget per combo (proxy for the full run).")
    ap.add_argument("--kl", type=float, nargs="+", default=[0.05, 0.1, 0.25])
    ap.add_argument("--use-variance", type=int, nargs="+", default=[0, 1],
                    help="0=drop variance term (R-EDL), 1=keep (Sensoy).")
    ap.add_argument("--prior", type=float, nargs="+", default=[1.0])
    ap.add_argument("--acc-tol", type=float, default=0.03,
                    help="Winner must be within this val-acc of the best combo.")
    args = ap.parse_args()

    logger = get_logger("bpeft.sweep")
    device = get_device()
    base = load_config(args.base_config)

    # Far-OOD (SVHN) backbone features: config-independent (frozen backbone),
    # so extract ONCE and reuse for every combo's val eval.
    set_seed(int(base.seed))
    m0 = build_model(base).to(device)
    svhn_x = get_svhn_ood(data_root=base.ood.data_root,
                          image_size=int(base.dataset.image_size),
                          num_samples=int(base.ood.num_samples), seed=int(base.ood.seed))
    svhn_feats = _extract_features(m0.backbone, svhn_x, device)
    del m0

    results = []
    for kl, uv, pr in itertools.product(args.kl, args.use_variance, args.prior):
        set_seed(int(base.seed))  # same init per combo -> fair comparison
        cfg = load_config(args.base_config)
        cfg.loss.kl_weight_max = float(kl)
        cfg.loss.use_variance = bool(uv)
        cfg.loss.prior_per_class = float(pr)
        logger.info(f"=== combo: kl={kl} use_variance={bool(uv)} prior={pr} "
                    f"({args.epochs} epochs) ===")
        m = _train_and_val(cfg, args.epochs, device, svhn_feats, logger=logger)
        m.update({"kl": float(kl), "use_variance": bool(uv), "prior": float(pr)})
        results.append(m)
        logger.info(f"kl={kl} var={bool(uv)} prior={pr} -> "
                    f"val_ECE={m['val_ece']:.4f} val_acc={m['val_acc']:.4f} "
                    f"val_svhn_AUROC={m['val_svhn_auroc']:.4f}")

    best_acc = max(r["val_acc"] for r in results)
    eligible = [r for r in results if r["val_acc"] >= best_acc - args.acc_tol]
    winner = min(eligible, key=lambda r: r["val_ece"])

    print("\n=== VAL sweep (selection on VAL only; test never touched) ===")
    print(f"{'kl':>6}{'var':>7}{'prior':>7}{'val_ECE':>10}{'val_acc':>10}{'val_AUROC':>11}")
    for r in sorted(results, key=lambda r: r["val_ece"]):
        mark = "  <== winner (lowest val ECE within acc tol)" if r is winner else ""
        print(f"{r['kl']:>6}{str(r['use_variance']):>7}{r['prior']:>7}"
              f"{r['val_ece']:>10.4f}{r['val_acc']:>10.4f}"
              f"{r['val_svhn_auroc']:>11.4f}{mark}")

    out = {
        "extends": Path(args.base_config).name,
        "_note": "Step 4.5 W2 — VAL-selected R-EDL config (scripts/step45_val_sweep.py). "
                 "Selected by lowest VAL pooled-ECE within acc tolerance; test untouched.",
        "loss": {"kl_weight_max": winner["kl"],
                 "use_variance": winner["use_variance"],
                 "prior_per_class": winner["prior"]},
    }
    with open(args.out_config, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"\nwrote {args.out_config}: kl_weight_max={winner['kl']} "
          f"use_variance={winner['use_variance']} prior_per_class={winner['prior']} "
          f"(val_ECE={winner['val_ece']:.4f}, val_acc={winner['val_acc']:.4f})")


if __name__ == "__main__":
    main()
