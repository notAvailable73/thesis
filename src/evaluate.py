import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import CFG
from .utils import set_seed, get_device, count_trainable_params
from .data import get_svhn_ood
from .model import BPEFTModel
from .metrics import accuracy, expected_calibration_error, brier_score, ood_auroc


# 4-way comparison: (adapter, mode) pairs
MODEL_KEYS = [
    ("bottleneck", "evidential"),
    ("bottleneck", "softmax"),
    ("lora",       "evidential"),
    ("lora",       "softmax"),
]

COLORS = {
    ("bottleneck", "evidential"): "steelblue",
    ("bottleneck", "softmax"):    "tomato",
    ("lora",       "evidential"): "mediumseagreen",
    ("lora",       "softmax"):    "darkorange",
}


def label(adapter, mode):
    return f"{adapter}-{mode}"


def get_probs_and_vacuity(model, x, mode, device, batch_size=50):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(device)
            all_outputs.append(model(batch).cpu())
    output = torch.cat(all_outputs, dim=0)

    if mode == "evidential":
        alpha = output + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / S
        vacuity = (CFG.num_classes / S).squeeze(-1)
    else:
        probs = torch.softmax(output, dim=-1)
        vacuity = 1.0 - probs.max(dim=-1).values  # proxy uncertainty

    return probs, vacuity


def plot_reliability_diagram(results, targets, save_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for key, res in results.items():
        probs = res["query_probs"]
        confs, preds = probs.max(dim=-1)
        correct = (preds == targets).float().numpy()
        confs = confs.numpy()

        bin_edges = np.linspace(0, 1, 11)
        bin_acc, bin_conf = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confs > lo) & (confs <= hi)
            if mask.sum() == 0:
                continue
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confs[mask].mean())

        ece = res["ece"]
        adapter, mode = key
        ax.plot(bin_conf, bin_acc, "o-", color=COLORS[key],
                label=f"{label(adapter, mode)} (ECE={ece:.3f})")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram (4-way comparison)")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_ood_histogram(results, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, key in zip(axes, MODEL_KEYS):
        if key not in results:
            continue
        res = results[key]
        adapter, mode = key
        id_u  = res["id_vacuity"].numpy()
        ood_u = res["ood_vacuity"].numpy()
        auroc = res["ood_auroc"]

        base = COLORS[key]
        ax.hist(id_u,  bins=30, alpha=0.7, color=base,        label="In-dist (CIFAR-FS)")
        ax.hist(ood_u, bins=30, alpha=0.5, color="lightgray", label="OOD (SVHN)",
                edgecolor=base)
        ax.set_title(f"{label(adapter, mode)}  (AUROC={auroc:.3f})")
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle("OOD Detection: Uncertainty Distributions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(results, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for key, res in results.items():
        hist = res["train_history"]
        steps = hist["step"]
        adapter, mode = key
        ax1.plot(steps, hist["loss"], color=COLORS[key], label=label(adapter, mode))
        ax2.plot(steps, hist["acc"],  color=COLORS[key], label=label(adapter, mode))

    ax1.set_title("Training Loss"); ax1.set_xlabel("Step"); ax1.legend()
    ax2.set_title("Training Accuracy"); ax2.set_xlabel("Step"); ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def evaluate():
    set_seed(CFG.seed)
    device = get_device()
    os.makedirs(CFG.results_dir, exist_ok=True)

    svhn_x = get_svhn_ood(CFG.data_root, CFG.image_size, CFG.ood_num_samples, CFG.seed)

    results = {}
    reference_query_y = None

    for key in MODEL_KEYS:
        adapter, mode = key
        ckpt_path = os.path.join(CFG.checkpoint_dir, f"model_{adapter}_{mode}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] {ckpt_path} not found")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")

        model = BPEFTModel(
            num_classes=CFG.num_classes,
            feature_dim=CFG.feature_dim,
            adapter_rank=CFG.adapter_rank,
            mode=mode,
            adapter_type=adapter,
            lora_alpha=CFG.lora_alpha,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        episode = ckpt["episode"]
        query_x = episode["query_x"]
        query_y = episode["query_y"]
        if reference_query_y is None:
            reference_query_y = query_y

        probs, vacuity = get_probs_and_vacuity(model, query_x, mode, device)

        acc   = accuracy(probs, query_y)
        ece   = expected_calibration_error(probs, query_y)
        brier = brier_score(probs, query_y, CFG.num_classes)

        _, ood_vacuity = get_probs_and_vacuity(model, svhn_x, mode, device)
        id_scores  = (1.0 - vacuity).numpy()
        ood_scores = (1.0 - ood_vacuity).numpy()
        auroc = ood_auroc(id_scores, ood_scores)

        n_params = count_trainable_params(model)

        results[key] = {
            "accuracy": acc,
            "ece": ece,
            "brier": brier,
            "ood_auroc": auroc,
            "n_params": n_params,
            "query_probs": probs,
            "id_vacuity": vacuity,
            "ood_vacuity": ood_vacuity,
            "train_history": ckpt["train_history"],
        }

        print(f"\n[{label(adapter, mode).upper()}]")
        print(f"  Accuracy : {acc:.3f}")
        print(f"  ECE      : {ece:.3f}")
        print(f"  Brier    : {brier:.3f}")
        print(f"  OOD AUROC: {auroc:.3f}")
        print(f"  Params   : {n_params:,}")

    if not results:
        print("No checkpoints found. Train first.")
        return

    # Save metrics.json
    metrics_out = {
        label(a, m): {
            "accuracy": results[(a, m)]["accuracy"],
            "ece":      results[(a, m)]["ece"],
            "brier":    results[(a, m)]["brier"],
            "ood_auroc":results[(a, m)]["ood_auroc"],
            "n_params": results[(a, m)]["n_params"],
        }
        for (a, m) in results
    }

    json_path = os.path.join(CFG.results_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Pretty summary table
    print("\n" + "=" * 72)
    print(f"{'Model':<24} {'Acc':>6} {'ECE':>6} {'Brier':>6} {'OOD AUROC':>10} {'Params':>10}")
    print("-" * 72)
    for (a, m) in results:
        r = results[(a, m)]
        print(f"{label(a, m):<24} {r['accuracy']:>6.3f} {r['ece']:>6.3f} "
              f"{r['brier']:>6.3f} {r['ood_auroc']:>10.3f} {r['n_params']:>10,}")
    print("=" * 72)

    # Plots
    plot_reliability_diagram(results, reference_query_y,
        os.path.join(CFG.results_dir, "reliability_plot.png"))
    plot_ood_histogram(results,
        os.path.join(CFG.results_dir, "ood_histogram.png"))
    plot_training_curves(results,
        os.path.join(CFG.results_dir, "training_curve.png"))

    print("\nDone.")


if __name__ == "__main__":
    evaluate()
