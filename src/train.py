import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import CFG
from .utils import set_seed, get_device, count_trainable_params
from .data import get_cifar100_test, sample_episode
from .model import BPEFTModel
from .losses import evidential_mse_loss, softmax_ce_loss


def train(mode: str = "evidential") -> dict:
    set_seed(CFG.seed)
    device = get_device()
    print(f"\n{'='*50}")
    print(f"Training mode: {mode.upper()}  |  device: {device}")
    print(f"{'='*50}")

    # ── Data ──────────────────────────────────────────
    dataset = get_cifar100_test(CFG.data_root, CFG.image_size)
    support_x, support_y, query_x, query_y = sample_episode(
        dataset, CFG.test_class_ids, CFG.num_classes,
        CFG.shots, CFG.query_per_class, CFG.seed
    )
    support_x = support_x.to(device)
    support_y = support_y.to(device)

    # ── Model ─────────────────────────────────────────
    model = BPEFTModel(
        num_classes=CFG.num_classes,
        feature_dim=CFG.feature_dim,
        adapter_rank=CFG.adapter_rank,
        mode=mode
    ).to(device)
    print(f"Trainable params: {count_trainable_params(model):,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )

    # ── One-hot targets for evidential loss ───────────
    target_onehot = F.one_hot(support_y, CFG.num_classes).float()

    # ── Training loop ─────────────────────────────────
    history = {"loss": [], "acc": [], "step": []}

    model.train()
    model.backbone.eval()  # keep BatchNorm in eval mode

    for step in range(1, CFG.num_steps + 1):
        optimizer.zero_grad()
        output = model(support_x)

        kl_weight = min(1.0, step / CFG.kl_anneal_steps)

        if mode == "evidential":
            loss = evidential_mse_loss(output, target_onehot, CFG.num_classes, kl_weight)
            with torch.no_grad():
                alpha = output + 1.0
                probs = alpha / alpha.sum(dim=-1, keepdim=True)
        else:
            loss = softmax_ce_loss(output, support_y)
            with torch.no_grad():
                probs = torch.softmax(output, dim=-1)

        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            acc = (probs.argmax(dim=-1) == support_y).float().mean().item()
            history["loss"].append(loss.item())
            history["acc"].append(acc)
            history["step"].append(step)
            print(f"  step {step:3d}/{CFG.num_steps}  loss={loss.item():.4f}  train_acc={acc:.3f}")

    # ── Save checkpoint ────────────────────────────────
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(CFG.checkpoint_dir, f"model_{mode}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "mode": mode,
        "train_history": history,
        "episode": {
            "support_x": support_x.cpu(),
            "support_y": support_y.cpu(),
            "query_x": query_x,
            "query_y": query_y,
        }
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")

    final_acc = history["acc"][-1]
    print(f"Final train accuracy: {final_acc:.3f}")
    if final_acc < 0.8:
        print("WARNING: final accuracy < 0.8 — check model/loss setup")

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evidential", "softmax"], required=True)
    args = parser.parse_args()
    train(args.mode)
