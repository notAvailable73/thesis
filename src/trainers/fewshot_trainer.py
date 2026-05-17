"""Few-shot training loop.

This version trains a single episode end-to-end (Step 1 protocol). Step 4
will extend this to multi-episode episodic training; we keep the same call
signature so the only change there is the outer loop.

Step 1 bug fixes that live here:
  - evidential mode trains with evidential MSE + KL loss (Sensoy 2018 Eq. 5+13),
    NOT cross-entropy on softplus(logits).
  - KL weight is linearly annealed 0 -> kl_weight_max over kl_anneal_steps,
    so MSE can grow evidence before the prior pulls wrong-class alpha back to 1.

Step 3 (action 3.1 + 3.2) adds an optional `wandb_run` so the trainer logs
per-step loss / support_acc / lr / kl_weight to W&B. When wandb_run is None
the loop is identical to Step 2 (the unit tests rely on that).
"""
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FewShotTrainer:
    def __init__(self, model: nn.Module, loss_fn: Callable, optimizer: torch.optim.Optimizer,
                 num_classes: int, num_steps: int = 200, log_every: int = 20,
                 logger=None, wandb_run=None,
                 kl_weight_max: Optional[float] = None,
                 kl_anneal_steps: Optional[int] = None,
                 wandb_step_offset: int = 0):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.log_every = log_every
        self.logger = logger
        self.wandb_run = wandb_run
        self.kl_weight_max = kl_weight_max
        self.kl_anneal_steps = kl_anneal_steps
        # Lets the evaluator log many episodes without colliding step indices.
        self.wandb_step_offset = int(wandb_step_offset)
        self.history = {"step": [], "loss": [], "support_acc": []}

    def fit_episode(self, support_x: torch.Tensor, support_y: torch.Tensor,
                    head_type: str) -> dict:
        """Train one episode.

        Accepts either raw images (4D, B,C,H,W) or pre-computed backbone
        features (2D, B,D). Features are far cheaper because the frozen
        backbone only runs once per episode — this is the Step 1 protocol.
        """
        self.model.train()
        self.model.backbone.eval()  # keep frozen BN in eval mode
        from_features = support_x.dim() == 2
        forward = (self.model.forward_from_features if from_features
                   else self.model.forward)

        for step in range(1, self.num_steps + 1):
            self.optimizer.zero_grad()
            output = forward(support_x)
            loss = self.loss_fn(output, support_y, step=step, num_classes=self.num_classes)
            loss.backward()
            self.optimizer.step()

            # --- Per-step W&B log (action 3.2) --------------------------------
            if self.wandb_run is not None:
                with torch.no_grad():
                    probs = _output_to_probs(output.detach(), head_type)
                    step_acc = (probs.argmax(dim=-1) == support_y).float().mean().item()
                log_data = {
                    "train/loss": float(loss.item()),
                    "train/support_acc": float(step_acc),
                    "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                    "train/step_in_episode": step,
                }
                if (head_type == "evidential"
                        and self.kl_weight_max is not None
                        and self.kl_anneal_steps is not None):
                    log_data["train/kl_weight"] = float(
                        min(1.0, step / max(1, int(self.kl_anneal_steps)))
                        * float(self.kl_weight_max)
                    )
                self.wandb_run.log(log_data, step=self.wandb_step_offset + step)

            if step == 1 or step % self.log_every == 0 or step == self.num_steps:
                with torch.no_grad():
                    probs = _output_to_probs(output.detach(), head_type)
                    acc = (probs.argmax(dim=-1) == support_y).float().mean().item()
                self.history["step"].append(step)
                self.history["loss"].append(float(loss.item()))
                self.history["support_acc"].append(float(acc))
                if self.logger is not None:
                    self.logger.info(
                        f"step {step:4d}/{self.num_steps}  loss={loss.item():.4f}  "
                        f"support_acc={acc:.3f}"
                    )
        return self.history


def _output_to_probs(output: torch.Tensor, head_type: str) -> torch.Tensor:
    if head_type == "evidential":
        alpha = output + 1.0
        return alpha / alpha.sum(dim=-1, keepdim=True)
    return torch.softmax(output, dim=-1)


def train_one_episode(model: nn.Module, support_x: torch.Tensor, support_y: torch.Tensor,
                      loss_fn: Callable, num_classes: int, head_type: str,
                      lr: float = 5e-3, weight_decay: float = 0.0,
                      num_steps: int = 200, log_every: int = 20, logger=None,
                      wandb_run=None, kl_weight_max: Optional[float] = None,
                      kl_anneal_steps: Optional[int] = None,
                      wandb_step_offset: int = 0) -> dict:
    """Convenience helper used by scripts/train.py."""
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    trainer = FewShotTrainer(
        model=model, loss_fn=loss_fn, optimizer=optimizer,
        num_classes=num_classes, num_steps=num_steps, log_every=log_every,
        logger=logger, wandb_run=wandb_run,
        kl_weight_max=kl_weight_max, kl_anneal_steps=kl_anneal_steps,
        wandb_step_offset=wandb_step_offset,
    )
    return trainer.fit_episode(support_x, support_y, head_type=head_type)
