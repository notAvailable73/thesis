"""Episodic meta-training for B-PEFT (Step 4 / Phase 2).

Outer loop = epochs. Inner step = ONE episode.

Per episode:
  1. Sample (support, query) of (n_way × k_shot) + (n_way × q_query)
     images from the TRAIN split (Bertinetto 64 train classes).
  2. Forward both through frozen backbone -> trainable adapter ->
     prototype head.
  3. Compute loss on query (cross-entropy for softmax interpretation,
     softplus -> evidential MSE+KL for evidential interpretation).
  4. Backprop. The adapter is the only thing that updates; the prototype
     head is parameter-free (or has a small learnable affine for
     evidential — see ScaledPrototypeHead in src/heads/prototype_head.py).

Per epoch:
  - episodes_per_epoch outer updates,
  - validation on val_episodes_per_epoch episodes from the VAL split
    (Bertinetto 16 val classes) using val_episodes.yaml's frozen seeds,
  - early stop on val accuracy plateau (no improvement for
    early_stop_patience epochs).

KL annealing for evidential mode: linear ramp 0 -> kl_weight_max over
kl_anneal_steps OUTER steps (1 outer step = 1 episode).

Collapse guard (R-EPISODIC-COLLAPSE — strengthened in action 4.20):
  After epoch 1 ALL of the following are checked; any failure raises
  EpisodicCollapse:
    (a) val_acc (from HEAD PROBABILITIES, not raw logits) <= threshold
    (b) evidential only: mean evidence < 1e-3 (softplus output ~0)
    (c) after epoch 2: train_loss unchanged (|loss2 - loss1| < 1e-5)
  Previously only (a) was checked, and val_acc was incorrectly computed
  from raw prototype scores, masking the evidential collapse.

Instrumentation added (action 4.19):
  - accuracy in both train and val is computed from the HEAD's predicted
    PROBABILITIES (same path as scripts/evaluate.py uses)
  - mean evidence and mean Dirichlet strength S are logged each epoch
    for evidential runs
  - adapter gradient-norm is logged after each epoch's last optimizer step
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicCollapse(RuntimeError):
    """Raised when the collapse guard fires after epoch 1 (or epoch 2).

    Original guard: val_acc <= threshold (from raw prototype scores — this
    was insufficient; it missed the evidential collapse in Step 4's first
    run because the backbone's raw features gave 0.78 accuracy even while
    the evidential output was stuck at 0.200).

    Strengthened guard (action 4.20): also checks mean evidence and loss
    stagnation.
    """


def _one_hot(target: torch.Tensor, num_classes: int,
             dtype, device) -> torch.Tensor:
    return torch.eye(num_classes, dtype=dtype, device=device)[target]


def _logits_to_probs(logits: torch.Tensor,
                     interpretation: str,
                     num_classes: int) -> torch.Tensor:
    """Convert prototype-head logits to a probability vector.

    This is the SAME path that scripts/evaluate.py uses, so the trainer's
    accuracy and the evaluator's accuracy are consistent (action 4.19).

    For softmax: softmax(logits).
    For evidential: softplus(logits) -> evidence -> alpha -> p = alpha/S.
    """
    if interpretation == "softmax":
        return torch.softmax(logits, dim=-1)
    # evidential
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    return alpha / S


def _logits_to_loss_input(logits: torch.Tensor,
                          interpretation: str) -> torch.Tensor:
    """Prototype-head logits to the tensor the loss function expects."""
    if interpretation == "softmax":
        return logits
    if interpretation == "evidential":
        return F.softplus(logits)
    raise ValueError(f"Unknown interpretation: {interpretation!r}")


@dataclass
class EpisodicHistory:
    """Per-epoch stats. Dumped into the checkpoint so writeup / wandb
    can read it back without recomputing."""
    epoch:              List[int]   = field(default_factory=list)
    train_loss:         List[float] = field(default_factory=list)
    train_acc:          List[float] = field(default_factory=list)
    val_acc:            List[float] = field(default_factory=list)
    val_loss:           List[float] = field(default_factory=list)
    kl_weight_at_end:   List[float] = field(default_factory=list)
    # Evidential-specific diagnostics (action 4.19).
    mean_evidence:      List[float] = field(default_factory=list)
    mean_strength_S:    List[float] = field(default_factory=list)
    adapter_grad_norm:  List[float] = field(default_factory=list)


class EpisodicTrainer:
    """Outer epoch loop + per-episode forward + outer step + val + early stop.

    Intentionally keeps the same shape as the Step 1-3 FewShotTrainer so
    scripts/train.py can branch on cfg.trainer.type without much extra
    wiring.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        num_classes: int,
        num_epochs: int,
        episodes_per_epoch: int,
        val_episodes_per_epoch: int,
        early_stop_patience: int,
        interpretation: str,           # "softmax" | "evidential"
        kl_weight_max: Optional[float] = None,
        kl_anneal_steps: Optional[int] = None,
        ece_bins: int = 15,
        logger=None,
        wandb_run=None,
        device: torch.device | str = "cpu",
        collapse_threshold: float = 0.25,
        # Minimum mean evidence below which collapse is declared (4.20b).
        min_evidence_threshold: float = 1e-3,
    ):
        if interpretation not in ("softmax", "evidential"):
            raise ValueError(
                f"interpretation must be 'softmax' or 'evidential', got "
                f"{interpretation!r}"
            )
        if interpretation == "evidential" and (
            kl_weight_max is None or kl_anneal_steps is None
        ):
            raise ValueError(
                "evidential interpretation requires kl_weight_max + "
                "kl_anneal_steps"
            )

        self.model = model
        self.optimizer = optimizer
        self.num_classes = int(num_classes)
        self.num_epochs = int(num_epochs)
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.val_episodes_per_epoch = int(val_episodes_per_epoch)
        self.early_stop_patience = int(early_stop_patience)
        self.interpretation = interpretation
        self.kl_weight_max = (None if kl_weight_max is None
                              else float(kl_weight_max))
        self.kl_anneal_steps = (None if kl_anneal_steps is None
                                else int(kl_anneal_steps))
        self.ece_bins = int(ece_bins)
        self.logger = logger
        self.wandb_run = wandb_run
        self.device = device
        self.collapse_threshold = float(collapse_threshold)
        self.min_evidence_threshold = float(min_evidence_threshold)

        self.history = EpisodicHistory()
        self.best_val_acc: float = -1.0
        self.best_val_epoch: int = -1
        self.best_state_dict: Optional[dict] = None

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _episode_loss(self, q_logits: torch.Tensor, query_y: torch.Tensor,
                      global_step: int) -> torch.Tensor:
        loss_input = _logits_to_loss_input(q_logits, self.interpretation)
        if self.interpretation == "softmax":
            return F.cross_entropy(loss_input, query_y)

        kl_w = (min(1.0, global_step / max(1, int(self.kl_anneal_steps)))
                * float(self.kl_weight_max))
        target_oh = _one_hot(
            query_y, self.num_classes, loss_input.dtype, loss_input.device,
        )
        from ..losses.evidential import evidential_mse_loss
        return evidential_mse_loss(
            loss_input, target_oh, num_classes=self.num_classes, kl_weight=kl_w,
        )

    def _kl_weight_at_step(self, step: int) -> float:
        if self.interpretation != "evidential":
            return 0.0
        return (min(1.0, step / max(1, int(self.kl_anneal_steps)))
                * float(self.kl_weight_max))

    # ------------------------------------------------------------------
    # Episode forward
    # ------------------------------------------------------------------
    def _forward_episode(self, sx, sy, qx):
        sx = sx.to(self.device); sy = sy.to(self.device)
        qx = qx.to(self.device)
        return self.model.forward_proto(sx, sy, qx)

    def _query_acc_from_logits(self, q_logits: torch.Tensor,
                               qy: torch.Tensor) -> float:
        """Accuracy from the HEAD's predicted probabilities (action 4.19).

        Uses the same softmax / evidential probability path as
        scripts/evaluate.py, so the trainer's reported accuracy matches
        what the evaluator measures. Previously used raw logit argmax,
        which gave correct results for softmax but masked evidential
        collapse (frozen backbone features still ranked correctly even
        when evidential probabilities were uniform).
        """
        probs = _logits_to_probs(q_logits, self.interpretation, self.num_classes)
        preds = probs.argmax(dim=-1)
        return float((preds == qy.to(q_logits.device)).float().mean().item())

    def _mean_evidence(self, q_logits: torch.Tensor) -> float:
        """Mean evidential strength (action 4.19 logging). 0 for softmax."""
        if self.interpretation != "evidential":
            return 0.0
        return float(F.softplus(q_logits).mean().item())

    def _adapter_grad_norm(self) -> float:
        """L2 norm of all adapter parameter gradients (action 4.19)."""
        sq_sum = sum(
            p.grad.detach().norm().item() ** 2
            for p in self.model.adapter.parameters()
            if p.grad is not None
        )
        return sq_sum ** 0.5

    # ------------------------------------------------------------------
    # Train / validate loops
    # ------------------------------------------------------------------
    def _run_train_epoch(
        self, train_iter: Iterable, global_step_start: int
    ) -> tuple[float, float, int, float, float]:
        """Run one training epoch.

        Returns (train_loss, train_acc, global_step, mean_evidence, adapter_grad_norm).
        """
        self.model.train()
        self.model.backbone.eval()
        for p in self.model.backbone.parameters():
            p.requires_grad = False

        total_loss      = 0.0
        total_acc       = 0.0
        total_evidence  = 0.0
        last_grad_norm  = 0.0
        n_eps = 0
        gs = global_step_start

        for sx, sy, qx, qy in train_iter:
            self.optimizer.zero_grad()
            q_logits = self._forward_episode(sx, sy, qx)
            loss = self._episode_loss(q_logits, qy.to(q_logits.device), gs)
            loss.backward()
            last_grad_norm = self._adapter_grad_norm()
            self.optimizer.step()

            total_loss     += float(loss.item())
            total_acc      += self._query_acc_from_logits(q_logits.detach(), qy)
            total_evidence += self._mean_evidence(q_logits.detach())
            n_eps += 1
            gs += 1

        n = max(1, n_eps)
        return (total_loss / n, total_acc / n, gs,
                total_evidence / n, last_grad_norm)

    @torch.no_grad()
    def _run_val_epoch(
        self, val_iter: Iterable
    ) -> tuple[float, float, float, float]:
        """Run one validation epoch.

        Returns (val_loss, val_acc, mean_evidence, mean_strength_S).
        """
        self.model.eval()
        total_loss     = 0.0
        total_acc      = 0.0
        total_evidence = 0.0
        total_S        = 0.0
        n_eps = 0

        for sx, sy, qx, qy in val_iter:
            q_logits = self._forward_episode(sx, sy, qx)
            loss_input = _logits_to_loss_input(q_logits, self.interpretation)
            if self.interpretation == "softmax":
                loss = F.cross_entropy(loss_input, qy.to(loss_input.device))
            else:
                target_oh = _one_hot(
                    qy.to(loss_input.device), self.num_classes,
                    loss_input.dtype, loss_input.device,
                )
                from ..losses.evidential import evidential_mse_loss
                loss = evidential_mse_loss(
                    loss_input, target_oh, num_classes=self.num_classes,
                    kl_weight=float(self.kl_weight_max or 0.0),
                )
                # Log evidence diagnostics.
                ev = loss_input  # already = softplus(logits) for evidential
                total_evidence += float(ev.mean().item())
                total_S        += float((ev + 1.0).sum(dim=-1).mean().item())

            total_loss += float(loss.item())
            total_acc  += self._query_acc_from_logits(q_logits, qy)
            n_eps += 1

        n = max(1, n_eps)
        return (total_loss / n, total_acc / n,
                total_evidence / n, total_S / n)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def fit(self, train_iterable_factory, val_iterable_factory) -> dict:
        """Run the full episodic training schedule.

        Args:
            train_iterable_factory: callable returning a fresh training-
                episode iterator for each epoch (called once per epoch).
            val_iterable_factory:   same for validation (fixed seed_offset
                so val comparisons across epochs are paired episode-for-
                episode).

        Returns:
            dict with: history, best_val_acc, best_val_epoch,
            best_state_dict.
        """
        global_step = 0
        no_improve  = 0

        for epoch in range(1, self.num_epochs + 1):
            (tr_loss, tr_acc, global_step,
             tr_evidence, adapter_gnorm) = self._run_train_epoch(
                train_iterable_factory(epoch), global_step,
            )
            val_loss, val_acc, val_evidence, val_S = self._run_val_epoch(
                val_iterable_factory(epoch),
            )
            kl_end = self._kl_weight_at_step(global_step)

            self.history.epoch.append(epoch)
            self.history.train_loss.append(tr_loss)
            self.history.train_acc.append(tr_acc)
            self.history.val_loss.append(val_loss)
            self.history.val_acc.append(val_acc)
            self.history.kl_weight_at_end.append(kl_end)
            self.history.mean_evidence.append(val_evidence)
            self.history.mean_strength_S.append(val_S)
            self.history.adapter_grad_norm.append(adapter_gnorm)

            if self.logger is not None:
                self.logger.info(
                    f"epoch {epoch:3d}/{self.num_epochs}  "
                    f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                    f"kl_w={kl_end:.3f}  "
                    f"mean_ev={val_evidence:.4f}  mean_S={val_S:.4f}  "
                    f"grad_norm={adapter_gnorm:.4f}  "
                    f"global_step={global_step}"
                )
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "train/epoch":        epoch,
                    "train/loss_epoch":   tr_loss,
                    "train/acc_epoch":    tr_acc,
                    "train/mean_evidence": tr_evidence,
                    "train/adapter_grad_norm": adapter_gnorm,
                    "val/loss":           val_loss,
                    "val/acc":            val_acc,
                    "val/mean_evidence":  val_evidence,
                    "val/mean_strength_S": val_S,
                    "train/kl_weight":    kl_end,
                    "train/global_step":  global_step,
                }, step=epoch)

            # ----------------------------------------------------------
            # Collapse guard (action 4.20 — strengthened).
            # Check after epoch 1 for (a) and (b); after epoch 2 for (c).
            # ----------------------------------------------------------
            if epoch == 1:
                _reason = self._collapse_reason_epoch1(val_acc, val_evidence)
                if _reason:
                    raise EpisodicCollapse(
                        f"Collapse detected after epoch 1: {_reason}  "
                        f"val_acc={val_acc:.3f}  mean_evidence={val_evidence:.6f}  "
                        f"train_loss={tr_loss:.6f}"
                    )
            if epoch == 2:
                _reason = self._collapse_reason_epoch2(tr_loss)
                if _reason:
                    raise EpisodicCollapse(
                        f"Collapse detected after epoch 2: {_reason}  "
                        f"train_loss_ep1={self.history.train_loss[0]:.6f}  "
                        f"train_loss_ep2={tr_loss:.6f}"
                    )

            # Early stop on val acc plateau.
            if val_acc > self.best_val_acc + 1e-6:
                self.best_val_acc   = val_acc
                self.best_val_epoch = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.early_stop_patience:
                    if self.logger is not None:
                        self.logger.info(
                            f"early stop: no val improvement for "
                            f"{no_improve} epochs; "
                            f"best={self.best_val_acc:.3f} at "
                            f"epoch {self.best_val_epoch}"
                        )
                    break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return {
            "history":        self.history,
            "best_val_acc":   float(self.best_val_acc),
            "best_val_epoch": int(self.best_val_epoch),
            "best_state_dict": self.best_state_dict,
        }

    # ------------------------------------------------------------------
    # Collapse guard helpers (action 4.20)
    # ------------------------------------------------------------------
    def _collapse_reason_epoch1(self, val_acc: float,
                                val_evidence: float) -> str:
        """Return a non-empty reason string if collapse is detected after
        epoch 1, else return empty string."""
        reasons = []
        # (a) probability-based val_acc at or below random-chance threshold.
        if val_acc <= self.collapse_threshold:
            reasons.append(
                f"val_acc {val_acc:.3f} <= threshold {self.collapse_threshold}"
            )
        # (b) evidential only: mean evidence near zero → softplus output ~0.
        if (self.interpretation == "evidential"
                and val_evidence < self.min_evidence_threshold):
            reasons.append(
                f"mean_evidence {val_evidence:.2e} < "
                f"min_evidence_threshold {self.min_evidence_threshold:.2e}"
            )
        return "; ".join(reasons)

    def _collapse_reason_epoch2(self, tr_loss_ep2: float) -> str:
        """Return a non-empty reason string if the loss did not move
        between epoch 1 and epoch 2 (stagnation = (c) in 4.20)."""
        if len(self.history.train_loss) < 2:
            return ""
        tr_loss_ep1 = self.history.train_loss[0]
        if abs(tr_loss_ep2 - tr_loss_ep1) < 1e-5:
            return (
                f"train_loss frozen: |{tr_loss_ep2:.6f} - "
                f"{tr_loss_ep1:.6f}| < 1e-5"
            )
        return ""
