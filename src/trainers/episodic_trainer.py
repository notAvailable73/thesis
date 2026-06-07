"""Episodic meta-training for B-PEFT (Step 4 / Phase 2).

Outer loop = epochs. Inner step = ONE episode.

Per episode:
  1. Sample (support, query) of (n_way × k_shot) + (n_way × q_query)
     images from the TRAIN split (Bertinetto 64 train classes).
  2. Forward both through frozen backbone -> trainable adapter ->
     parameter-free prototype head.
  3. Compute loss on query (cross-entropy for softmax interpretation,
     softplus -> evidential MSE+KL for evidential interpretation).
  4. Backprop. The adapter is the only thing that updates; the prototype
     head is parameter-free.

Per epoch:
  - episodes_per_epoch outer updates,
  - validation on val_episodes_per_epoch episodes from the VAL split
    (Bertinetto 16 val classes) using val_episodes.yaml's frozen seeds,
  - early stop on val accuracy plateau (no improvement for
    early_stop_patience epochs).

KL annealing for evidential mode: linear ramp 0 -> kl_weight_max over
kl_anneal_steps OUTER steps (1 outer step = 1 episode).

Smoke-collapse guard:
  After epoch 1 the validation accuracy MUST exceed 1/n_way + 0.05
  (the spec's R-EPISODIC-COLLAPSE guard, implementation.txt Step 4).
  Below that threshold the trainer raises EpisodicCollapse so Colab
  doesn't burn GPU on a doomed run.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicCollapse(RuntimeError):
    """Raised when validation accuracy after epoch 1 is at or below
    random-chance + 0.05. The spec's R-EPISODIC-COLLAPSE guard."""


def _one_hot(target: torch.Tensor, num_classes: int,
             dtype, device) -> torch.Tensor:
    return torch.eye(num_classes, dtype=dtype, device=device)[target]


@dataclass
class EpisodicHistory:
    """Per-epoch stats. The trainer dumps this into the checkpoint so
    the writeup / wandb summary can read it back without recomputing.
    """
    epoch:            List[int]            = field(default_factory=list)
    train_loss:       List[float]          = field(default_factory=list)
    train_acc:        List[float]          = field(default_factory=list)
    val_acc:          List[float]          = field(default_factory=list)
    val_loss:         List[float]          = field(default_factory=list)
    kl_weight_at_end: List[float]          = field(default_factory=list)


class EpisodicTrainer:
    """Outer epoch loop + per-episode forward + outer step + val + early
    stop. Intentionally KEEPS the same shape as the Step 1-3
    FewShotTrainer so scripts/train.py can branch on cfg.trainer.type
    without much extra wiring.
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

        self.history = EpisodicHistory()
        self.best_val_acc: float = -1.0
        self.best_val_epoch: int = -1
        self.best_state_dict: Optional[dict] = None

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _episode_loss(self, q_logits: torch.Tensor, query_y: torch.Tensor,
                      global_step: int) -> torch.Tensor:
        if self.interpretation == "softmax":
            return F.cross_entropy(q_logits, query_y)

        # Evidential: logits -> evidence via the head's to_evidence (the
        # single source of truth shared with the evaluator, so the
        # softplus/recentre can never drift between train and test).
        evidence = self.model.head.to_evidence(q_logits)
        kl_w = (min(1.0, global_step / max(1, int(self.kl_anneal_steps)))
                * float(self.kl_weight_max))
        target_oh = _one_hot(
            query_y, self.num_classes, evidence.dtype, evidence.device,
        )
        # Reuse the existing Sensoy 2018 implementation.
        from ..losses.evidential import evidential_mse_loss
        return evidential_mse_loss(
            evidence, target_oh, num_classes=self.num_classes, kl_weight=kl_w,
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
        preds = q_logits.argmax(dim=-1)
        return float((preds == qy.to(q_logits.device)).float().mean().item())

    # ------------------------------------------------------------------
    # Train / validate loops
    # ------------------------------------------------------------------
    def _run_train_epoch(self, train_iter: Iterable,
                         global_step_start: int) -> tuple[float, float, int]:
        self.model.train()
        # Make sure frozen-BatchNorm in the backbone stays in eval mode
        # (same invariant as Step 1's FewShotTrainer.fit_episode).
        self.model.backbone.eval()
        for p in self.model.backbone.parameters():
            p.requires_grad = False

        total_loss = 0.0
        total_acc = 0.0
        n_eps = 0
        gs = global_step_start

        for sx, sy, qx, qy in train_iter:
            self.optimizer.zero_grad()
            q_logits = self._forward_episode(sx, sy, qx)
            loss = self._episode_loss(q_logits, qy.to(q_logits.device), gs)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_acc  += self._query_acc_from_logits(q_logits, qy)
            n_eps += 1
            gs += 1

        return total_loss / max(1, n_eps), total_acc / max(1, n_eps), gs

    @torch.no_grad()
    def _run_val_epoch(self, val_iter: Iterable) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_eps = 0
        for sx, sy, qx, qy in val_iter:
            q_logits = self._forward_episode(sx, sy, qx)
            # Use kl_weight=kl_weight_max for the val loss (no anneal at
            # val time). It's only used for monitoring; the early-stop
            # signal is val ACCURACY.
            if self.interpretation == "softmax":
                loss = F.cross_entropy(q_logits, qy.to(q_logits.device))
            else:
                evidence = self.model.head.to_evidence(q_logits)
                target_oh = _one_hot(
                    qy.to(evidence.device), self.num_classes,
                    evidence.dtype, evidence.device,
                )
                from ..losses.evidential import evidential_mse_loss
                loss = evidential_mse_loss(
                    evidence, target_oh, num_classes=self.num_classes,
                    kl_weight=float(self.kl_weight_max or 0.0),
                )
            total_loss += float(loss.item())
            total_acc  += self._query_acc_from_logits(q_logits, qy)
            n_eps += 1
        return total_loss / max(1, n_eps), total_acc / max(1, n_eps)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def fit(self, train_iterable_factory, val_iterable_factory) -> dict:
        """Run the full episodic training schedule.

        Args:
            train_iterable_factory: a *callable* that returns a fresh
                training-episode iterator for the next epoch. The
                trainer calls this once per epoch so each epoch can use
                a different seed_offset and the episodes are NOT the
                same across epochs.
            val_iterable_factory:   same, for the validation stream.
                Typically called with a FIXED seed_offset so val
                comparisons across epochs are paired episode-for-
                episode.

        Returns:
            dict with: history (EpisodicHistory), best_val_acc,
            best_val_epoch, best_state_dict (the meta-trained adapter
            state at the best-validation epoch).
        """
        global_step = 0
        no_improve = 0
        for epoch in range(1, self.num_epochs + 1):
            tr_loss, tr_acc, global_step = self._run_train_epoch(
                train_iterable_factory(epoch), global_step,
            )
            val_loss, val_acc = self._run_val_epoch(
                val_iterable_factory(epoch),
            )
            kl_end = self._kl_weight_at_step(global_step)

            self.history.epoch.append(epoch)
            self.history.train_loss.append(tr_loss)
            self.history.train_acc.append(tr_acc)
            self.history.val_loss.append(val_loss)
            self.history.val_acc.append(val_acc)
            self.history.kl_weight_at_end.append(kl_end)

            if self.logger is not None:
                self.logger.info(
                    f"epoch {epoch:3d}/{self.num_epochs}  "
                    f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                    f"kl_w={kl_end:.3f}  global_step={global_step}"
                )
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "train/epoch": epoch,
                    "train/loss_epoch": tr_loss,
                    "train/acc_epoch":  tr_acc,
                    "val/loss":         val_loss,
                    "val/acc":          val_acc,
                    "train/kl_weight":  kl_end,
                    "train/global_step": global_step,
                }, step=epoch)

            # Collapse guard (R-EPISODIC-COLLAPSE).
            if epoch == 1 and val_acc <= self.collapse_threshold:
                raise EpisodicCollapse(
                    f"validation accuracy after epoch 1 was {val_acc:.3f}, "
                    f"<= collapse_threshold ({self.collapse_threshold}). "
                    f"Likely causes: KL anneal too aggressive, LR too "
                    f"high, or sampler / dataset mis-wired. Abort before "
                    f"burning more compute."
                )

            # Early stop on val acc plateau.
            if val_acc > self.best_val_acc + 1e-6:
                self.best_val_acc = val_acc
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

        # Restore best weights so the caller's `model` is the one with
        # the best val accuracy (not the last-epoch one).
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return {
            "history": self.history,
            "best_val_acc":   float(self.best_val_acc),
            "best_val_epoch": int(self.best_val_epoch),
            "best_state_dict": self.best_state_dict,
        }
