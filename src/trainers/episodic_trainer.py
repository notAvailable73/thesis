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
    # Instrumentation (adopted from the step4_reform diagnostics): per-epoch
    # mean Dirichlet evidence and adapter gradient norm. mean_evidence ~ 0 is
    # the fingerprint of the evidential collapse (softplus starved to zero);
    # adapter_grad_norm ~ 0 confirms no learning signal is reaching the PEFT
    # module. For softmax runs mean_evidence is logged as 0.0 (not applicable).
    mean_evidence:    List[float]          = field(default_factory=list)
    adapter_grad_norm: List[float]         = field(default_factory=list)


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
        evid_prior_per_class: float = 1.0,
        evid_use_variance: bool = True,
        freeze_backbone: bool = True,
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
        # R-EDL knobs (Survey EDL): tunable prior mass + optional variance drop.
        # Defaults reproduce the Sensoy loss exactly; the retuned Phase-2 config
        # sweeps these on VAL to lower ID under-confidence / ECE.
        self.evid_prior_per_class = float(evid_prior_per_class)
        self.evid_use_variance = bool(evid_use_variance)
        # Step 5: Bottleneck / Linear-Probe keep the backbone frozen (default).
        # LoRA / BitFit / Full-FT set freeze_backbone=False so the trainable
        # parameters inside the backbone keep requires_grad=True across epochs.
        self.freeze_backbone = bool(freeze_backbone)

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
            prior_per_class=self.evid_prior_per_class,
            use_variance=self.evid_use_variance,
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
    def _adapter_grad_norm(self) -> float:
        """L2 norm over ALL trainable gradients after backward(). A value of
        ~0 means no learning signal is reaching the PEFT module (the
        evidential-collapse fingerprint on the gradient side).

        Step 5: iterate over every trainable parameter (not just
        model.adapter), because LoRA/BitFit/Full-FT keep their trainable
        parameters inside the backbone, where model.adapter is empty. For
        post-pool adapters (Bottleneck) this is identical to the old
        adapter-only sum, since the backbone is frozen."""
        sq = 0.0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
        return sq ** 0.5

    def _run_train_epoch(self, train_iter: Iterable,
                         global_step_start: int
                         ) -> tuple[float, float, float, float, int]:
        self.model.train()
        # Keep BatchNorm in the backbone in eval mode for ALL adapter types so
        # its running statistics stay frozen at their ImageNet values (updating
        # them from 25-image support batches is unstable and would leak query
        # stats). eval() does NOT block gradients — for LoRA/BitFit/Full-FT the
        # affine/conv params inside the backbone still update.
        self.model.backbone.eval()
        # Only re-freeze the backbone parameters for post-pool / linear-probe
        # adapters. LoRA/BitFit/Full-FT keep their in-backbone params trainable.
        if self.freeze_backbone:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        total_loss = 0.0
        total_acc = 0.0
        total_evidence = 0.0
        total_grad_norm = 0.0
        n_eps = 0
        gs = global_step_start

        for sx, sy, qx, qy in train_iter:
            self.optimizer.zero_grad()
            q_logits = self._forward_episode(sx, sy, qx)
            loss = self._episode_loss(q_logits, qy.to(q_logits.device), gs)
            loss.backward()
            total_grad_norm += self._adapter_grad_norm()  # after backward, pre-step
            self.optimizer.step()

            # Mean Dirichlet evidence this episode (evidential only). Uses the
            # SAME head.to_evidence as the loss, so it tracks exactly what the
            # model trains on. softmax runs have no evidence -> log 0.0.
            if self.interpretation == "evidential":
                with torch.no_grad():
                    total_evidence += float(
                        self.model.head.to_evidence(q_logits).mean().item()
                    )

            total_loss += float(loss.item())
            total_acc  += self._query_acc_from_logits(q_logits, qy)
            n_eps += 1
            gs += 1

        d = max(1, n_eps)
        return (total_loss / d, total_acc / d,
                total_evidence / d, total_grad_norm / d, gs)

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
            tr_loss, tr_acc, tr_evidence, tr_grad_norm, global_step = \
                self._run_train_epoch(
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
            self.history.mean_evidence.append(tr_evidence)
            self.history.adapter_grad_norm.append(tr_grad_norm)

            if self.logger is not None:
                self.logger.info(
                    f"epoch {epoch:3d}/{self.num_epochs}  "
                    f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                    f"kl_w={kl_end:.3f}  mean_ev={tr_evidence:.4f}  "
                    f"grad_norm={tr_grad_norm:.4f}  global_step={global_step}"
                )
            if self.wandb_run is not None:
                self.wandb_run.log({
                    "train/epoch": epoch,
                    "train/loss_epoch": tr_loss,
                    "train/acc_epoch":  tr_acc,
                    "val/loss":         val_loss,
                    "val/acc":          val_acc,
                    "train/kl_weight":  kl_end,
                    "train/mean_evidence":  tr_evidence,
                    "train/adapter_grad_norm": tr_grad_norm,
                    "train/global_step": global_step,
                }, step=epoch)

            # Collapse guard (R-EPISODIC-COLLAPSE), now two-sided:
            #   1. val accuracy at/below chance after epoch 1, OR
            #   2. evidential mean evidence ~ 0 after epoch 1 (the softplus-
            #      starvation fingerprint) -> the model cannot express
            #      confidence and no gradient flows. Abort before Colab burns
            #      a full session on a doomed run.
            # collapse_threshold <= 0 disables BOTH guards. The Full-FT baseline
            # (Step 5) is EXPECTED to behave pathologically (overfit / evidence
            # collapse), so its config sets collapse_threshold: 0.0 to let the
            # run finish and produce the honest (bad) numbers instead of aborting.
            if epoch == 1 and self.collapse_threshold > 0:
                if val_acc <= self.collapse_threshold:
                    raise EpisodicCollapse(
                        f"validation accuracy after epoch 1 was {val_acc:.3f}, "
                        f"<= collapse_threshold ({self.collapse_threshold}). "
                        f"Likely causes: KL anneal too aggressive, LR too "
                        f"high, or sampler / dataset mis-wired. Abort before "
                        f"burning more compute."
                    )
                if self.interpretation == "evidential" and tr_evidence <= 1e-3:
                    raise EpisodicCollapse(
                        f"mean Dirichlet evidence after epoch 1 was "
                        f"{tr_evidence:.2e} (~0): the evidence mapping is "
                        f"starved (softplus saturated to zero) so the model is "
                        f"a uniform-Dirichlet 'I don't know' for every input "
                        f"and no gradient reaches the adapter "
                        f"(grad_norm={tr_grad_norm:.2e}). Check head.metric / "
                        f"evidence_scale_init / evidence_bias_init."
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
