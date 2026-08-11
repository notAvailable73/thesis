"""GPU-only partition runtime loaded from the uploaded thesis source bundle."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import random
from pathlib import Path
from typing import Any


def _vgpu_require_ml():
    import numpy as np
    import torch
    return np, torch


def _vgpu_cpu_state(model) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in {n for n, p in model.named_parameters() if p.requires_grad}
    }


def _vgpu_load_partial(model, state: dict[str, Any]) -> None:
    current = model.state_dict()
    unknown = sorted(set(state) - set(current))
    if unknown:
        raise ValueError(f"compact checkpoint has unknown model keys: {unknown}")
    current.update(state)
    model.load_state_dict(current)


def _vgpu_history_dict(history) -> dict[str, list[Any]]:
    return {
        name: list(getattr(history, name))
        for name in (
            "epoch", "train_loss", "train_acc", "val_acc", "val_loss",
            "kl_weight_at_end", "mean_evidence", "adapter_grad_norm",
        )
    }


def _vgpu_restore_history(history, raw: dict[str, Any]) -> None:
    for name, values in raw.items():
        if hasattr(history, name):
            setattr(history, name, list(values))


def _vgpu_optimizer_to_device(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


def vgpu_validate_staged_assets(data_root: str | Path) -> dict[str, Any]:
    """Refuse implicit dataset downloads; accept the existing loader layouts."""

    root = Path(data_root).resolve()
    from src.datasets.mini_imagenet import (
        _find_csv_layout,
        _find_imagefolder_layout,
        _find_zenodo_pkls,
    )
    from src.datasets.cifar_fs import _find_staged_cifar100_root
    from src.datasets.svhn_ood import _find_staged_svhn_root
    from src.datasets.tinyimagenet_ood import _find_extracted_tin_root

    cache_ready = all(
        (root / f"mini_imagenet_84_{split}.npy").is_file()
        and (root / f"mini_imagenet_84_{split}.json").is_file()
        for split in ("train", "val", "test")
    )
    pkls = _find_zenodo_pkls(str(root)) or {}
    pkl_ready = all(split in pkls for split in ("train", "val", "test"))
    csv_ready = _find_csv_layout(str(root)) is not None
    folder_ready = _find_imagefolder_layout(str(root)) is not None
    if not any((cache_ready, pkl_ready, csv_ready, folder_ready)):
        raise FileNotFoundError(
            "VGPU_DATA_DIR has no complete loader-compatible MiniImageNet "
            "train/val/test source; automatic download is disabled in Step 9 v2"
        )
    cifar_root = _find_staged_cifar100_root(str(root))
    cifar_archive = root / "cifar-100-python.tar.gz"
    if cifar_root is None and not cifar_archive.is_file():
        raise FileNotFoundError(
            "VGPU_DATA_DIR has neither extracted cifar-100-python train/test/meta "
            "nor cifar-100-python.tar.gz; automatic download is disabled in Step 9 v2"
        )
    svhn_root = _find_staged_svhn_root(str(root))
    if svhn_root is None:
        raise FileNotFoundError(
            "VGPU_DATA_DIR has no staged SVHN test_32x32.mat; "
            "automatic download is disabled in Step 9 v2"
        )
    tiny_root = _find_extracted_tin_root(str(root))
    tiny_zip = root / "tiny-imagenet-200.zip"
    if tiny_root is None and not tiny_zip.is_file():
        raise FileNotFoundError(
            "VGPU_DATA_DIR has neither extracted TinyImageNet nor "
            "tiny-imagenet-200.zip; automatic download is disabled in Step 9 v2"
        )
    return {
        "mini_imagenet": (
            "cache" if cache_ready else
            "pkl" if pkl_ready else
            "csv" if csv_ready else "imagefolder"
        ),
        "cifar100": str(cifar_root or cifar_archive),
        "svhn_root": str(svhn_root),
        "tinyimagenet": str(tiny_root or tiny_zip),
    }


class VGPUConfigSession:
    """One persistent config session, partitioned at epoch and eval-shard boundaries."""

    def __init__(
        self,
        *,
        workspace: Path,
        config_name: str,
        identity: dict[str, str],
        checkpoint_bytes: bytes | None,
        k_shot_override: int | None = None,
    ) -> None:
        np, torch = _vgpu_require_ml()
        if not torch.cuda.is_available():
            raise RuntimeError("Step 9 v2 runtime requires CUDA")

        from src.datasets import get_id_split, EpisodicIterableDataset
        from src.models import build_model
        from src.trainers.episodic_trainer import EpisodicTrainer
        from src.utils import count_trainable_params, load_config, set_seed

        self.np = np
        self.torch = torch
        self.workspace = Path(workspace).resolve()
        self.repo_root = self.workspace / "source"
        self.data_root = self.workspace / "data"
        self.config_name = config_name
        self.config_path = self.repo_root / "configs" / f"{config_name}.yaml"
        if not self.config_path.exists():
            raise FileNotFoundError(f"unknown Step 9 config: {self.config_path}")
        self.identity = dict(identity)
        self.device = torch.device("cuda")
        self.asset_status = vgpu_validate_staged_assets(self.data_root)

        old_cwd = Path.cwd()
        try:
            os.chdir(self.repo_root)
            self.cfg = load_config(self.config_path)
            self.cfg.dataset.data_root = str(self.data_root)
            self.cfg.ood.data_root = str(self.data_root)
            if k_shot_override is not None:
                if int(k_shot_override) not in (1, 5):
                    raise ValueError("Step 9 v2 k-shot override must be 1 or 5")
                self.cfg.dataset.k_shot = int(k_shot_override)
            set_seed(int(self.cfg.seed))
            self.train_split = get_id_split(self.cfg.dataset, split="train")
            self.val_split = get_id_split(self.cfg.dataset, split="val")
            self.test_split = get_id_split(self.cfg.dataset, split="test")
            self.model = build_model(self.cfg).to(self.device)
        finally:
            os.chdir(old_cwd)

        self.n_params = int(count_trainable_params(self.model))
        self.interpretation = str(self.cfg.head.get("interpretation", "evidential"))
        self.n_way = int(self.cfg.dataset.n_way)
        self.k_shot = int(self.cfg.dataset.k_shot)
        self.q_query = int(self.cfg.dataset.q_query)
        self.episodes_per_epoch = int(self.cfg.trainer.episodes_per_epoch)
        self.val_episodes_per_epoch = int(self.cfg.trainer.val_episodes_per_epoch)
        self.max_epochs = int(self.cfg.trainer.num_epochs)
        self.train_seed_offset = int(self.cfg.trainer.train_seed_offset)
        with (self.repo_root / self.cfg.eval.val_episodes_file).open() as handle:
            import yaml
            self.val_seeds = list(yaml.safe_load(handle)["seeds"])
        self.val_seed_offset = int(self.val_seeds[0])

        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = (
            torch.optim.Adam(
                self.trainable,
                lr=float(self.cfg.train.lr),
                weight_decay=float(self.cfg.train.weight_decay),
            )
            if self.trainable else None
        )
        self.trainer = (
            EpisodicTrainer(
                model=self.model,
                optimizer=self.optimizer,
                num_classes=self.n_way,
                num_epochs=self.max_epochs,
                episodes_per_epoch=self.episodes_per_epoch,
                val_episodes_per_epoch=self.val_episodes_per_epoch,
                early_stop_patience=int(self.cfg.trainer.early_stop_patience),
                interpretation=self.interpretation,
                kl_weight_max=(
                    float(self.cfg.loss.kl_weight_max)
                    if self.interpretation == "evidential" else None
                ),
                kl_anneal_steps=(
                    int(self.cfg.loss.kl_anneal_steps)
                    if self.interpretation == "evidential" else None
                ),
                ece_bins=int(self.cfg.eval.ece_bins),
                device=self.device,
                collapse_threshold=float(self.cfg.trainer.collapse_threshold),
                evid_prior_per_class=float(self.cfg.loss.get("prior_per_class", 1.0)),
                evid_use_variance=bool(self.cfg.loss.get("use_variance", True)),
                freeze_backbone=not bool(
                    getattr(self.model, "backbone_trainable", False)
                ),
            )
            if self.optimizer is not None else None
        )
        self.epoch = 0
        self.global_step = 0
        self.no_improve = 0
        self.done = False
        self.best_trainable_state: dict[str, Any] | None = None
        self.temperature = 0.0
        self._ood_pools: dict[str, Any] | None = None

        if checkpoint_bytes:
            self.vgpu_restore_checkpoint(checkpoint_bytes)
        elif self.optimizer is None:
            self._vgpu_initialize_trainfree()

    def _vgpu_train_iter(self, epoch: int):
        from src.datasets import EpisodicIterableDataset
        return EpisodicIterableDataset(
            self.train_split,
            n_way=self.n_way,
            k_shot=self.k_shot,
            q_query=self.q_query,
            num_episodes=self.episodes_per_epoch,
            seed_offset=self.train_seed_offset + (epoch - 1) * self.episodes_per_epoch,
        )

    def _vgpu_val_iter(self):
        from src.datasets import EpisodicIterableDataset
        return EpisodicIterableDataset(
            self.val_split,
            n_way=self.n_way,
            k_shot=self.k_shot,
            q_query=self.q_query,
            num_episodes=self.val_episodes_per_epoch,
            seed_offset=self.val_seed_offset,
        )

    def _vgpu_initialize_trainfree(self) -> None:
        self.model.eval()
        correct = total = 0
        with self.torch.no_grad():
            for sx, sy, qx, qy in self._vgpu_val_iter():
                logits = self.model.forward_proto(
                    sx.to(self.device), sy.to(self.device), qx.to(self.device)
                )
                correct += int((logits.argmax(-1).cpu() == qy).sum().item())
                total += int(len(qy))
        val_acc = correct / max(1, total)
        self.best_val_acc = val_acc
        self.best_val_epoch = 0
        self.best_trainable_state = {}
        self.done = True

    @property
    def best_val_acc(self) -> float:
        if self.trainer is None:
            return float(getattr(self, "_trainfree_best_val_acc", -1.0))
        return float(self.trainer.best_val_acc)

    @best_val_acc.setter
    def best_val_acc(self, value: float) -> None:
        if self.trainer is None:
            self._trainfree_best_val_acc = float(value)
        else:
            self.trainer.best_val_acc = float(value)

    @property
    def best_val_epoch(self) -> int:
        if self.trainer is None:
            return int(getattr(self, "_trainfree_best_val_epoch", -1))
        return int(self.trainer.best_val_epoch)

    @best_val_epoch.setter
    def best_val_epoch(self, value: int) -> None:
        if self.trainer is None:
            self._trainfree_best_val_epoch = int(value)
        else:
            self.trainer.best_val_epoch = int(value)

    def vgpu_status(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "epoch": self.epoch,
            "max_epochs": self.max_epochs,
            "global_step": self.global_step,
            "no_improve": self.no_improve,
            "done": self.done,
            "best_val_acc": self.best_val_acc,
            "best_val_epoch": self.best_val_epoch,
            "n_params": self.n_params,
            "interpretation": self.interpretation,
            "device": self.torch.cuda.get_device_name(0),
            "metadata": self.vgpu_metadata(),
            "assets": self.asset_status,
        }

    def _vgpu_capture_rng(self) -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": self.np.random.get_state(),
            "torch_cpu": self.torch.get_rng_state(),
            "torch_cuda": self.torch.cuda.get_rng_state_all(),
        }

    def _vgpu_restore_rng(self, raw: dict[str, Any]) -> None:
        random.setstate(raw["python"])
        self.np.random.set_state(raw["numpy"])
        self.torch.set_rng_state(raw["torch_cpu"])
        self.torch.cuda.set_rng_state_all(raw["torch_cuda"])

    def vgpu_checkpoint_bytes(self) -> bytes:
        history = (
            _vgpu_history_dict(self.trainer.history)
            if self.trainer is not None else {
                "epoch": [], "train_loss": [], "train_acc": [], "val_acc": [],
                "val_loss": [], "kl_weight_at_end": [], "mean_evidence": [],
                "adapter_grad_norm": [],
            }
        )
        payload = {
            "format": "vgpu_compact_v1",
            "identity": self.identity,
            "config_name": self.config_name,
            "k_shot": self.k_shot,
            "trainable_state": _vgpu_cpu_state(self.model),
            "optimizer_state": (
                self.optimizer.state_dict() if self.optimizer is not None else None
            ),
            "best_trainable_state": self.best_trainable_state,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "no_improve": self.no_improve,
            "done": self.done,
            "best_val_acc": self.best_val_acc,
            "best_val_epoch": self.best_val_epoch,
            "history": history,
            "temperature": self.temperature,
            "rng": self._vgpu_capture_rng(),
        }
        buffer = io.BytesIO()
        self.torch.save(payload, buffer)
        return buffer.getvalue()

    def vgpu_restore_checkpoint(self, payload: bytes) -> None:
        raw = self.torch.load(
            io.BytesIO(payload), map_location=self.device, weights_only=False
        )
        if raw.get("format") != "vgpu_compact_v1":
            raise ValueError("unsupported compact checkpoint format")
        if raw.get("identity") != self.identity:
            raise ValueError("compact checkpoint identity differs from this run")
        if raw.get("config_name") != self.config_name:
            raise ValueError("compact checkpoint belongs to another config")
        if int(raw.get("k_shot", self.k_shot)) != self.k_shot:
            raise ValueError("compact checkpoint belongs to another shot protocol")
        _vgpu_load_partial(self.model, raw["trainable_state"])
        if self.optimizer is not None and raw.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(raw["optimizer_state"])
            _vgpu_optimizer_to_device(self.optimizer, self.device)
        self.best_trainable_state = raw.get("best_trainable_state")
        self.epoch = int(raw["epoch"])
        self.global_step = int(raw["global_step"])
        self.no_improve = int(raw["no_improve"])
        self.done = bool(raw["done"])
        self.best_val_acc = float(raw["best_val_acc"])
        self.best_val_epoch = int(raw["best_val_epoch"])
        self.temperature = float(raw.get("temperature", 0.0))
        if self.trainer is not None:
            _vgpu_restore_history(self.trainer.history, raw.get("history", {}))
        self._vgpu_restore_rng(raw["rng"])
        if self.done and self.best_trainable_state is not None:
            _vgpu_load_partial(self.model, self.best_trainable_state)

    def vgpu_train_epoch(self, ctx: Any = None) -> dict[str, Any]:
        from src.trainers.episodic_trainer import EpisodicCollapse

        if self.done:
            checkpoint = self.vgpu_checkpoint_bytes()
            return {
                **self.vgpu_status(),
                "checkpoint_b64": base64.b64encode(checkpoint).decode("ascii"),
                "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
            }
        if self.trainer is None:
            raise RuntimeError("train-free session should already be done")
        epoch = self.epoch + 1
        if ctx is not None:
            ctx.progress(
                step=epoch - 1, total=self.max_epochs, phase="train",
                message=f"{self.config_name}: epoch {epoch}",
            )
        tr_loss, tr_acc, tr_evidence, tr_grad, global_step = (
            self.trainer._run_train_epoch(
                self._vgpu_train_iter(epoch), self.global_step
            )
        )
        val_loss, val_acc = self.trainer._run_val_epoch(self._vgpu_val_iter())
        self.global_step = global_step
        self.epoch = epoch
        history = self.trainer.history
        history.epoch.append(epoch)
        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.kl_weight_at_end.append(
            self.trainer._kl_weight_at_step(global_step)
        )
        history.mean_evidence.append(tr_evidence)
        history.adapter_grad_norm.append(tr_grad)

        threshold = float(self.trainer.collapse_threshold)
        if epoch == 1 and threshold > 0:
            if val_acc <= threshold:
                raise EpisodicCollapse(
                    f"validation accuracy after epoch 1 was {val_acc:.3f}, "
                    f"<= collapse_threshold ({threshold})"
                )
            if self.interpretation == "evidential" and tr_evidence <= 1e-3:
                raise EpisodicCollapse(
                    f"mean Dirichlet evidence after epoch 1 was "
                    f"{tr_evidence:.2e} (~0)"
                )

        improved = val_acc > self.best_val_acc + 1e-6
        if improved:
            self.best_val_acc = val_acc
            self.best_val_epoch = epoch
            self.best_trainable_state = _vgpu_cpu_state(self.model)
            self.no_improve = 0
        else:
            self.no_improve += 1
        self.done = (
            self.no_improve >= int(self.cfg.trainer.early_stop_patience)
            or epoch >= self.max_epochs
        )
        if self.done and self.best_trainable_state is not None:
            _vgpu_load_partial(self.model, self.best_trainable_state)
        checkpoint = self.vgpu_checkpoint_bytes()
        if len(checkpoint) >= 45 * 1024 * 1024:
            raise RuntimeError(
                f"compact checkpoint is {len(checkpoint)} bytes; exceeds safe relay limit"
            )
        if ctx is not None:
            ctx.progress(
                step=epoch, total=self.max_epochs, phase="train",
                message=(
                    f"val_acc={val_acc:.3f}; "
                    f"{'done' if self.done else 'continue'}"
                ),
                val_acc=val_acc,
            )
        return {
            **self.vgpu_status(),
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "mean_evidence": tr_evidence,
            "adapter_grad_norm": tr_grad,
            "improved": improved,
            "checkpoint_b64": base64.b64encode(checkpoint).decode("ascii"),
            "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        }

    def vgpu_fit_temperature(self, ctx: Any = None) -> dict[str, Any]:
        if not self.done:
            raise RuntimeError("training must finish before temperature fitting")
        if self.interpretation != "softmax":
            self.temperature = 0.0
            return {"temperature": 0.0, "skipped": True}
        from src.evaluators.temperature import fit_temperature

        logits_all, targets_all = [], []
        self.model.eval()
        with self.torch.no_grad():
            for index, (sx, sy, qx, qy) in enumerate(self._vgpu_val_iter()):
                sf = self.model.backbone(sx.to(self.device))
                qf = self.model.backbone(qx.to(self.device))
                logits = self.model.forward_proto_from_features(
                    sf, sy.to(self.device), qf
                )
                logits_all.append(logits.cpu())
                targets_all.append(qy.cpu())
                if ctx is not None and (index + 1) % 10 == 0:
                    ctx.progress(
                        step=index + 1, total=self.val_episodes_per_epoch,
                        phase="temperature",
                    )
        self.temperature = float(
            fit_temperature(
                self.torch.cat(logits_all), self.torch.cat(targets_all)
            )
        )
        return {"temperature": self.temperature, "skipped": False}

    def _vgpu_extract_features(self, images, batch_size: int = 64):
        chunks = []
        self.model.backbone.eval()
        with self.torch.no_grad():
            for start in range(0, len(images), batch_size):
                chunks.append(
                    self.model.backbone(
                        images[start:start + batch_size].to(self.device)
                    )
                )
        return self.torch.cat(chunks, dim=0)

    def _vgpu_prepare_ood(self) -> dict[str, Any]:
        if self._ood_pools is not None:
            return self._ood_pools
        from src.datasets import (
            get_gaussian_ood,
            get_heldout_near_ood,
            get_svhn_ood,
            get_tinyimagenet_ood,
        )
        from src.datasets.mini_imagenet import MINI_IMAGENET_ALL_WNIDS

        image_size = int(self.cfg.dataset.image_size)
        count = int(self.cfg.ood.num_samples)
        seed = int(self.cfg.ood.seed)
        pools: dict[str, Any] = {}
        pools["svhn_far"] = self._vgpu_extract_features(
            get_svhn_ood(
                data_root=str(self.data_root), image_size=image_size,
                num_samples=count, seed=seed,
            )
        )
        near_name, near_images = get_heldout_near_ood(
            self.cfg.dataset, num_samples=count, seed=seed, heldout_split="val"
        )
        pools[near_name] = self._vgpu_extract_features(near_images)
        tin_images = get_tinyimagenet_ood(
            data_root=str(self.data_root), image_size=image_size,
            num_samples=count, seed=seed,
            exclude_wnids=MINI_IMAGENET_ALL_WNIDS,
        )
        pools["tin_near"] = self._vgpu_extract_features(tin_images)
        pools["gaussian_far"] = self._vgpu_extract_features(
            get_gaussian_ood(
                image_size=image_size, num_samples=count, seed=seed
            )
        )
        self._ood_pools = pools
        return pools

    def vgpu_evaluate_shard(
        self,
        *,
        seed_start: int,
        seed_count: int,
        ctx: Any = None,
    ) -> dict[str, Any]:
        if not self.done:
            raise RuntimeError("training must finish before evaluation")
        if seed_start < 0 or seed_count <= 0 or seed_start + seed_count > 600:
            raise ValueError("evaluation shard must be within frozen seeds 0..599")

        from src.datasets import EpisodicIterableDataset
        from src.evaluators.accuracy import accuracy, f1_macro
        from src.evaluators.calibration import (
            brier_score, expected_calibration_error,
        )
        from src.evaluators.episodic import _id_score_set, _logits_to_probs
        from src.evaluators.ood import fpr_at_95_tpr, ood_auroc

        pools = self._vgpu_prepare_ood()
        iterator = EpisodicIterableDataset(
            self.test_split,
            n_way=self.n_way,
            k_shot=self.k_shot,
            q_query=self.q_query,
            num_episodes=seed_count,
            seed_offset=seed_start,
        )
        prior = float(self.cfg.loss.get("prior_per_class", 1.0))
        temperature = self.temperature if self.interpretation == "softmax" else None
        per = {"accuracy": [], "f1_macro": [], "ece": [], "brier": []}
        ood: dict[str, dict[str, dict[str, list[float]]]] = {}
        pooled_probs, pooled_logits, pooled_targets = [], [], []
        last_id = last_ood = None
        native = "vacuity" if self.interpretation == "evidential" else "msp"

        self.model.eval()
        with self.torch.no_grad():
            for index, (sx, sy, qx, qy) in enumerate(iterator):
                sf = self.model.backbone(sx.to(self.device))
                qf = self.model.backbone(qx.to(self.device))
                logits = self.model.forward_proto_from_features(
                    sf, sy.to(self.device), qf
                )
                probs = _logits_to_probs(
                    logits, self.n_way, self.interpretation,
                    self.model.head, prior,
                )
                per["accuracy"].append(accuracy(probs, qy.to(self.device)))
                per["f1_macro"].append(
                    f1_macro(probs, qy.to(self.device), num_classes=self.n_way)
                )
                per["ece"].append(
                    expected_calibration_error(
                        probs, qy.to(self.device),
                        num_bins=int(self.cfg.eval.ece_bins),
                    )
                )
                per["brier"].append(
                    brier_score(probs, qy.to(self.device), self.n_way)
                )
                pooled_probs.extend(probs.cpu().tolist())
                pooled_logits.extend(logits.cpu().tolist())
                pooled_targets.extend(qy.tolist())

                id_scores = _id_score_set(
                    logits, self.interpretation, self.model.head,
                    self.n_way, temperature, prior,
                )
                for pool_name, pool_features in pools.items():
                    pool_logits = self.model.forward_proto_from_features(
                        sf, sy.to(self.device), pool_features
                    )
                    pool_scores = _id_score_set(
                        pool_logits, self.interpretation, self.model.head,
                        self.n_way, temperature, prior,
                    )
                    for score_name, id_tensor in id_scores.items():
                        id_np = id_tensor.cpu().numpy()
                        pool_np = pool_scores[score_name].cpu().numpy()
                        cell = ood.setdefault(pool_name, {}).setdefault(
                            score_name, {"auroc": [], "fpr": []}
                        )
                        cell["auroc"].append(float(ood_auroc(id_np, pool_np)))
                        cell["fpr"].append(float(fpr_at_95_tpr(id_np, pool_np)))
                        if pool_name == "svhn_far" and score_name == native:
                            last_id = id_np.tolist()
                            last_ood = pool_np.tolist()
                if ctx is not None and (index + 1) % 5 == 0:
                    ctx.progress(
                        step=index + 1, total=seed_count, phase="evaluate",
                        message=f"seeds {seed_start}..{seed_start + seed_count - 1}",
                    )

        is_final = seed_start + seed_count == 600
        return {
            "config_name": self.config_name,
            "identity": self.identity,
            "k_shot": self.k_shot,
            "seeds": list(range(seed_start, seed_start + seed_count)),
            "per_episode": per,
            "ood": ood,
            "pooled_probs": pooled_probs,
            "pooled_logits": pooled_logits,
            "pooled_targets": pooled_targets,
            "last_id_scores": last_id if is_final else None,
            "last_ood_scores": last_ood if is_final else None,
            "temperature": self.temperature,
        }

    def vgpu_export_checkpoint(self) -> dict[str, Any]:
        if not self.done:
            raise RuntimeError("training must finish before checkpoint export")
        if self.best_trainable_state is not None:
            _vgpu_load_partial(self.model, self.best_trainable_state)
        artifacts = self.workspace / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        filename = f"{self.config_name}.pt"
        path = artifacts / filename
        history = (
            _vgpu_history_dict(self.trainer.history)
            if self.trainer is not None else {
                "epoch": [], "train_loss": [], "train_acc": [], "val_loss": [],
                "val_acc": [self.best_val_acc], "kl_weight_at_end": [],
                "mean_evidence": [], "adapter_grad_norm": [],
            }
        )
        temporary_checkpoint = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        self.torch.save({
            "state_dict": self.model.state_dict(),
            "config_path": str(self.config_path),
            "head_type": self.cfg.head.type,
            "adapter_type": self.cfg.adapter.type,
            "trainer_type": "episodic",
            "interpretation": self.interpretation,
            "best_val_acc": self.best_val_acc,
            "best_val_epoch": self.best_val_epoch,
            "train_history": history,
        }, temporary_checkpoint)
        os.replace(temporary_checkpoint, path)
        # Match scripts/evaluate.py's compatibility boundary: construct the
        # configured model and load the exported state dict strictly.
        from src.models import build_model
        exported = self.torch.load(
            path, map_location="cpu", weights_only=False
        )
        probe_model = build_model(self.cfg)
        probe_model.load_state_dict(exported["state_dict"], strict=True)
        del probe_model, exported
        artifact_id = hashlib.sha256(
            f"{self.config_name}:{path.name}".encode()
        ).hexdigest()[:24]
        info = {
            "relative_path": path.relative_to(self.workspace).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "filename": filename,
        }
        index_path = artifacts / "index.json"
        index = (
            json.loads(index_path.read_text(encoding="utf-8"))
            if index_path.exists() else {}
        )
        index[artifact_id] = info
        temporary = index_path.with_name(".index.json.tmp")
        temporary.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, index_path)
        return {"artifact_id": artifact_id, "loader_validated": True, **info}

    def vgpu_metadata(self) -> dict[str, Any]:
        return {
            "adapter_type": str(self.cfg.adapter.type),
            "config_path": str(self.config_path),
            "episodes_file": str(self.cfg.eval.episodes_file),
            "head_type": str(self.cfg.head.type),
            "interpretation": self.interpretation,
            "n_params": self.n_params,
            "seed": int(self.cfg.seed),
            "trainer_type": "episodic",
            "best_val_epoch": self.best_val_epoch,
            "temperature": self.temperature,
            "prior_per_class": float(self.cfg.loss.get("prior_per_class", 1.0)),
            "num_classes": self.n_way,
            "ece_bins": int(self.cfg.eval.ece_bins),
            "primary_ood_pool": "svhn_far",
        }


def vgpu_create_session(
    *,
    workspace: str | Path,
    config_name: str,
    identity: dict[str, str],
    checkpoint_bytes: bytes | None = None,
    k_shot_override: int | None = None,
) -> VGPUConfigSession:
    return VGPUConfigSession(
        workspace=Path(workspace),
        config_name=config_name,
        identity=identity,
        checkpoint_bytes=checkpoint_bytes,
        k_shot_override=k_shot_override,
    )
