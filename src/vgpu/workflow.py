"""High-level, restart-safe Step 9 v2 workflow used by the notebook."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import zlib
from pathlib import Path
from typing import Any, Callable

from .aggregation import vgpu_merge_evaluation_shards
from .controller import (
    VGPUController,
    VGPU_RUNS,
    vgpu_build_data_manifest,
    vgpu_build_source_bundle,
    vgpu_connect_worker,
    vgpu_download_artifact,
    vgpu_load_env,
    vgpu_reattach_worker,
    vgpu_submit_partition,
    vgpu_teardown,
    vgpu_upload_file,
    vgpu_verify_client,
)
from .state import (
    VGPUState,
    vgpu_atomic_write_bytes,
    vgpu_atomic_write_json,
    vgpu_mirror_file,
    vgpu_sha256_file,
)


def vgpu_worker_sha256(repo_root: str | Path) -> str:
    return vgpu_sha256_file(
        Path(repo_root) / "src" / "vgpu" / "vgpu_step9_worker.py"
    )


def vgpu_resolve_data_dir(
    repo_root: str | Path,
    data_dir: str | Path | None = None,
) -> Path:
    """Resolve local datasets, defaulting to the repository's data/ folder."""

    root = Path(repo_root).resolve()
    env = vgpu_load_env(root / ".env")
    candidate = data_dir or env.get("VGPU_DATA_DIR") or (root / "data")
    resolved = Path(candidate).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Step 9 data directory is missing: {resolved}")
    return resolved


def vgpu_bootstrap_cell(repo_root: str | Path) -> str:
    """Return the one self-contained cell pasted before the Kaggle JOIN cell."""

    worker = Path(repo_root) / "src" / "vgpu" / "vgpu_step9_worker.py"
    raw = worker.read_bytes()
    encoded = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
    expected = hashlib.sha256(raw).hexdigest()
    return f'''# Step 9 v2 temporary worker bootstrap (run after CONFIG, before JOIN)
import base64, hashlib, os, subprocess, zlib
from pathlib import Path
repo = os.environ.get("GPU_POOL_REPO") or globals().get("GPU_POOL_REPO")
if not repo:
    raise RuntimeError("run sharedGPU's CONFIG cell so GPU_POOL_REPO is set")
clone = Path("/kaggle/working/gpu_pool_repo")
if not (clone / ".git").is_dir():
    subprocess.run(["git", "clone", "--depth", "1", repo, str(clone)], check=True)
target = clone / "pool_models" / "vgpu_step9.py"
raw = zlib.decompress(base64.b64decode("{encoded}"))
assert hashlib.sha256(raw).hexdigest() == "{expected}"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_bytes(raw)
print("vgpu worker:", target)
print("worker sha256:", hashlib.sha256(target.read_bytes()).hexdigest())
'''


def vgpu_create_controller(
    repo_root: str | Path,
    *,
    run_id: str,
    data_dir: str | Path | None = None,
    drive_dir: str | Path | None = None,
) -> tuple[VGPUController, dict[str, Any]]:
    root = Path(repo_root).resolve()
    env = vgpu_load_env(root / ".env")
    data = vgpu_resolve_data_dir(root, data_dir)
    drive = drive_dir or env.get("VGPU_DRIVE_DIR") or None
    state = VGPUState(root, run_id, drive_dir=drive)
    source_bundle = state.root / "source.tar.gz"
    source = vgpu_build_source_bundle(root, source_bundle)
    vgpu_atomic_write_json(
        state.root / "source_manifest.json",
        {
            "bundle_path": source["path"],
            "bundle_size": source["size"],
            "bundle_sha256": source["sha256"],
            "file_count": source["file_count"],
            "files": source["files"],
        },
    )
    data_manifest = vgpu_build_data_manifest(data)
    vgpu_atomic_write_json(state.root / "data_manifest.json", data_manifest)
    worker_sha = vgpu_worker_sha256(root)
    identity = {
        "worker_sha256": worker_sha,
        "source_sha256": source["sha256"],
        "data_manifest_sha256": data_manifest["sha256"],
    }
    state.vgpu_assert_identity(identity)

    url = env.get("GPU_POOL_URL", "")
    key = env.get("GPU_POOL_API_KEY", "")
    if not url or not key or "example.com" in url or key == "replace-with-your-api-key":
        raise RuntimeError("set real GPU_POOL_URL and GPU_POOL_API_KEY in .env")
    client_module = vgpu_verify_client(root)
    pool = client_module.PoolClient(url, api_key=key)
    manifest = state.vgpu_load_manifest()
    recorded_deployment = (manifest.get("remote") or {}).get("deployment_id")
    attached = None
    if recorded_deployment:
        attached = vgpu_reattach_worker(
            pool,
            deployment_id=recorded_deployment,
            worker_sha256=worker_sha,
            source_sha256=source["sha256"],
            data_manifest_sha256=data_manifest["sha256"],
            run_id=run_id,
        )
    if attached is None:
        deployment, node, remote_status = vgpu_connect_worker(
            pool, worker_sha256=worker_sha
        )
    else:
        deployment, node, remote_status = attached
    controller = VGPUController(
        repo_root=root,
        state=state,
        pool=pool,
        deployment=deployment,
        worker_sha256=worker_sha,
        source_sha256=source["sha256"],
        data_manifest_sha256=data_manifest["sha256"],
    )
    prior_remote = manifest.get("remote") or {}
    manifest["remote"] = {
        **prior_remote,
        "deployment_id": deployment.deployment_id,
        "node_id": deployment.node_id,
        "active_job_id": prior_remote.get("active_job_id"),
        "active_partition_id": prior_remote.get("active_partition_id"),
    }
    state.vgpu_save_manifest(manifest)
    return controller, {
        "source": source,
        "data": data_manifest,
        "node": node,
        "remote_status": remote_status,
    }


def vgpu_stage_run(
    controller: VGPUController,
    preparation: dict[str, Any],
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    source = preparation["source"]
    with controller.state.vgpu_lock():
        manifest = controller.state.vgpu_load_manifest()
        if not manifest.get("integration", {}).get("upload_resume_verified"):
            progress("integration gate: interrupt source upload after one chunk")
            partial = vgpu_upload_file(
                controller,
                source["path"],
                "source.tar.gz",
                file_id="source_bundle",
                stop_after_chunks=1,
            )
            resumed = vgpu_upload_file(
                controller,
                source["path"],
                "source.tar.gz",
                file_id="source_bundle",
            )
            if not resumed.get("complete"):
                raise RuntimeError("source upload did not resume to completion")
            manifest = controller.state.vgpu_load_manifest()
            manifest.setdefault("integration", {})[
                "upload_resume_verified"
            ] = True
            controller.state.vgpu_save_manifest(manifest)
        else:
            progress(f"upload source: {source['size'] / 1e6:.1f} MB")
            vgpu_upload_file(
                controller,
                source["path"],
                "source.tar.gz",
                file_id="source_bundle",
            )
        files = preparation["data"]["files"]
        for index, item in enumerate(files, 1):
            progress(
                f"data {index}/{len(files)}: {item['relative_path']} "
                f"({item['size'] / 1e6:.1f} MB)"
            )
            file_id = hashlib.sha256(
                f"data/{item['relative_path']}".encode()
            ).hexdigest()[:24]
            if (
                index == 1
                and not controller.state.vgpu_load_manifest().get(
                    "integration", {}
                ).get("data_upload_resume_verified")
            ):
                progress("integration gate: interrupt first data upload after one chunk")
                partial = vgpu_upload_file(
                    controller,
                    item["absolute_path"],
                    f"data/{item['relative_path']}",
                    file_id=file_id,
                    stop_after_chunks=1,
                )
            finished = vgpu_upload_file(
                controller,
                item["absolute_path"],
                f"data/{item['relative_path']}",
                file_id=file_id,
            )
            if not finished.get("complete"):
                raise RuntimeError("data upload did not resume to completion")
            if index == 1:
                manifest = controller.state.vgpu_load_manifest()
                manifest.setdefault("integration", {})[
                    "data_upload_resume_verified"
                ] = True
                controller.state.vgpu_save_manifest(manifest)
        prepared = controller.deployment.infer({
            "action": "vgpu_prepare_run",
            "run_id": controller.state.run_id,
            **controller.vgpu_identity(),
            "source_archive": "source.tar.gz",
        })
        return prepared


def _vgpu_checkpoint_name(config_name: str, k_shot: int) -> str:
    interpretation = (
        "prototype-evidential"
        if config_name.endswith("evidential") else "prototype-softmax"
    )
    run_tag = config_name.removeprefix("exp_phase5_")
    if run_tag.endswith("_evidential"):
        run_tag = run_tag[:-len("_evidential")]
    elif run_tag.endswith("_softmax"):
        run_tag = run_tag[:-len("_softmax")]
    suffix = "" if k_shot == 5 else f"_{k_shot}shot"
    return f"model_phase2_{run_tag}{suffix}_{interpretation}_seed42.pt"


def _vgpu_result_name(run: dict[str, str], k_shot: int) -> str:
    if k_shot == 5:
        return run["result"]
    return run["result"].replace("_metrics.json", f"_{k_shot}shot_metrics.json")


def _vgpu_save_config_manifest(
    controller: VGPUController,
    config_name: str,
    k_shot: int,
    update: dict[str, Any],
) -> None:
    manifest = controller.state.vgpu_load_manifest()
    key = f"{config_name}:{k_shot}shot"
    cell = manifest.setdefault("configs", {}).setdefault(key, {})
    cell.update(update)
    controller.state.vgpu_save_manifest(manifest)


def _vgpu_validate_local_shard(
    controller: VGPUController,
    shard: dict[str, Any],
    *,
    config_name: str,
    k_shot: int,
    start: int,
) -> None:
    if shard.get("config_name") != config_name:
        raise ValueError("local shard belongs to another configuration")
    if int(shard.get("k_shot", -1)) != k_shot:
        raise ValueError("local shard belongs to another shot protocol")
    if shard.get("identity") != controller.vgpu_identity():
        raise ValueError("local shard identity hashes do not match this run")
    if shard.get("seeds") != list(range(start, start + 50)):
        raise ValueError("local shard does not contain the expected ordered seeds")


def _vgpu_commit_checkpoint_result(
    controller: VGPUController,
    *,
    compact_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = base64.b64decode(
        result.pop("checkpoint_b64"), validate=True
    )
    observed = hashlib.sha256(checkpoint).hexdigest()
    if observed != result["checkpoint_sha256"]:
        raise ValueError("returned compact checkpoint failed SHA-256")
    controller.state.vgpu_save_blob(
        compact_path.relative_to(controller.state.root), checkpoint
    )
    return result


def vgpu_run_config(
    controller: VGPUController,
    run: dict[str, str],
    *,
    k_shot: int = 5,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    config_name = run["config"]
    state_dir = controller.state.root
    compact_path = state_dir / "checkpoints" / f"{config_name}_{k_shot}shot.pt"
    config_key = f"{config_name}:{k_shot}shot"
    current = (
        controller.state.vgpu_load_manifest().get("configs", {}).get(config_key, {})
    )
    result_path = controller.repo_root / "results" / _vgpu_result_name(run, k_shot)
    checkpoint_path = (
        controller.repo_root / "checkpoints" /
        _vgpu_checkpoint_name(config_name, k_shot)
    )
    if current.get("evaluation_complete"):
        expected = {
            result_path: current.get("metrics_sha256"),
            checkpoint_path: current.get("full_checkpoint_sha256"),
        }
        if all(
            path.is_file() and expected_sha and
            vgpu_sha256_file(path) == expected_sha
            for path, expected_sha in expected.items()
        ):
            progress(f"{config_name}: verified complete; skip")
            return {
                "config_name": config_name,
                "skipped": True,
                "metrics_path": str(result_path),
                "checkpoint_path": str(checkpoint_path),
            }
    checkpoint_b64 = (
        base64.b64encode(compact_path.read_bytes()).decode("ascii")
        if compact_path.exists() else None
    )
    init_payload = {
        "action": "vgpu_initialize_config",
        "run_id": controller.state.run_id,
        **controller.vgpu_identity(),
        "config_name": config_name,
        "k_shot_override": k_shot,
    }
    if checkpoint_b64:
        init_payload["checkpoint_b64"] = checkpoint_b64
    status = vgpu_submit_partition(
        controller,
        init_payload,
        partition_id=f"{config_name}:{k_shot}shot:initialize",
        max_runtime_seconds=1800,
    )
    metadata = status["metadata"]
    progress(
        f"{config_name}: epoch {status['epoch']}/{status['max_epochs']}, "
        f"done={status['done']}"
    )

    if status["done"] and not compact_path.exists():
        terminal = vgpu_submit_partition(
            controller,
            {"action": "vgpu_train_epoch", "config_name": config_name},
            partition_id=f"{config_name}:{k_shot}shot:trainfree",
            max_runtime_seconds=1800,
        )
        terminal = _vgpu_commit_checkpoint_result(
            controller, compact_path=compact_path, result=terminal
        )
        status = terminal
        metadata = status["metadata"]

    while not status["done"]:
        epoch = int(status["epoch"]) + 1
        status = vgpu_submit_partition(
            controller,
            {
                "action": "vgpu_train_epoch",
                "config_name": config_name,
            },
            partition_id=f"{config_name}:{k_shot}shot:train:{epoch}",
            max_runtime_seconds=3600,
            on_progress=lambda job: progress(
                f"{config_name}: {job.progress.get('message') or job.progress}"
            ),
        )
        status = _vgpu_commit_checkpoint_result(
            controller, compact_path=compact_path, result=status
        )
        _vgpu_save_config_manifest(controller, config_name, k_shot, {
            "k_shot": k_shot,
            "epoch": status["epoch"],
            "done": status["done"],
            "best_val_acc": status["best_val_acc"],
            "best_val_epoch": status["best_val_epoch"],
            "compact_checkpoint": str(compact_path),
            "compact_checkpoint_sha256": status["checkpoint_sha256"],
        })
        progress(
            f"{config_name}: epoch {status['epoch']} "
            f"val={status['val_acc']:.3f}, done={status['done']}"
        )
        metadata = status["metadata"]

    temp_result = vgpu_submit_partition(
        controller,
        {"action": "vgpu_fit_temperature", "config_name": config_name},
        partition_id=f"{config_name}:{k_shot}shot:temperature",
        max_runtime_seconds=1800,
    )
    metadata["temperature"] = float(temp_result["temperature"])
    shards = []
    shard_dir = state_dir / "evaluation" / f"{config_name}_{k_shot}shot"
    for start in range(0, 600, 50):
        shard_path = shard_dir / f"seeds_{start:03d}_{start + 49:03d}.json"
        if shard_path.exists():
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            _vgpu_validate_local_shard(
                controller, shard, config_name=config_name,
                k_shot=k_shot, start=start,
            )
            progress(f"{config_name}: skip verified local shard {start}-{start + 49}")
        else:
            shard = vgpu_submit_partition(
                controller,
                {
                    "action": "vgpu_evaluate_shard",
                    "config_name": config_name,
                    "seed_start": start,
                    "seed_count": 50,
                },
                partition_id=f"{config_name}:{k_shot}shot:eval:{start:03d}",
                max_runtime_seconds=3600,
            )
            _vgpu_validate_local_shard(
                controller, shard, config_name=config_name,
                k_shot=k_shot, start=start,
            )
            controller.state.vgpu_commit_blob(
                shard_path.relative_to(controller.state.root),
                json.dumps(shard, indent=2, sort_keys=True).encode() + b"\n",
            )
            progress(f"{config_name}: saved shard {start}-{start + 49}")
        shards.append(shard)

    merged = vgpu_merge_evaluation_shards(
        shards, metadata=metadata, expected_seeds=list(range(600))
    )
    result_bytes = (
        json.dumps(merged["summary"], indent=2, sort_keys=True).encode() + b"\n"
    )
    state_result = controller.state.vgpu_commit_blob(
        Path("results") / result_path.name, result_bytes
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists() and vgpu_sha256_file(result_path) != hashlib.sha256(
        result_bytes
    ).hexdigest():
        raise ValueError(f"refusing to overwrite completed metrics {result_path}")
    if not result_path.exists():
        vgpu_atomic_write_bytes(result_path, state_result.read_bytes())
    _vgpu_materialize_plots(controller.repo_root, result_path, merged)

    exported = vgpu_submit_partition(
        controller,
        {"action": "vgpu_export_checkpoint", "config_name": config_name},
        partition_id=f"{config_name}:{k_shot}shot:export",
        max_runtime_seconds=1800,
    )
    if not exported.get("loader_validated"):
        raise RuntimeError("remote full checkpoint did not pass strict loader validation")
    state_checkpoint = (
        controller.state.root / "full_checkpoints" / checkpoint_path.name
    )
    downloaded = vgpu_download_artifact(
        controller, exported["artifact_id"], state_checkpoint
    )
    if controller.state.drive_dir:
        vgpu_mirror_file(
            state_checkpoint,
            controller.state.drive_dir / "full_checkpoints",
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        if vgpu_sha256_file(checkpoint_path) != downloaded["sha256"]:
            raise ValueError(
                f"refusing to overwrite completed checkpoint {checkpoint_path}"
            )
    else:
        temporary = checkpoint_path.with_name(
            f".{checkpoint_path.name}.{os.getpid()}.tmp"
        )
        shutil.copy2(state_checkpoint, temporary)
        os.replace(temporary, checkpoint_path)
    _vgpu_save_config_manifest(controller, config_name, k_shot, {
        "metrics": str(result_path),
        "metrics_sha256": vgpu_sha256_file(result_path),
        "full_checkpoint": str(checkpoint_path),
        "state_full_checkpoint": downloaded["path"],
        "full_checkpoint_sha256": downloaded["sha256"],
        "evaluation_complete": True,
    })
    return {
        "config_name": config_name,
        "summary": merged["summary"],
        "metrics_path": str(result_path),
        "checkpoint_path": str(checkpoint_path),
    }


def vgpu_run_integration_gate(
    controller: VGPUController,
    *,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Exercise restore/replay and finish the first config before the matrix."""

    run = VGPU_RUNS[0]
    config_name = run["config"]
    compact_path = (
        controller.state.root / "checkpoints" / f"{config_name}_5shot.pt"
    )
    with controller.state.vgpu_lock():
        manifest = controller.state.vgpu_load_manifest()
        if manifest.get("integration", {}).get("training_gate_complete"):
            progress("integration gate: verified complete; skip")
            return {"skipped": True, "config_name": config_name}

        init = {
            "action": "vgpu_initialize_config",
            "run_id": controller.state.run_id,
            **controller.vgpu_identity(),
            "config_name": config_name,
            "k_shot_override": 5,
        }
        if compact_path.exists():
            init["checkpoint_b64"] = base64.b64encode(
                compact_path.read_bytes()
            ).decode("ascii")
        status = vgpu_submit_partition(
            controller,
            init,
            partition_id=f"{config_name}:5shot:gate-initialize",
            max_runtime_seconds=1800,
        )
        if int(status["epoch"]) == 0 and not status["done"]:
            status = vgpu_submit_partition(
                controller,
                {"action": "vgpu_train_epoch", "config_name": config_name},
                partition_id=f"{config_name}:5shot:train:1",
                max_runtime_seconds=3600,
                on_progress=lambda job: progress(
                    job.progress.get("message") or str(job.progress)
                ),
            )
            status = _vgpu_commit_checkpoint_result(
                controller, compact_path=compact_path, result=status
            )
            progress("integration gate: epoch 1 committed locally")

        checkpoint_epoch = int(status["epoch"])
        restore = {
            **init,
            "checkpoint_b64": base64.b64encode(
                compact_path.read_bytes()
            ).decode("ascii"),
        }
        restored = vgpu_submit_partition(
            controller,
            restore,
            partition_id=f"{config_name}:5shot:gate-restore:{checkpoint_epoch}",
            max_runtime_seconds=1800,
        )
        if int(restored["epoch"]) != checkpoint_epoch:
            raise RuntimeError("remote compact-checkpoint restore changed the epoch")
        progress(f"integration gate: restored epoch {checkpoint_epoch}")

        if not restored["done"]:
            next_epoch = checkpoint_epoch + 1
            restored = vgpu_submit_partition(
                controller,
                {"action": "vgpu_train_epoch", "config_name": config_name},
                partition_id=f"{config_name}:5shot:train:{next_epoch}",
                max_runtime_seconds=3600,
                on_progress=lambda job: progress(
                    job.progress.get("message") or str(job.progress)
                ),
            )
            restored = _vgpu_commit_checkpoint_result(
                controller, compact_path=compact_path, result=restored
            )
            progress(f"integration gate: epoch {next_epoch} committed locally")

        result = vgpu_run_config(controller, run, k_shot=5, progress=progress)
        for start in (0, 50):
            shard_path = (
                controller.state.root / "evaluation" /
                f"{config_name}_5shot" /
                f"seeds_{start:03d}_{start + 49:03d}.json"
            )
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            _vgpu_validate_local_shard(
                controller, shard, config_name=config_name,
                k_shot=5, start=start,
            )
        manifest = controller.state.vgpu_load_manifest()
        manifest.setdefault("integration", {}).update({
            "training_gate_complete": True,
            "restored_epoch": checkpoint_epoch,
            "verified_eval_shards": 2,
            "full_checkpoint_reload": True,
        })
        controller.state.vgpu_save_manifest(manifest)
        return result


def _vgpu_materialize_plots(
    repo_root: Path,
    metrics_path: Path,
    merged: dict[str, Any],
) -> None:
    from src.utils.plots import (
        confusion_matrix, ood_histogram, reliability_diagram,
    )

    stem = metrics_path.name.removesuffix("_metrics.json")
    results = repo_root / "results"
    summary = merged["summary"]
    reliability_diagram(
        merged["pooled_probs"], merged["pooled_targets"],
        results / f"{stem}_reliability.png",
        num_bins=15,
        title=f"Reliability ECE={summary['ece_pooled']:.3f}",
    )
    if merged["last_id_scores"] is not None:
        ood_histogram(
            merged["last_id_scores"], merged["last_ood_scores"],
            results / f"{stem}_ood_histogram.png",
            title=f"ID vs SVHN AUROC={summary['ood_auroc_mean']:.3f}",
        )
    confusion_matrix(
        merged["pooled_probs"], merged["pooled_targets"],
        results / f"{stem}_confusion_matrix.png",
        num_classes=5,
        title=f"Confusion acc={summary['accuracy_mean']:.3f}",
    )


def vgpu_run_step9(
    controller: VGPUController,
    *,
    k_shot: int = 5,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    completed = []
    for index, run in enumerate(VGPU_RUNS, 1):
        progress(f"[{index}/10] {run['config']}")
        with controller.state.vgpu_lock():
            completed.append(
                vgpu_run_config(controller, run, k_shot=k_shot, progress=progress)
            )
    return completed


def vgpu_finalize_manifest(controller: VGPUController) -> Path:
    manifest = controller.state.vgpu_load_manifest()
    artifacts = {}
    for root_name in ("results", "checkpoints"):
        root = controller.repo_root / root_name
        for path in sorted(root.glob("*")):
            if not path.is_file():
                continue
            if (
                "phase5_mini" in path.name
                or "step9_" in path.name
                or path.name.startswith("model_phase2_mini")
            ):
                artifacts[path.relative_to(controller.repo_root).as_posix()] = {
                    "size": path.stat().st_size,
                    "sha256": vgpu_sha256_file(path),
                }
    for path in sorted(controller.state.root.rglob("*")):
        if (
            not path.is_file()
            or path == controller.state.lock_path
            or path.name.startswith(".")
            or path.name in {"manifest.json", "MANIFEST.json", "source.tar.gz"}
        ):
            continue
        artifacts[
            "state/" + path.relative_to(controller.state.root).as_posix()
        ] = {
            "size": path.stat().st_size,
            "sha256": vgpu_sha256_file(path),
        }
    manifest["artifacts"] = artifacts
    controller.state.vgpu_save_manifest(manifest)
    output = controller.state.root / "MANIFEST.json"
    vgpu_atomic_write_json(output, manifest)
    return output


def vgpu_close(controller: VGPUController) -> None:
    vgpu_teardown(controller)
