"""Local controller for the Step 9 v2 virtual-GPU workflow."""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import io
import json
import os
import runpy
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .state import (
    VGPUState,
    vgpu_atomic_write_bytes,
    vgpu_atomic_write_json,
    vgpu_sha256_file,
)

VGPU_CLIENT_SHA256 = "a68de6e03e639b77aabcccf55b36719e2702f1b9fee04cee8393302a65345fee"
VGPU_CHUNK_BYTES = 16 * 1024 * 1024
VGPU_WORKER_ENTRYPOINT = "pool_models.vgpu_step9:vgpu_build"

VGPU_RUNS: tuple[dict[str, str], ...] = (
    {
        "config": "exp_phase5_mini_postpool_evidential",
        "suffix": "phase5_mini_postpool",
        "result": "phase5_mini_postpool_bottleneck_prototype-evidential_metrics.json",
    },
    {
        "config": "exp_phase5_mini_postpool_softmax",
        "suffix": "phase5_mini_postpool",
        "result": "phase5_mini_postpool_bottleneck_prototype-softmax_metrics.json",
    },
    {
        "config": "exp_phase5_mini_parallel_evidential",
        "suffix": "phase5_mini_parallel",
        "result": "phase5_mini_parallel_bottleneck_prototype-evidential_metrics.json",
    },
    {
        "config": "exp_phase5_mini_parallel_softmax",
        "suffix": "phase5_mini_parallel",
        "result": "phase5_mini_parallel_bottleneck_prototype-softmax_metrics.json",
    },
    {
        "config": "exp_phase5_mini_mbnet_postpool_evidential",
        "suffix": "phase5_mini_mbnet_postpool",
        "result": "phase5_mini_mbnet_postpool_bottleneck_prototype-evidential_metrics.json",
    },
    {
        "config": "exp_phase5_mini_mbnet_postpool_softmax",
        "suffix": "phase5_mini_mbnet_postpool",
        "result": "phase5_mini_mbnet_postpool_bottleneck_prototype-softmax_metrics.json",
    },
    {
        "config": "exp_phase5_mini_mbnet_parallel_evidential",
        "suffix": "phase5_mini_mbnet_parallel",
        "result": "phase5_mini_mbnet_parallel_bottleneck_prototype-evidential_metrics.json",
    },
    {
        "config": "exp_phase5_mini_mbnet_parallel_softmax",
        "suffix": "phase5_mini_mbnet_parallel",
        "result": "phase5_mini_mbnet_parallel_bottleneck_prototype-softmax_metrics.json",
    },
    {
        "config": "exp_phase5_mini_linear_probe_evidential",
        "suffix": "phase5_mini",
        "result": "phase5_mini_linear_probe_prototype-evidential_metrics.json",
    },
    {
        "config": "exp_phase5_mini_linear_probe_softmax",
        "suffix": "phase5_mini",
        "result": "phase5_mini_linear_probe_prototype-softmax_metrics.json",
    },
)


def vgpu_verify_client(
    repo_root: str | Path,
    *,
    expected_sha256: str = VGPU_CLIENT_SHA256,
):
    """Hash and import the exact standalone client verified by the relay guide."""

    client_path = Path(repo_root).resolve() / "mainul-doc" / "client.py"
    observed = vgpu_sha256_file(client_path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"unverified GPU relay client: {observed}; expected {expected_sha256}"
        )
    spec = importlib.util.spec_from_file_location("vgpu_verified_pool_client", client_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import verified relay client from {client_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def vgpu_load_env(path: str | Path) -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = Path(path)
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    for key in ("GPU_POOL_URL", "GPU_POOL_API_KEY", "VGPU_DATA_DIR", "VGPU_DRIVE_DIR"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def _vgpu_source_paths(repo_root: Path) -> Iterable[Path]:
    excluded_roots = {
        ".git", "data", "results", "checkpoints", "notebooks", "wandb",
        ".venv", "venv", ".pytest_cache",
    }
    allowed_top_files = {
        "requirements.txt", "implementation.txt", "plan.txt", "progress.txt",
        "proposal.txt",
    }
    for path in sorted(repo_root.rglob("*")):
        relative = path.relative_to(repo_root)
        if not relative.parts:
            continue
        if relative.parts[0] in excluded_roots:
            continue
        if any(part == "__pycache__" for part in relative.parts):
            continue
        if path.is_dir():
            continue
        if relative.parts[0] in {"src", "scripts", "configs"}:
            yield path
        elif (
            len(relative.parts) == 1
            and (
                relative.name in allowed_top_files
                or (
                    relative.name.startswith("requirements")
                    and relative.suffix == ".txt"
                )
            )
        ):
            yield path


def vgpu_build_source_bundle(
    repo_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create a deterministic gzip-compressed source tarball."""

    root = Path(repo_root).resolve()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    files = list(_vgpu_source_paths(root))
    source_manifest = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": vgpu_sha256_file(path),
        }
        for path in files
    ]
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for path in files:
                    relative = path.relative_to(root)
                    info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, target)
    return {
        "path": str(target.resolve()),
        "size": target.stat().st_size,
        "sha256": vgpu_sha256_file(target),
        "file_count": len(files),
        "files": source_manifest,
    }


def vgpu_build_data_manifest(data_dir: str | Path) -> dict[str, Any]:
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VGPU_DATA_DIR is not a directory: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or "__pycache__" in relative.parts
            or (relative.parts and relative.parts[0] == "vgpu_step9_state")
        ):
            continue
        files.append({
            "relative_path": relative.as_posix(),
            "absolute_path": str(path),
            "size": path.stat().st_size,
            "sha256": vgpu_sha256_file(path),
        })
    if not files:
        raise FileNotFoundError(f"VGPU_DATA_DIR contains no files: {root}")
    logical_files = [
        {
            "relative_path": item["relative_path"],
            "size": item["size"],
            "sha256": item["sha256"],
        }
        for item in files
    ]
    canonical = json.dumps(
        logical_files, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "root": str(root),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def vgpu_connect_worker(
    pool,
    *,
    worker_sha256: str,
    min_free_gib: float = 12.0,
    min_quota_seconds: float = 4 * 3600,
    mem_gib: float = 12.0,
):
    candidates = []
    for node in pool.nodes():
        if node.get("provider") != "kaggle" or node.get("draining"):
            continue
        if "T4" not in str(node.get("gpuName", "")).upper():
            continue
        free = float(node.get("freeVramGib", 0.0))
        quota = node.get("quotaRemainingSeconds")
        if free < min_free_gib:
            continue
        if quota is not None and float(quota) < min_quota_seconds:
            continue
        candidates.append(node)
    if not candidates:
        raise RuntimeError(
            f"no non-draining Kaggle T4 has {min_free_gib:g} GiB free and "
            f"{min_quota_seconds / 3600:g}h remaining"
        )
    node = max(
        candidates,
        key=lambda item: (
            float(item.get("quotaRemainingSeconds") or 10**12),
            float(item.get("freeVramGib", 0.0)),
        ),
    )
    deployment = pool.deploy(
        {
            "kind": "python",
            "entrypoint": VGPU_WORKER_ENTRYPOINT,
            "args": {"expected_sha256": worker_sha256},
        },
        mem_gib=mem_gib,
        node_id=node["nodeId"],
    )
    status = deployment.infer({"action": "vgpu_status"})
    if status.get("worker_sha256") != worker_sha256:
        deployment.teardown()
        raise RuntimeError("Kaggle worker checksum does not match local worker")
    if not status.get("cuda"):
        deployment.teardown()
        raise RuntimeError("vgpu worker loaded on a node without CUDA")
    return deployment, node, status


def vgpu_reattach_worker(
    pool,
    *,
    deployment_id: str,
    worker_sha256: str,
    source_sha256: str,
    data_manifest_sha256: str,
    run_id: str,
):
    """Reuse a recorded live deployment only when its complete identity agrees."""

    live = {
        item.get("deploymentId"): item
        for item in pool.deployments()
    }
    item = live.get(deployment_id)
    if item is None:
        return None
    client_module = sys.modules.get("vgpu_verified_pool_client")
    if client_module is None:
        raise RuntimeError("verified relay client must be loaded before reattachment")
    deployment = client_module.Deployment(
        pool,
        deployment_id,
        item["nodeId"],
        item.get("modelSpec") or {},
    )
    status = deployment.infer({"action": "vgpu_status"})
    if status.get("worker_sha256") != worker_sha256:
        deployment.teardown()
        return None
    identity = deployment.infer({
        "action": "vgpu_run_identity",
        "run_id": run_id,
        "worker_sha256": worker_sha256,
        "source_sha256": source_sha256,
        "data_manifest_sha256": data_manifest_sha256,
    })
    expected = {
        "worker_sha256": worker_sha256,
        "source_sha256": source_sha256,
        "data_manifest_sha256": data_manifest_sha256,
    }
    if identity.get("identity") != expected:
        deployment.teardown()
        return None
    return deployment, item, status


@dataclass
class VGPUController:
    repo_root: Path
    state: VGPUState
    pool: Any
    deployment: Any
    worker_sha256: str
    source_sha256: str
    data_manifest_sha256: str

    def vgpu_identity(self) -> dict[str, str]:
        return {
            "worker_sha256": self.worker_sha256,
            "source_sha256": self.source_sha256,
            "data_manifest_sha256": self.data_manifest_sha256,
        }


def _vgpu_base_payload(controller: VGPUController) -> dict[str, Any]:
    return {
        "run_id": controller.state.run_id,
        **controller.vgpu_identity(),
    }


def vgpu_upload_file(
    controller: VGPUController,
    local_path: str | Path,
    remote_path: str,
    *,
    file_id: str | None = None,
    chunk_bytes: int = VGPU_CHUNK_BYTES,
    stop_after_chunks: int | None = None,
) -> dict[str, Any]:
    path = Path(local_path)
    identity = {
        "file_id": file_id or hashlib.sha256(remote_path.encode()).hexdigest()[:24],
        "remote_path": remote_path,
        "size": path.stat().st_size,
        "sha256": vgpu_sha256_file(path),
    }
    base = _vgpu_base_payload(controller)
    begin = controller.deployment.infer({
        **base, "action": "vgpu_begin_upload", **identity,
    })
    if bool(begin.get("complete")):
        return begin
    offset = int(begin["offset"])
    sent = 0
    with path.open("rb") as handle:
        handle.seek(offset)
        while offset < identity["size"]:
            block = handle.read(min(chunk_bytes, identity["size"] - offset))
            block_sha = hashlib.sha256(block).hexdigest()
            reply = controller.deployment.infer({
                **base,
                "action": "vgpu_upload_chunk",
                "file_id": identity["file_id"],
                "offset": offset,
                "data_b64": base64.b64encode(block).decode("ascii"),
                "chunk_sha256": block_sha,
            })
            offset = int(reply["offset"])
            sent += 1
            if stop_after_chunks is not None and sent >= stop_after_chunks:
                return {**identity, "offset": offset, "complete": False}
    finished = controller.deployment.infer({
        **base, "action": "vgpu_finish_upload", **identity,
    })
    manifest = controller.state.vgpu_load_manifest()
    manifest.setdefault("uploads", {})[identity["file_id"]] = finished
    controller.state.vgpu_save_manifest(manifest)
    return finished


def vgpu_submit_partition(
    controller: VGPUController,
    payload: dict[str, Any],
    *,
    partition_id: str,
    max_runtime_seconds: float = 3600,
    on_progress: Callable[[Any], None] | None = None,
) -> Any:
    body = {
        **_vgpu_base_payload(controller),
        **payload,
        "partition_id": partition_id,
    }
    manifest = controller.state.vgpu_load_manifest()
    remote = manifest.get("remote") or {}
    old_job_id = remote.get("active_job_id")
    old_partition_id = remote.get("active_partition_id")
    job = None
    if old_job_id and old_partition_id == partition_id:
        try:
            candidate = controller.pool.job_handle(old_job_id)
            if candidate.state == "succeeded":
                result = candidate.result()
                remote.update({
                    "active_job_id": None,
                    "active_partition_id": None,
                    "last_completed_job_id": old_job_id,
                })
                manifest["remote"] = remote
                controller.state.vgpu_save_manifest(manifest)
                return result
            if not candidate.done:
                job = candidate
        except Exception:
            job = None
    elif old_job_id:
        try:
            previous = controller.pool.job_handle(old_job_id)
            if not previous.done:
                previous.wait(on_progress=on_progress)
            elif previous.state == "succeeded":
                previous.result()
        except Exception:
            # A failed/lost previous partition is safe to rerun from the last
            # locally committed boundary. Its error remains in relay history.
            pass
        manifest = controller.state.vgpu_load_manifest()
        manifest.setdefault("remote", {}).update({
            "active_job_id": None,
            "active_partition_id": None,
            "last_resolved_job_id": old_job_id,
        })
        controller.state.vgpu_save_manifest(manifest)
    if job is None:
        job = controller.deployment.submit(
            body,
            max_runtime_seconds=max_runtime_seconds,
            resumable=False,
        )
    manifest["remote"] = {
        **manifest.get("remote", {}),
        "deployment_id": controller.deployment.deployment_id,
        "node_id": controller.deployment.node_id,
        "active_job_id": job.job_id,
        "active_partition_id": partition_id,
    }
    controller.state.vgpu_save_manifest(manifest)
    try:
        result = job.wait(on_progress=on_progress)
    except BaseException as exc:
        failure = {
            "job_id": job.job_id,
            "partition_id": partition_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "recorded_unix": time.time(),
        }
        controller.state.vgpu_save_blob(
            Path("logs") / f"failure_{job.job_id}.json",
            json.dumps(failure, indent=2, sort_keys=True).encode() + b"\n",
        )
        raise
    else:
        manifest = controller.state.vgpu_load_manifest()
        manifest.setdefault("remote", {}).update({
            "active_job_id": None,
            "active_partition_id": None,
            "last_completed_job_id": job.job_id,
        })
        controller.state.vgpu_save_manifest(manifest)
        return result


def vgpu_resume_run(controller: VGPUController) -> dict[str, Any]:
    manifest = controller.state.vgpu_assert_identity(controller.vgpu_identity())
    remote = manifest.get("remote") or {}
    job_id = remote.get("active_job_id")
    if not job_id:
        return {"state": "idle", "manifest": manifest}
    try:
        job = controller.pool.job_handle(job_id)
    except Exception as exc:
        return {"state": "unreachable", "job_id": job_id, "error": str(exc)}
    if job.done:
        result = job.result() if job.state == "succeeded" else None
        return {"state": job.state, "job": job, "result": result}
    return {"state": job.state, "job": job}


def vgpu_download_artifact(
    controller: VGPUController,
    artifact_id: str,
    local_path: str | Path,
    *,
    chunk_bytes: int = VGPU_CHUNK_BYTES,
) -> dict[str, Any]:
    base = _vgpu_base_payload(controller)
    info = controller.deployment.infer({
        **base, "action": "vgpu_download_chunk", "artifact_id": artifact_id,
        "offset": 0, "length": 0,
    })
    total = int(info["size"])
    expected_sha = info["sha256"]
    output = Path(local_path)
    if output.exists():
        observed = vgpu_sha256_file(output)
        if output.stat().st_size == total and observed == expected_sha:
            return {"path": str(output), "size": total, "sha256": expected_sha}
        raise ValueError(
            f"refusing to overwrite completed artifact {output}: "
            f"{observed} != {expected_sha}"
        )
    part = output.with_name(f".{output.name}.part")
    part.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    if offset > total:
        part.unlink()
        offset = 0
    with part.open("ab") as handle:
        while offset < total:
            reply = controller.deployment.infer({
                **base, "action": "vgpu_download_chunk",
                "artifact_id": artifact_id, "offset": offset,
                "length": min(chunk_bytes, total - offset),
            })
            block = base64.b64decode(reply["data_b64"], validate=True)
            if hashlib.sha256(block).hexdigest() != reply["chunk_sha256"]:
                raise ValueError("downloaded artifact chunk failed SHA-256")
            handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
            offset += len(block)
    if vgpu_sha256_file(part) != expected_sha:
        raise ValueError("downloaded artifact failed final SHA-256")
    os.replace(part, output)
    return {"path": str(output), "size": total, "sha256": expected_sha}


def vgpu_teardown(controller: VGPUController) -> None:
    error: BaseException | None = None
    try:
        try:
            controller.deployment.infer({
                **_vgpu_base_payload(controller),
                "action": "vgpu_cleanup",
                "remove_remote_files": False,
            })
        except Exception:
            pass
        controller.deployment.teardown()
        if hasattr(controller.pool, "deployments"):
            if any(
                item.get("deploymentId") == controller.deployment.deployment_id
                for item in controller.pool.deployments()
            ):
                raise RuntimeError(
                    f"deployment {controller.deployment.deployment_id} still appears live"
                )
    except BaseException as exc:
        error = exc
    finally:
        manifest = controller.state.vgpu_load_manifest()
        if error is None:
            manifest.setdefault("remote", {}).update({
                "deployment_id": None,
                "node_id": None,
                "active_job_id": None,
                "active_partition_id": None,
                "teardown_confirmed": True,
            })
        else:
            manifest.setdefault("remote", {})["teardown_error"] = str(error)
        controller.state.vgpu_save_manifest(manifest)
    if error is not None:
        raise error


def vgpu_step9_consolidate(repo_root: str | Path) -> None:
    root = Path(repo_root).resolve()
    old_cwd = Path.cwd()
    try:
        os.chdir(root)
        runpy.run_path(
            str(root / "scripts" / "step9_dataset_compare.py"),
            run_name="__main__",
        )
    finally:
        os.chdir(old_cwd)
