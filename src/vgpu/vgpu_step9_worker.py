"""Standalone temporary Kaggle worker for Step 9 v2.

This file is copied *untracked* to ``pool_models/vgpu_step9.py`` by the
bootstrap cell.  It deliberately has no imports from the thesis package until
the locally-built source bundle has been uploaded and verified.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable

VGPU_WORKER_VERSION = 1
VGPU_MAX_CHUNK_BYTES = 24 * 1024 * 1024


def vgpu_sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def vgpu_atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def vgpu_safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def vgpu_safe_token(value: str, field: str) -> str:
    token = str(value)
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    if not token or any(char not in allowed for char in token):
        raise ValueError(f"{field} must contain only letters, digits, '-' and '_'")
    return token


def vgpu_safe_extract(archive_path: str | Path, destination: str | Path) -> None:
    """Extract regular files/directories only, beneath ``destination``."""

    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"archive path escapes destination: {member.name!r}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"archive contains unsupported entry: {member.name!r}")
            resolved = (root / member_path).resolve()
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"archive path escapes destination: {member.name!r}")
        archive.extractall(root)


class VGPUStep9Worker:
    def __init__(
        self,
        *,
        expected_sha256: str | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.worker_sha256 = vgpu_sha256_file(__file__)
        if expected_sha256 and self.worker_sha256 != expected_sha256:
            raise RuntimeError(
                f"worker SHA mismatch: {self.worker_sha256} != {expected_sha256}"
            )
        if root is None:
            if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
                root = "/kaggle/working/vgpu_step9"
            else:
                root = Path(tempfile.gettempdir()) / "vgpu_step9"
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[tuple[str, str], Any] = {}

    def infer(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("vgpu payload must be a JSON object")
        action = str(payload.get("action", ""))
        if not action.startswith("vgpu_"):
            raise ValueError(f"invalid vgpu action: {action!r}")
        method = getattr(self, action, None)
        if not callable(method):
            raise ValueError(f"unknown vgpu action: {action}")
        if action not in {"vgpu_status", "vgpu_run_identity"}:
            self._vgpu_verify_run_identity(payload)
        return method(payload, ctx=ctx)

    def _vgpu_workspace(self, run_id: str) -> Path:
        safe = vgpu_safe_token(run_id, "run_id")
        workspace = self.root / safe
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _vgpu_verify_run_identity(self, payload: dict[str, Any]) -> None:
        required = ("run_id", "worker_sha256", "source_sha256", "data_manifest_sha256")
        missing = [name for name in required if not payload.get(name)]
        if missing:
            raise ValueError(f"missing run identity fields: {missing}")
        if payload["worker_sha256"] != self.worker_sha256:
            raise ValueError("payload worker_sha256 does not match loaded worker")
        workspace = self._vgpu_workspace(str(payload["run_id"]))
        identity_path = workspace / "identity.json"
        incoming = {name: str(payload[name]) for name in required[1:]}
        if identity_path.exists():
            current = json.loads(identity_path.read_text(encoding="utf-8"))
            if current != incoming:
                raise ValueError(
                    "remote run identity changed; use a new run_id instead of "
                    "mixing source/data partitions"
                )
        else:
            vgpu_atomic_json(identity_path, incoming)

    def vgpu_status(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        cuda = False
        device = "cpu"
        free_gib = total_gib = 0.0
        try:
            import torch
            cuda = bool(torch.cuda.is_available())
            if cuda:
                device = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info()
                free_gib = round(free / 1024**3, 2)
                total_gib = round(total / 1024**3, 2)
        except Exception:
            pass
        runs = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        return {
            "worker_version": VGPU_WORKER_VERSION,
            "worker_sha256": self.worker_sha256,
            "cuda": cuda,
            "device": device,
            "vram_free_gib": free_gib,
            "vram_total_gib": total_gib,
            "root": str(self.root),
            "runs": runs,
        }

    def vgpu_run_identity(
        self, payload: dict[str, Any], ctx: Any = None
    ) -> dict[str, Any]:
        workspace = self._vgpu_workspace(str(payload["run_id"]))
        identity_path = workspace / "identity.json"
        identity = (
            json.loads(identity_path.read_text(encoding="utf-8"))
            if identity_path.exists() else None
        )
        return {"run_id": payload["run_id"], "identity": identity}

    def vgpu_begin_upload(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        workspace = self._vgpu_workspace(str(payload["run_id"]))
        file_id = vgpu_safe_token(payload["file_id"], "file_id")
        remote = vgpu_safe_relative(str(payload["remote_path"]))
        expected = {
            "file_id": file_id,
            "remote_path": remote.as_posix(),
            "size": int(payload["size"]),
            "sha256": str(payload["sha256"]),
        }
        uploads = workspace / ".uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        meta_path = uploads / f"{file_id}.json"
        part_path = uploads / f"{file_id}.part"
        final_path = workspace / remote
        if final_path.exists():
            if (
                final_path.stat().st_size == expected["size"]
                and vgpu_sha256_file(final_path) == expected["sha256"]
            ):
                return {**expected, "offset": expected["size"], "complete": True}
            raise ValueError(f"remote destination exists with wrong content: {remote}")
        if meta_path.exists():
            current = json.loads(meta_path.read_text(encoding="utf-8"))
            if current != expected:
                raise ValueError(f"file_id {file_id!r} was already used for another file")
        else:
            vgpu_atomic_json(meta_path, expected)
        offset = part_path.stat().st_size if part_path.exists() else 0
        if offset > expected["size"]:
            part_path.unlink()
            offset = 0
        return {**expected, "offset": offset, "complete": False}

    def vgpu_upload_chunk(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        workspace = self._vgpu_workspace(str(payload["run_id"]))
        file_id = vgpu_safe_token(payload["file_id"], "file_id")
        uploads = workspace / ".uploads"
        meta_path = uploads / f"{file_id}.json"
        part_path = uploads / f"{file_id}.part"
        if not meta_path.exists():
            raise FileNotFoundError(f"upload {file_id!r} was not begun")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        offset = part_path.stat().st_size if part_path.exists() else 0
        requested = int(payload["offset"])
        if requested != offset:
            return {"file_id": file_id, "offset": offset, "accepted": False}
        block = base64.b64decode(str(payload["data_b64"]).encode(), validate=True)
        if len(block) > VGPU_MAX_CHUNK_BYTES:
            raise ValueError("vgpu upload chunk exceeds worker limit")
        if hashlib.sha256(block).hexdigest() != payload["chunk_sha256"]:
            raise ValueError("vgpu upload chunk failed SHA-256")
        if offset + len(block) > int(meta["size"]):
            raise ValueError("vgpu upload chunk exceeds declared file size")
        with part_path.open("ab") as handle:
            handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        return {
            "file_id": file_id,
            "offset": offset + len(block),
            "accepted": True,
        }

    def vgpu_finish_upload(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        workspace = self._vgpu_workspace(str(payload["run_id"]))
        file_id = vgpu_safe_token(payload["file_id"], "file_id")
        uploads = workspace / ".uploads"
        meta_path = uploads / f"{file_id}.json"
        part_path = uploads / f"{file_id}.part"
        requested = {
            "file_id": file_id,
            "remote_path": str(payload["remote_path"]),
            "size": int(payload["size"]),
            "sha256": str(payload["sha256"]),
        }
        final_path = workspace / vgpu_safe_relative(requested["remote_path"])
        if final_path.exists():
            if (
                final_path.stat().st_size == requested["size"]
                and vgpu_sha256_file(final_path) == requested["sha256"]
            ):
                return {**requested, "offset": requested["size"], "complete": True}
            raise ValueError("existing final upload has the wrong content")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta != requested:
            raise ValueError("finish-upload metadata differs from begin-upload")
        if meta["size"] == 0 and not part_path.exists():
            part_path.touch()
        if not part_path.exists() or part_path.stat().st_size != meta["size"]:
            raise ValueError("uploaded file size is incomplete")
        if vgpu_sha256_file(part_path) != meta["sha256"]:
            raise ValueError("uploaded file failed final SHA-256")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part_path, final_path)
        meta_path.unlink(missing_ok=True)
        return {**meta, "offset": meta["size"], "complete": True}

    def vgpu_prepare_run(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        workspace = self._vgpu_workspace(str(payload["run_id"]))
        archive_rel = vgpu_safe_relative(str(payload.get("source_archive", "source.tar.gz")))
        archive = workspace / archive_rel
        if not archive.is_file():
            raise FileNotFoundError(f"source archive was not uploaded: {archive_rel}")
        if vgpu_sha256_file(archive) != payload["source_sha256"]:
            raise ValueError("source archive SHA does not match run identity")
        source = workspace / "source"
        marker = source / ".vgpu_source_sha256"
        if marker.exists() and marker.read_text().strip() == payload["source_sha256"]:
            prepared = False
        else:
            if source.exists():
                shutil.rmtree(source)
            source.mkdir(parents=True)
            vgpu_safe_extract(archive, source)
            marker.write_text(str(payload["source_sha256"]) + "\n", encoding="utf-8")
            prepared = True
        data = workspace / "data"
        data.mkdir(exist_ok=True)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        return {
            "prepared": prepared,
            "source": str(source),
            "data": str(data),
            "source_sha256": payload["source_sha256"],
        }

    def _vgpu_runtime(self, run_id: str):
        workspace = self._vgpu_workspace(run_id)
        source = workspace / "source"
        if not source.is_dir():
            raise RuntimeError("vgpu_prepare_run must complete first")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        return importlib.import_module("src.vgpu.runtime")

    def _vgpu_cached_partition(
        self,
        payload: dict[str, Any],
        callback: Callable[[], dict[str, Any]],
        replay: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        workspace = self._vgpu_workspace(str(payload["run_id"]))
        partition_id = str(payload.get("partition_id") or "")
        if not partition_id:
            return callback()
        safe_id = hashlib.sha256(partition_id.encode()).hexdigest()
        cache = workspace / ".partitions" / f"{safe_id}.json"
        fingerprint = hashlib.sha256(
            json.dumps(
                {key: value for key, value in payload.items() if key != "checkpoint_b64"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if cache.exists():
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved["fingerprint"] != fingerprint:
                raise ValueError("partition_id was reused with a different payload")
            result = saved["result"]
            if replay is not None:
                replay(result)
            return result
        result = callback()
        vgpu_atomic_json(cache, {"fingerprint": fingerprint, "result": result})
        return result

    def vgpu_initialize_config(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        run_id = str(payload["run_id"])
        config_name = vgpu_safe_token(payload["config_name"], "config_name")
        runtime = self._vgpu_runtime(run_id)
        workspace = self._vgpu_workspace(run_id)
        checkpoint = payload.get("checkpoint_b64")
        session = runtime.vgpu_create_session(
            workspace=workspace,
            config_name=config_name,
            identity={
                "worker_sha256": payload["worker_sha256"],
                "source_sha256": payload["source_sha256"],
                "data_manifest_sha256": payload["data_manifest_sha256"],
            },
            checkpoint_bytes=(
                base64.b64decode(checkpoint, validate=True) if checkpoint else None
            ),
            k_shot_override=(
                int(payload["k_shot_override"])
                if payload.get("k_shot_override") is not None else None
            ),
        )
        self._sessions[(run_id, config_name)] = session
        return session.vgpu_status()

    def _vgpu_session(self, payload: dict[str, Any]):
        key = (str(payload["run_id"]), str(payload["config_name"]))
        session = self._sessions.get(key)
        if session is None:
            raise RuntimeError(
                f"config session {key[1]!r} is not initialized; call "
                "vgpu_initialize_config (with the latest local checkpoint after recovery)"
            )
        return session

    def vgpu_train_epoch(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        session = self._vgpu_session(payload)

        def replay(result: dict[str, Any]) -> None:
            checkpoint = base64.b64decode(
                result["checkpoint_b64"], validate=True
            )
            if hashlib.sha256(checkpoint).hexdigest() != result["checkpoint_sha256"]:
                raise ValueError("cached compact checkpoint failed SHA-256")
            session.vgpu_restore_checkpoint(checkpoint)

        return self._vgpu_cached_partition(
            payload,
            lambda: session.vgpu_train_epoch(ctx=ctx),
            replay=replay,
        )

    def vgpu_fit_temperature(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        session = self._vgpu_session(payload)
        return self._vgpu_cached_partition(
            payload,
            lambda: session.vgpu_fit_temperature(ctx=ctx),
            replay=lambda result: setattr(
                session, "temperature", float(result["temperature"])
            ),
        )

    def vgpu_evaluate_shard(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        start = int(payload["seed_start"])
        count = int(payload.get("seed_count", 50))
        return self._vgpu_cached_partition(
            payload,
            lambda: self._vgpu_session(payload).vgpu_evaluate_shard(
                seed_start=start, seed_count=count, ctx=ctx
            ),
        )

    def vgpu_export_checkpoint(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        result = self._vgpu_cached_partition(
            payload,
            lambda: self._vgpu_session(payload).vgpu_export_checkpoint(),
        )
        return result

    def vgpu_download_chunk(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        workspace = self._vgpu_workspace(str(payload["run_id"]))
        artifact_id = str(payload["artifact_id"])
        artifacts = workspace / "artifacts"
        index_path = artifacts / "index.json"
        if not index_path.exists():
            raise FileNotFoundError("no exported artifacts")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if artifact_id not in index:
            raise FileNotFoundError(f"unknown artifact_id: {artifact_id}")
        info = index[artifact_id]
        path = workspace / vgpu_safe_relative(info["relative_path"])
        offset = int(payload.get("offset", 0))
        length = int(payload.get("length", 0))
        response = {
            "artifact_id": artifact_id,
            "size": int(info["size"]),
            "sha256": str(info["sha256"]),
            "offset": offset,
        }
        if length <= 0:
            return response
        if length > VGPU_MAX_CHUNK_BYTES:
            raise ValueError("download chunk exceeds worker limit")
        with path.open("rb") as handle:
            handle.seek(offset)
            block = handle.read(length)
        return {
            **response,
            "data_b64": base64.b64encode(block).decode("ascii"),
            "chunk_sha256": hashlib.sha256(block).hexdigest(),
            "next_offset": offset + len(block),
        }

    def vgpu_cleanup(self, payload: dict[str, Any], ctx: Any = None) -> dict[str, Any]:
        run_id = str(payload["run_id"])
        for key in [key for key in self._sessions if key[0] == run_id]:
            self._sessions.pop(key, None)
        if bool(payload.get("remove_remote_files", False)):
            shutil.rmtree(self._vgpu_workspace(run_id), ignore_errors=True)
        return {"cleaned": True, "run_id": run_id}


def vgpu_build(
    expected_sha256: str | None = None,
    root: str | None = None,
) -> VGPUStep9Worker:
    return VGPUStep9Worker(expected_sha256=expected_sha256, root=root)
