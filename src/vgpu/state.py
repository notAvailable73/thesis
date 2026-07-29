"""Local, atomic state for Step 9 v2.

The local manifest is authoritative.  Remote jobs may disappear; a committed
local epoch checkpoint or evaluation shard never does.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

VGPU_STATE_VERSION = 1


def vgpu_sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def vgpu_atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def vgpu_atomic_write_json(path: str | Path, payload: Any) -> Path:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return vgpu_atomic_write_bytes(path, encoded)


def vgpu_mirror_file(path: str | Path, drive_dir: str | Path | None) -> None:
    if not drive_dir:
        return
    source = Path(path)
    destination = Path(drive_dir) / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


class VGPUState:
    """Atomic manifest/checkpoint/shard store with a single-controller lock."""

    def __init__(
        self,
        repo_root: str | Path,
        run_id: str,
        *,
        drive_dir: str | Path | None = None,
    ) -> None:
        if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
            raise ValueError("run_id must contain only letters, digits, '-' and '_'")
        self.repo_root = Path(repo_root).resolve()
        self.run_id = run_id
        self.root = self.repo_root / "data" / "vgpu_step9_state" / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.drive_dir = Path(drive_dir).resolve() / run_id if drive_dir else None
        self.manifest_path = self.root / "manifest.json"
        self.lock_path = self.root / ".controller.lock"

    def vgpu_load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "version": VGPU_STATE_VERSION,
                "run_id": self.run_id,
                "created_unix": time.time(),
                "updated_unix": time.time(),
                "identity": {},
                "remote": {},
                "uploads": {},
                "configs": {},
                "artifacts": {},
            }
        with self.manifest_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("version") != VGPU_STATE_VERSION:
            raise ValueError(
                f"unsupported vGPU state version {data.get('version')!r}; "
                f"expected {VGPU_STATE_VERSION}"
            )
        if data.get("run_id") != self.run_id:
            raise ValueError("manifest run_id does not match state directory")
        return data

    def vgpu_save_manifest(self, manifest: dict[str, Any]) -> None:
        manifest = dict(manifest)
        manifest["version"] = VGPU_STATE_VERSION
        manifest["run_id"] = self.run_id
        manifest["updated_unix"] = time.time()
        vgpu_atomic_write_json(self.manifest_path, manifest)
        if self.drive_dir:
            vgpu_mirror_file(self.manifest_path, self.drive_dir)

    def vgpu_assert_identity(self, expected: dict[str, str]) -> dict[str, Any]:
        manifest = self.vgpu_load_manifest()
        current = manifest.get("identity") or {}
        if current and current != expected:
            raise ValueError(
                "run identity changed; start a new run_id instead of mixing "
                f"partitions. current={current}, expected={expected}"
            )
        if not current:
            manifest["identity"] = dict(expected)
            self.vgpu_save_manifest(manifest)
        return manifest

    def vgpu_save_blob(self, relative_path: str | Path, payload: bytes) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("state blob path must stay below the run directory")
        path = vgpu_atomic_write_bytes(self.root / relative, payload)
        if self.drive_dir:
            mirror = self.drive_dir / relative
            mirror.parent.mkdir(parents=True, exist_ok=True)
            temporary = mirror.with_name(f".{mirror.name}.{os.getpid()}.tmp")
            shutil.copy2(path, temporary)
            os.replace(temporary, mirror)
        return path

    def vgpu_commit_blob(self, relative_path: str | Path, payload: bytes) -> Path:
        """Commit an immutable partition result, accepting byte-identical repeats."""

        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("state blob path must stay below the run directory")
        path = self.root / relative
        if path.exists():
            existing = vgpu_sha256_file(path)
            incoming = hashlib.sha256(payload).hexdigest()
            if existing != incoming:
                raise ValueError(
                    f"refusing to overwrite completed state blob {relative}: "
                    f"{existing} != {incoming}"
                )
            return path
        return self.vgpu_save_blob(relative, payload)

    @contextlib.contextmanager
    def vgpu_lock(self) -> Iterator[None]:
        try:
            fd = os.open(
                self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as exc:
            owner = self.lock_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            pid = None
            for token in owner.split():
                if token.startswith("pid="):
                    try:
                        pid = int(token.split("=", 1)[1])
                    except ValueError:
                        pass
            stale = False
            if pid is not None:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    stale = True
                except PermissionError:
                    stale = False
            if stale:
                self.lock_path.unlink(missing_ok=True)
                fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            else:
                raise RuntimeError(
                    f"another controller owns run {self.run_id!r} "
                    f"({owner or 'unknown owner'})"
                ) from exc
        try:
            os.write(fd, f"pid={os.getpid()} started={time.time()}\n".encode())
            os.close(fd)
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)
