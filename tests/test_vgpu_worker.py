import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from src.vgpu.vgpu_step9_worker import (
    VGPUStep9Worker,
    vgpu_safe_extract,
    vgpu_sha256_file,
)


def _identity(worker, run_id="run1"):
    return {
        "run_id": run_id,
        "worker_sha256": worker.worker_sha256,
        "source_sha256": "source",
        "data_manifest_sha256": "data",
    }


def test_vgpu_worker_upload_resumes_exact_offset(tmp_path):
    worker = VGPUStep9Worker(root=tmp_path)
    raw = b"abcdefghij"
    meta = {
        **_identity(worker),
        "file_id": "file1",
        "remote_path": "data/a.bin",
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert worker.infer({"action": "vgpu_begin_upload", **meta})["offset"] == 0
    first = raw[:4]
    reply = worker.infer({
        "action": "vgpu_upload_chunk", **_identity(worker),
        "file_id": "file1", "offset": 0,
        "data_b64": base64.b64encode(first).decode(),
        "chunk_sha256": hashlib.sha256(first).hexdigest(),
    })
    assert reply["offset"] == 4

    restarted = VGPUStep9Worker(root=tmp_path)
    meta["worker_sha256"] = restarted.worker_sha256
    assert restarted.infer({"action": "vgpu_begin_upload", **meta})["offset"] == 4
    rest = raw[4:]
    restarted.infer({
        "action": "vgpu_upload_chunk", **_identity(restarted),
        "file_id": "file1", "offset": 4,
        "data_b64": base64.b64encode(rest).decode(),
        "chunk_sha256": hashlib.sha256(rest).hexdigest(),
    })
    done = restarted.infer({"action": "vgpu_finish_upload", **meta})
    assert done["complete"]
    assert (tmp_path / "run1" / "data" / "a.bin").read_bytes() == raw
    assert restarted.infer({"action": "vgpu_begin_upload", **meta})["complete"]


def test_vgpu_worker_rejects_corrupt_chunk(tmp_path):
    worker = VGPUStep9Worker(root=tmp_path)
    raw = b"abc"
    meta = {
        **_identity(worker), "file_id": "file1", "remote_path": "a.bin",
        "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    }
    worker.infer({"action": "vgpu_begin_upload", **meta})
    with pytest.raises(ValueError, match="chunk failed SHA"):
        worker.infer({
            "action": "vgpu_upload_chunk", **_identity(worker),
            "file_id": "file1", "offset": 0,
            "data_b64": base64.b64encode(raw).decode(),
            "chunk_sha256": "0" * 64,
        })


def test_vgpu_worker_rejects_wrong_offset_without_writing(tmp_path):
    worker = VGPUStep9Worker(root=tmp_path)
    raw = b"abc"
    meta = {
        **_identity(worker), "file_id": "file1", "remote_path": "a.bin",
        "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    }
    worker.infer({"action": "vgpu_begin_upload", **meta})
    reply = worker.infer({
        "action": "vgpu_upload_chunk", **_identity(worker),
        "file_id": "file1", "offset": 1,
        "data_b64": base64.b64encode(raw).decode(),
        "chunk_sha256": hashlib.sha256(raw).hexdigest(),
    })
    assert reply == {"file_id": "file1", "offset": 0, "accepted": False}


def test_vgpu_safe_extract_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        handle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="escapes"):
        vgpu_safe_extract(archive, tmp_path / "out")


def test_vgpu_safe_extract_rejects_symlink(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp"
        handle.addfile(info)
    with pytest.raises(ValueError, match="unsupported"):
        vgpu_safe_extract(archive, tmp_path / "out")


def test_vgpu_worker_refuses_identity_change(tmp_path):
    worker = VGPUStep9Worker(root=tmp_path)
    worker.infer({
        "action": "vgpu_begin_upload", **_identity(worker),
        "file_id": "x", "remote_path": "x", "size": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    })
    changed = _identity(worker)
    changed["source_sha256"] = "different"
    with pytest.raises(ValueError, match="identity changed"):
        worker.infer({
            "action": "vgpu_begin_upload", **changed,
            "file_id": "y", "remote_path": "y", "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        })


def test_vgpu_worker_rejects_unsafe_file_id(tmp_path):
    worker = VGPUStep9Worker(root=tmp_path)
    with pytest.raises(ValueError, match="file_id"):
        worker.infer({
            "action": "vgpu_begin_upload", **_identity(worker),
            "file_id": "../escape", "remote_path": "safe.bin", "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        })


def test_vgpu_worker_accepts_verified_empty_file(tmp_path):
    worker = VGPUStep9Worker(root=tmp_path)
    payload = {
        **_identity(worker),
        "file_id": "empty",
        "remote_path": "data/empty.txt",
        "size": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    worker.infer({"action": "vgpu_begin_upload", **payload})
    done = worker.infer({"action": "vgpu_finish_upload", **payload})
    assert done["complete"]
    assert (tmp_path / "run1" / "data" / "empty.txt").read_bytes() == b""
