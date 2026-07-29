import json
from pathlib import Path

import pytest

from src.vgpu.state import VGPUState, vgpu_sha256_file


def test_vgpu_manifest_is_atomic_and_round_trips(tmp_path):
    state = VGPUState(tmp_path, "run-1")
    manifest = state.vgpu_load_manifest()
    manifest["configs"]["x"] = {"epoch": 2}
    state.vgpu_save_manifest(manifest)

    loaded = json.loads(state.manifest_path.read_text())
    assert loaded["configs"]["x"]["epoch"] == 2
    assert not list(state.root.glob("*.tmp"))


def test_vgpu_identity_refuses_mixed_run(tmp_path):
    state = VGPUState(tmp_path, "run-1")
    state.vgpu_assert_identity({"source_sha256": "a"})
    with pytest.raises(ValueError, match="identity changed"):
        state.vgpu_assert_identity({"source_sha256": "b"})


def test_vgpu_lock_refuses_second_controller(tmp_path):
    state = VGPUState(tmp_path, "run-1")
    with state.vgpu_lock():
        with pytest.raises(RuntimeError, match="another controller"):
            with state.vgpu_lock():
                pass


def test_vgpu_lock_recovers_dead_owner(tmp_path):
    state = VGPUState(tmp_path, "run-1")
    state.lock_path.write_text("pid=999999999 started=0\n")
    with state.vgpu_lock():
        assert state.lock_path.exists()
    assert not state.lock_path.exists()


def test_vgpu_blob_stays_below_state_root(tmp_path):
    state = VGPUState(tmp_path, "run-1")
    with pytest.raises(ValueError, match="below the run directory"):
        state.vgpu_save_blob("../escape", b"x")


def test_vgpu_blob_sha(tmp_path):
    state = VGPUState(tmp_path, "run-1")
    path = state.vgpu_save_blob("checkpoints/a.bin", b"checkpoint")
    assert vgpu_sha256_file(path) == (
        "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )


def test_vgpu_completed_blob_only_accepts_identical_bytes(tmp_path):
    state = VGPUState(tmp_path, "run-1")
    state.vgpu_commit_blob("evaluation/a.json", b"same")
    state.vgpu_commit_blob("evaluation/a.json", b"same")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        state.vgpu_commit_blob("evaluation/a.json", b"different")
