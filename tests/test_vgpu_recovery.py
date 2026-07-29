import base64
import hashlib

import pytest

from src.vgpu.controller import (
    VGPUController,
    vgpu_download_artifact,
    vgpu_submit_partition,
    vgpu_teardown,
)
from src.vgpu.state import VGPUState
from src.vgpu.vgpu_step9_worker import VGPUStep9Worker


class _Job:
    def __init__(self, job_id, result, state="running"):
        self.job_id = job_id
        self._result = result
        self.state = state
        self.progress = {}
        self.done = state in {"succeeded", "failed", "canceled"}

    def wait(self, on_progress=None):
        self.state = "succeeded"
        self.done = True
        return self._result

    def result(self):
        return self._result


class _Pool:
    def __init__(self, old_job=None):
        self.old_job = old_job

    def job_handle(self, job_id):
        assert self.old_job.job_id == job_id
        return self.old_job

    def deployments(self):
        return []


class _Deployment:
    deployment_id = "dep"
    node_id = "node"

    def __init__(self, new_job=None):
        self.new_job = new_job
        self.submit_count = 0
        self.torn_down = False

    def submit(self, payload, **kwargs):
        self.submit_count += 1
        return self.new_job

    def infer(self, payload):
        if payload["action"] == "vgpu_cleanup":
            return {"cleaned": True}
        raise AssertionError(payload)

    def teardown(self):
        self.torn_down = True


def _controller(tmp_path, pool, deployment):
    return VGPUController(
        repo_root=tmp_path,
        state=VGPUState(tmp_path, "run"),
        pool=pool,
        deployment=deployment,
        worker_sha256="worker",
        source_sha256="source",
        data_manifest_sha256="data",
    )


def test_vgpu_controller_reattaches_same_running_partition(tmp_path):
    old = _Job("old", {"value": 7})
    pool = _Pool(old)
    deployment = _Deployment(new_job=_Job("new", {"value": 9}))
    controller = _controller(tmp_path, pool, deployment)
    manifest = controller.state.vgpu_load_manifest()
    manifest["remote"] = {
        "active_job_id": "old",
        "active_partition_id": "config:train:1",
    }
    controller.state.vgpu_save_manifest(manifest)

    result = vgpu_submit_partition(
        controller,
        {"action": "vgpu_train_epoch", "config_name": "config"},
        partition_id="config:train:1",
    )
    assert result == {"value": 7}
    assert deployment.submit_count == 0
    assert controller.state.vgpu_load_manifest()["remote"]["active_job_id"] is None


def test_vgpu_controller_resolves_old_partition_before_new_one(tmp_path):
    old = _Job("old", {"old": True})
    new = _Job("new", {"new": True})
    pool = _Pool(old)
    deployment = _Deployment(new_job=new)
    controller = _controller(tmp_path, pool, deployment)
    manifest = controller.state.vgpu_load_manifest()
    manifest["remote"] = {
        "active_job_id": "old",
        "active_partition_id": "config:train:1",
    }
    controller.state.vgpu_save_manifest(manifest)

    result = vgpu_submit_partition(
        controller,
        {"action": "vgpu_train_epoch", "config_name": "config"},
        partition_id="config:train:2",
    )
    assert result == {"new": True}
    assert deployment.submit_count == 1


def test_vgpu_worker_replays_cached_training_state(tmp_path):
    worker = VGPUStep9Worker(root=tmp_path)
    identity = {
        "run_id": "run",
        "worker_sha256": worker.worker_sha256,
        "source_sha256": "source",
        "data_manifest_sha256": "data",
    }

    class Session:
        calls = 0
        restored = 0

        def vgpu_train_epoch(self, ctx=None):
            self.calls += 1
            raw = b"compact"
            return {
                "checkpoint_b64": base64.b64encode(raw).decode(),
                "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
            }

        def vgpu_restore_checkpoint(self, raw):
            assert raw == b"compact"
            self.restored += 1

    session = Session()
    worker._sessions[("run", "config")] = session
    payload = {
        **identity,
        "action": "vgpu_train_epoch",
        "config_name": "config",
        "partition_id": "config:train:1",
    }
    first = worker.infer(payload)
    second = worker.infer(payload)
    assert first == second
    assert session.calls == 1
    assert session.restored == 1


def test_vgpu_teardown_failure_is_recorded(tmp_path):
    class Broken(_Deployment):
        def teardown(self):
            raise RuntimeError("unload failed")

    controller = _controller(tmp_path, _Pool(), Broken())
    with pytest.raises(RuntimeError, match="unload failed"):
        vgpu_teardown(controller)
    remote = controller.state.vgpu_load_manifest()["remote"]
    assert remote["teardown_error"] == "unload failed"
    assert remote.get("teardown_confirmed") is not True


def test_vgpu_download_rejects_corrupt_chunk(tmp_path):
    raw = b"full checkpoint"

    class Download(_Deployment):
        def infer(self, payload):
            if payload["action"] == "vgpu_download_chunk":
                if payload["length"] == 0:
                    return {
                        "size": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                return {
                    "data_b64": base64.b64encode(raw).decode(),
                    "chunk_sha256": "0" * 64,
                }
            return super().infer(payload)

    controller = _controller(tmp_path, _Pool(), Download())
    with pytest.raises(ValueError, match="chunk failed SHA"):
        vgpu_download_artifact(controller, "artifact", tmp_path / "model.pt")
