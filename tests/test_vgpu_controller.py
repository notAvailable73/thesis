import hashlib
import base64
import re
import zlib
from pathlib import Path

from src.vgpu.aggregation import vgpu_validate_shards
from src.vgpu.controller import (
    VGPUController,
    vgpu_build_data_manifest,
    vgpu_build_source_bundle,
    vgpu_teardown,
    vgpu_upload_file,
)
from src.vgpu.state import VGPUState
from src.vgpu.vgpu_step9_worker import VGPUStep9Worker
from src.vgpu.workflow import (
    vgpu_bootstrap_cell,
    vgpu_resolve_data_dir,
    vgpu_worker_sha256,
)


class _Deployment:
    deployment_id = "dep-1"
    node_id = "node-1"

    def __init__(self, worker):
        self.worker = worker
        self.torn_down = False

    def infer(self, payload):
        return self.worker.infer(payload)

    def teardown(self):
        self.torn_down = True


def _controller(tmp_path):
    worker = VGPUStep9Worker(root=tmp_path / "remote")
    state = VGPUState(tmp_path, "run1")
    return VGPUController(
        repo_root=tmp_path,
        state=state,
        pool=object(),
        deployment=_Deployment(worker),
        worker_sha256=worker.worker_sha256,
        source_sha256="source",
        data_manifest_sha256="data",
    )


def test_vgpu_controller_upload_resumes(tmp_path):
    controller = _controller(tmp_path)
    source = tmp_path / "large.bin"
    source.write_bytes(b"0123456789")
    partial = vgpu_upload_file(
        controller, source, "data/large.bin",
        file_id="large", chunk_bytes=4, stop_after_chunks=1,
    )
    assert partial["offset"] == 4 and not partial["complete"]
    complete = vgpu_upload_file(
        controller, source, "data/large.bin",
        file_id="large", chunk_bytes=4,
    )
    assert complete["complete"]
    assert (
        tmp_path / "remote" / "run1" / "data" / "large.bin"
    ).read_bytes() == source.read_bytes()


def test_vgpu_source_bundle_excludes_heavy_roots(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "large").write_bytes(b"x")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "old.json").write_text("{}")
    bundle = vgpu_build_source_bundle(tmp_path, tmp_path / "state" / "source.tar.gz")
    assert bundle["file_count"] == 1
    assert bundle["sha256"]


def test_vgpu_data_manifest_is_stable(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "b").write_bytes(b"2")
    (data / "a").write_bytes(b"1")
    first = vgpu_build_data_manifest(data)
    second = vgpu_build_data_manifest(data)
    assert first["sha256"] == second["sha256"]
    assert [item["relative_path"] for item in first["files"]] == ["a", "b"]


def test_vgpu_data_manifest_excludes_controller_state(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "real.bin").write_bytes(b"data")
    state = data / "vgpu_step9_state" / "run"
    state.mkdir(parents=True)
    (state / "source.tar.gz").write_bytes(b"controller")
    manifest = vgpu_build_data_manifest(data)
    assert [item["relative_path"] for item in manifest["files"]] == ["real.bin"]


def test_vgpu_data_dir_defaults_to_repository_data(tmp_path):
    expected = tmp_path / "data"
    expected.mkdir()
    assert vgpu_resolve_data_dir(tmp_path) == expected.resolve()


def test_vgpu_bootstrap_embeds_exact_worker(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    cell = vgpu_bootstrap_cell(repo)
    assert vgpu_worker_sha256(repo) in cell
    assert "pool_models\" / \"vgpu_step9.py" in cell
    assert "git\", \"clone\"" in cell
    encoded = re.search(r'b64decode\("([A-Za-z0-9+/=]+)"\)', cell).group(1)
    assert zlib.decompress(base64.b64decode(encoded)) == (
        repo / "src" / "vgpu" / "vgpu_step9_worker.py"
    ).read_bytes()


def test_vgpu_teardown_clears_remote_state(tmp_path):
    controller = _controller(tmp_path)
    controller.state.vgpu_save_manifest({
        **controller.state.vgpu_load_manifest(),
        "remote": {"deployment_id": "dep-1", "node_id": "node-1"},
    })
    vgpu_teardown(controller)
    assert controller.deployment.torn_down
    assert controller.state.vgpu_load_manifest()["remote"]["deployment_id"] is None


def test_vgpu_shard_validation_detects_missing_and_duplicates():
    good = [{"seeds": [0, 1]}, {"seeds": [2, 3]}]
    assert len(vgpu_validate_shards(good, [0, 1, 2, 3])) == 2
    try:
        vgpu_validate_shards([{"seeds": [0, 1]}, {"seeds": [1, 2]}], [0, 1, 2])
    except ValueError as exc:
        assert "do not match" in str(exc)
    else:
        raise AssertionError("duplicate seed was accepted")
