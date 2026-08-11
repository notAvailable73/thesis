"""Additive virtual-GPU support for the locally controlled Step 9 v2 run.

Nothing in this package is imported by the original training/evaluation paths.
The ``vgpu_`` prefix is intentional: it keeps every new public callable
visibly separate from the frozen Steps 1-9 implementation.
"""

from .controller import (
    VGPUController,
    VGPU_RUNS,
    vgpu_build_data_manifest,
    vgpu_build_source_bundle,
    vgpu_connect_worker,
    vgpu_download_artifact,
    vgpu_resume_run,
    vgpu_reattach_worker,
    vgpu_step9_consolidate,
    vgpu_submit_partition,
    vgpu_teardown,
    vgpu_upload_file,
    vgpu_verify_client,
)
from .state import VGPUState, vgpu_sha256_file
from .workflow import (
    vgpu_bootstrap_cell,
    vgpu_close,
    vgpu_create_controller,
    vgpu_finalize_manifest,
    vgpu_run_config,
    vgpu_run_integration_gate,
    vgpu_run_step9,
    vgpu_resolve_data_dir,
    vgpu_stage_run,
    vgpu_worker_sha256,
)

__all__ = [
    "VGPUController",
    "VGPUState",
    "VGPU_RUNS",
    "vgpu_build_data_manifest",
    "vgpu_build_source_bundle",
    "vgpu_bootstrap_cell",
    "vgpu_close",
    "vgpu_connect_worker",
    "vgpu_create_controller",
    "vgpu_download_artifact",
    "vgpu_finalize_manifest",
    "vgpu_resume_run",
    "vgpu_reattach_worker",
    "vgpu_run_config",
    "vgpu_run_integration_gate",
    "vgpu_run_step9",
    "vgpu_resolve_data_dir",
    "vgpu_sha256_file",
    "vgpu_step9_consolidate",
    "vgpu_submit_partition",
    "vgpu_stage_run",
    "vgpu_teardown",
    "vgpu_upload_file",
    "vgpu_verify_client",
    "vgpu_worker_sha256",
]
