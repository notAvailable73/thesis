"""Verified consumer smoke test for gpu-pool.

Run from the directory containing .env:

    python3 -u docs/verified_gpu_pool_probe.py

The API key is read but never printed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

EXPECTED_CLIENT_SHA256 = (
    "a68de6e03e639b77aabcccf55b36719e2702f1b9fee04cee8393302a65345fee"
)
def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    client_path = Path("mainul-doc/client.py").resolve()
    actual_client_hash = file_sha256(client_path)
    if actual_client_hash != EXPECTED_CLIENT_SHA256:
        raise RuntimeError(
            "Unverified gpu_pool/client.py. Expected SHA-256 "
            f"{EXPECTED_CLIENT_SHA256}, got {actual_client_hash}."
        )
    spec = importlib.util.spec_from_file_location(
        "verified_gpu_pool_client", client_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load verified client from {client_path}")
    client_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = client_module
    spec.loader.exec_module(client_module)
    PoolClient = client_module.PoolClient

    values = load_env(Path(".env"))
    pool_url = values.get("GPU_POOL_URL", "").rstrip("/")
    api_key = values.get("GPU_POOL_API_KEY", "")
    if not pool_url.startswith(("https://", "http://")):
        raise RuntimeError("GPU_POOL_URL must start with https:// or http://")
    if not api_key or "replace-with" in api_key:
        raise RuntimeError("GPU_POOL_API_KEY is missing or still a placeholder")

    pool = PoolClient(pool_url, api_key=api_key, timeout=30)
    status = pool.status()
    limits = pool.limits()
    nodes = [
        node
        for node in pool.nodes()
        if node.get("provider") in {"colab", "kaggle"}
        and not node.get("draining", False)
    ]
    if not nodes:
        raise RuntimeError("No non-draining Colab/Kaggle node is connected")
    node = max(nodes, key=lambda item: float(item.get("freeVramGib", 0)))

    print(
        json.dumps(
            {
                "verifiedClient": str(client_path),
                "clientSha256": actual_client_hash,
                "poolNodeCount": status["nodeCount"],
                "inferTimeoutSeconds": limits["inferTimeoutSeconds"],
                "loadTimeoutSeconds": limits["loadTimeoutSeconds"],
                "selectedNode": {
                    "nodeId": node["nodeId"],
                    "provider": node["provider"],
                    "gpuName": node["gpuName"],
                    "freeVramGib": node["freeVramGib"],
                    "quotaRemainingSeconds": node.get("quotaRemainingSeconds"),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    deployment = pool.deploy(
        {"kind": "python", "entrypoint": "pool_models.gpu_probe:build"},
        mem_gib=1,
        node_id=node["nodeId"],
    )
    try:
        deployment.infer({"size": 2048})  # CUDA warm-up
        started = time.perf_counter()
        result = deployment.infer({"size": 4096})
        result["roundTripSeconds"] = round(time.perf_counter() - started, 3)
        if result.get("cuda") is not True:
            raise RuntimeError(f"Remote node did not execute with CUDA: {result}")
        print(json.dumps({"probe": result}, indent=2, sort_keys=True), flush=True)
    finally:
        deployment.teardown()
        print(f"Teardown confirmed for {deployment.deployment_id}", flush=True)

    if any(
        item["deploymentId"] == deployment.deployment_id
        for item in pool.deployments()
    ):
        raise RuntimeError("Deployment still appears in the control-plane listing")
    print("VERIFIED: vendored client -> control plane -> remote CUDA -> teardown")


if __name__ == "__main__":
    main()
