# PixelPals GPU Relay

## Verified consumer guide

This guide contains one consumer setup that was executed from beginning to end.
It does not present untested alternatives.

Verification date: **July 30, 2026**

Verified `client.py` SHA-256:

```text
a68de6e03e639b77aabcccf55b36719e2702f1b9fee04cee8393302a65345fee
```

The test used the exact [client.py](../mainul-doc/client.py) supplied with this
repository as a standalone vendored file. It
connected to the live control plane, selected a Kaggle Tesla T4, ran real CUDA
work, tore down the deployment, and confirmed the deployment disappeared from
the control-plane listing.

## What is included

This repository includes the exact reviewed single-file client used by the
verified probe:

```text
mainul-doc/client.py
```

It also includes the executable probe:

```text
docs/verified_gpu_pool_probe.py
```

Do not copy this Markdown file by itself. The guide, `mainul-doc/client.py`, and
the probe program form one tested bundle.

## What you need from the pool operator

Ask for:

- the current consumer HTTPS URL;
- the consumer API key;
- confirmation that at least one Colab or Kaggle GPU node is connected.

The consumer API key is not the node token.

### Why this guide does not give a public download command

During verification, both public repository URLs named by the project returned
HTTP 404 for `gpu_pool/client.py`. A public `curl` command would therefore be a
false instruction.

Use the reviewed file included in this repository. Do not download a similarly
named package from PyPI, and do not execute an unreviewed client with your API
key.

## Step 1: confirm the tested bundle is complete

From the repository root:

```bash
test -f mainul-doc/client.py
test -f docs/verified_gpu_pool_probe.py
```

The relevant files should look like this:

```text
your-project/
├── .env.example
├── docs/
│   └── verified_gpu_pool_probe.py
└── mainul-doc/
    └── client.py
```

The probe source is [verified_gpu_pool_probe.py](verified_gpu_pool_probe.py).

## Step 2: verify the client before giving it credentials

Run:

```bash
sha256sum mainul-doc/client.py
```

For the client revision tested by this guide, the output must be:

```text
a68de6e03e639b77aabcccf55b36719e2702f1b9fee04cee8393302a65345fee  mainul-doc/client.py
```

If it differs, stop. The file may be a legitimate newer version, but this guide
does not claim that version was tested. Ask the operator for its reviewed
checksum and matching instructions.

This checksum comparison was performed during verification. The supplied file
was byte-identical to the client used in the earlier live GPU tests.

## Step 3: create `.env`

Copy the supplied example:

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env`:

```dotenv
GPU_POOL_URL=https://current-pool-host.example.com
GPU_POOL_API_KEY=your-consumer-api-key
```

Rules checked by the probe:

- `GPU_POOL_URL` must begin with `https://` or `http://`.
- A normal remote consumer uses `https://`.
- Do not use `wss://`; that scheme is used by GPU nodes.
- Do not add `/v1` to the base URL.
- The API key cannot be empty or the example placeholder.

The probe reads the key but never prints it.

`.env` is ignored by this repository. Its permission mode was checked as `600`
during verification.

## Step 4: run the verified probe

From the project root:

```bash
python3 -u docs/verified_gpu_pool_probe.py
```

The program performs these operations in order:

1. Hashes `mainul-doc/client.py` before importing it.
2. Refuses any unverified revision.
3. Imports that exact file as a standalone Python module.
4. Reads `.env`.
5. Calls the live pool status, limits, and node endpoints.
6. Selects a connected, non-draining Colab or Kaggle node with the most
   scheduler-reported free VRAM.
7. Deploys `pool_models.gpu_probe:build` with a 1 GiB reservation.
8. Runs a 2048×2048 fp16 matrix multiplication as CUDA warm-up.
9. Runs and measures a 4096×4096 fp16 matrix multiplication.
10. Requires the result to contain `cuda: true`.
11. Tears down in a `finally` block.
12. Confirms that deployment ID no longer appears in the control-plane list.

The exact program, not a hand-edited equivalent, produced:

```text
client SHA-256:       a68de6e03e639b77aabcccf55b36719e2702f1b9fee04cee8393302a65345fee
pool nodes:           1
selected GPU:         Tesla T4
GPU VRAM total:       14.56 GiB
matrix size:          4096 × 4096
matrix dtype:         fp16
CUDA time:            6.50 ms
client round trip:    0.584 seconds
CUDA result:          true
teardown:             confirmed
deployment listing:   removed
```

The final line was:

```text
VERIFIED: vendored client -> control plane -> remote CUDA -> teardown
```

If you do not see that line, do not treat the setup as working.

## What a successful probe proves

It proves that this exact path worked:

```text
vendored client
    → consumer HTTPS authentication
    → live control plane
    → node selection
    → workload deployment
    → remote Tesla T4 CUDA execution
    → JSON result
    → teardown request
    → deployment removed from control-plane listing
```

It does not prove that every other workload is correct.

## Live behavior observed beyond the probe

The following results were measured with the same client bytes before the clean
vendoring test.

### Matrix scaling and warm-up

| Operation | CUDA time | Client round trip |
|---|---:|---:|
| 2048² fp16, first CUDA call | 138.66 ms | 0.685 s |
| 4096² fp16, after warm-up | 6.69 ms | 0.523 s |
| 8192² fp16 | 51.32 ms | 0.576 s |

The first CUDA call included initialization overhead. The verified probe
therefore warms up before reporting the 4096² result.

### Real LLM

`Qwen/Qwen2.5-1.5B-Instruct` was deployed through
`pool_models.llm:build` in fp16:

| Observation | Measured value |
|---|---:|
| Scheduler reservation | 4 GiB |
| Deployment load | 40.14 s |
| Workload-reported VRAM use | 3.17 GiB |
| Blocking generation | 96 tokens in 4.948 s |
| Blocking throughput | 19.4 tokens/s |
| Job generation | 168 tokens in 6.713 s |
| Job throughput | 25.03 tokens/s |
| Job placements | 1 |

Both blocking inference and job submission completed. The job result reported
`useCuda: true` and `device: Tesla T4`.

This section reports what was observed. It is not a second setup recipe.

### Job progress

A resumable-batch workload processed 5,000 items in 6.51 seconds on one
placement. Its last published progress snapshot showed 4,750/5,000, while the
terminal result correctly reported 5,000/5,000.

Therefore, treat progress as a possibly delayed snapshot and the terminal result
as authoritative.

Actual checkpoint recovery after a node loss was not tested and is not
documented as a working procedure here.

### Workload output must be validated

The current CLIP workload loaded and executed but returned a nested embedding
reported as dimension 1 instead of a flat 512-value vector.

This means:

```text
successful HTTP request ≠ correct model output
```

Consumers must validate expected shapes, types, and required flags such as
`cuda: true`.

### Reservation release is not proof of physical VRAM reclamation

After the earlier model tests, the control plane reported:

```text
scheduler free VRAM: 14.56 GiB
scheduler reserved:   0 GiB
dirty deployments:    none
```

A fresh CUDA probe on the same node reported:

```text
physical CUDA free VRAM: 11.39 GiB
physical CUDA total:     14.56 GiB
```

The discrepancy persisted through another confirmed probe teardown. Therefore,
this guide does **not** claim that confirmed teardown reclaimed all physical
VRAM. It only confirms that the deployment disappeared and the scheduler
reservation was released.

## If the probe fails

Only the following failures were directly observed during this work.

### Client checksum error

The probe refuses to run when `client.py` does not match the tested SHA-256.

Action: obtain the reviewed file or a separately verified guide for the newer
revision. Do not remove the checksum check merely to make the program continue.

### `GPU_POOL_URL` uses `wss://`

The initial consumer `.env` contained a node WebSocket scheme. It was corrected
to the corresponding `https://` consumer URL before successful requests.

Action: obtain the current consumer HTTPS URL from the operator.

### Base endpoint returns HTTP 404

The temporary ngrok endpoint worked, later returned HTTP 404, and subsequently
worked again.

Action: stop. Ask the operator whether the tunnel/control-plane URL is current.
Do not create retry loops that submit work blindly.

### No final `VERIFIED` line

The probe prints its final success line only after CUDA succeeds, teardown is
confirmed, and the deployment disappears from the listing.

Action: read the preceding error. Do not report success based only on an earlier
line such as node discovery.

## Rules for an autonomous agent

An agent following this guide must:

1. Use only the operator-supplied client whose hash matches this guide.
2. Never print or transmit the API key except as the client's authorization
   header.
3. Refuse `ws://` and `wss://` consumer URLs.
4. Run the exact verified probe before claiming GPU access works.
5. Require `cuda: true`.
6. Keep teardown in `finally`.
7. Require the final `VERIFIED` line.
8. Treat progress snapshots as non-authoritative.
9. Validate every model's output schema.
10. Distinguish scheduler reservations from physical CUDA memory.
11. Stop and ask the operator when the base endpoint returns 404.
12. Never claim an untested workflow is supported.

## Deliberately omitted

These topics appeared in the earlier guide but were removed as procedures
because they were not verified end to end during this work:

- public client download;
- raw `curl` API usage;
- job reattachment from a second process;
- cooperative cancellation;
- checkpoint recovery after node loss;
- forced teardown;
- custom workload Git delivery and allowlist changes;
- large-file/object-storage workflows.

The underlying system may implement them. This guide makes no working claim
about them.

For protocol design details—not a guarantee that every path was exercised—see
the local [upstream consumer reference](../mainul-doc/new-gpu-system.md).
