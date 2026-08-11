# Consumer integration guide

This is the guide for **any application integrating with an already-running pool** —
Python or not. If you're setting up the control plane itself, see
[`docs/SETUP.md`](SETUP.md) instead.

## What the pool is

`gpu-pool` aggregates free-tier GPU capacity (Colab/Kaggle notebooks, or any GPU box that
dials in) into one allocatable pool behind a single HTTP API. You deploy a model onto it
(`mem_gib` describes how much VRAM it needs), then either call it synchronously
(`infer()`) or hand it long-running work as a job (`submit()`/`run()`) that survives the
node it started on dying.

### The one hard rule

The pool scales **out**, not **up**: it gives you aggregate allocatable capacity across
many nodes, but it does not fuse multiple GPUs into one bigger card. **Each individual
model must fit on a single node** (~15 GB on a Colab T4, ~30 GB on Kaggle's 2×T4).
Spreading one oversized model across internet-separated GPUs is a different, much
slower problem (Petals-style tensor sharding) and is out of scope here. Everything
typical — CLIP, Whisper/Conformer STT, Piper TTS, a 7B LLM in 4-bit, a LoRA fine-tune —
fits one node comfortably.

## Integration options

**Option A — vendor `gpu_pool/client.py` (recommended for Python).** It's a single,
dependency-free file (stdlib `urllib`/`json` only) with no imports from the rest of this
repo. `gpu_pool` itself is **not published to PyPI** — every consumer in this repo
path-imports it (`sys.path.insert(0, ...)`) rather than `pip install`ing it, so copying
this one file into your own project is the actual supported integration path, not a
workaround. Drop `client.py` in, `from client import PoolClient` (or keep the package
name), done.

**Option B — raw HTTP, any language.** Every route is plain JSON over HTTPS (plus one
WebSocket endpoint that's node-facing only, never called by a consumer). `GET /v1/errors`
serves the whole error registry as data, so a non-Python caller can codegen a Go/TS/Java
enum from it instead of hardcoding string codes. See the curl quickstart below.

## Auth — two separate secrets, never confuse them

| secret | who holds it | where it goes | env var (operator sets it) |
|---|---|---|---|
| **API key** | your backend | `Authorization: Bearer <key>` on every `/v1/*` call | `GPU_POOL_API_KEY` |
| **Node token** | a Colab/Kaggle notebook | the WebSocket join handshake, and the node's own checkpoint-store HTTP client | `GPU_POOL_NODE_TOKEN` |

These guard different, non-overlapping surfaces. A backend never needs the node token;
a notebook never needs the API key. If you get a `401 invalid_api_key` or
`401 invalid_node_token`, the first thing to check is whether the two got swapped.

## Quickstart

**Python:**

```python
from gpu_pool.client import PoolClient

pool = PoolClient("https://pool.example.com", api_key="...")

clip = pool.deploy(
    {"kind": "transformers", "task": "feature-extraction", "model": "openai/clip-vit-base-patch32"},
    mem_gib=2,
)
print(clip.infer({"inputs": "a photo of a cat"}))
print(pool.status())   # aggregate + per-node free VRAM
clip.teardown()
```

**curl:**

```bash
# Deploy (blocks until the model is loaded on a node -- can take minutes for a real model).
curl -s -X POST https://pool.example.com/v1/deployments \
  -H "Authorization: Bearer $GPU_POOL_API_KEY" \
  -d '{"modelSpec": {"kind": "echo"}, "memBytes": 1073741824}'
# -> {"deploymentId":"dep-...","nodeId":"node-...","modelSpec":{"kind":"echo"}}

curl -s -X POST https://pool.example.com/v1/deployments/$DEP/infer \
  -H "Authorization: Bearer $GPU_POOL_API_KEY" \
  -d '{"payload": {"ping": 1}}'
# -> {"result": {"echo": {"ping": 1}}}

# Long-running work as a job instead:
curl -s -X POST https://pool.example.com/v1/deployments/$DEP/jobs \
  -H "Authorization: Bearer $GPU_POOL_API_KEY" \
  -d '{"payload": {"prompt": "..."}}'
# -> 202 {"jobId":"job-...","deploymentId":"dep-...","nodeId":"node-...","state":"queued"}

curl -s "https://pool.example.com/v1/jobs/$JOB?wait=25&sinceSeq=0" \
  -H "Authorization: Bearer $GPU_POOL_API_KEY"
# long-polls up to 25s, returns as soon as state or progress changes

curl -s https://pool.example.com/v1/deployments/$DEP -X DELETE \
  -H "Authorization: Bearer $GPU_POOL_API_KEY"
```

## The deployment lifecycle, and what `mem_gib` actually means

`pool.deploy(model_spec, mem_gib=N)` is a **synchronous, blocking call** — it returns
once the model has actually finished loading on a node (or raises). It can legitimately
take minutes for a real model that pip-installs dependencies or downloads weights on
first load; the client's default deploy timeout is 660s specifically to accommodate the
control plane's own 600s load budget (`GPU_POOL_LOAD_TIMEOUT`) plus margin — don't pass a
shorter `timeout=` to `PoolClient(...)` unless you also shorten what you expect a load
to take.

**`mem_gib` (or `mem_bytes`) is a pure bookkeeping reservation the scheduler uses to
bin-pack deployments onto nodes — it does not constrain, measure, or enforce anything
about the model's *actual* VRAM use.** If you under-declare it, the scheduler may place
your model on a node that doesn't really have room, and the model will fail or OOM on the
node itself — the pool has no way to know that happened except through the model's own
error. Size it to what the model genuinely needs (check the node's free VRAM first via
`pool.status()`/`pool.nodes()`), not to whatever number happens to make placement easy.

Tearing down (`dep.teardown()` / `DELETE /v1/deployments/{id}`) has three honest
outcomes, not one:

- Node confirmed the unload → `{"released": true, "nodeConfirmed": true}`.
- Node wasn't even connected → `{"released": true, "nodeConfirmed": false}` (nothing to
  protect — the reservation is released).
- Node **is** connected but didn't confirm within `GPU_POOL_UNLOAD_TIMEOUT` → raises
  `PoolServerTimeoutError` (`504 unload_deadline_exceeded`) and **keeps the VRAM
  reserved**, rather than silently releasing it out from under a node that might still
  hold it. Pass `teardown(force=True)` to release anyway — the response's
  `vramPossiblyLeaked: true` means exactly what it says.

## Choosing blocking `infer()` vs a job

| expected duration | use | why |
|---|---|---|
| well under a minute | `dep.infer(payload)` | Simplest call shape; one request, one response. |
| up to ~2 minutes | `dep.infer(payload, timeout=N)` | You can request a shorter deadline than the operator's ceiling via `timeout=`, never longer — the control plane caps it at `GPU_POOL_INFER_TIMEOUT` (120s by default) regardless of what you ask for. |
| minutes to ~30 hours | `dep.submit(payload, ...)` / `dep.run(payload, ...)` | A job returns in milliseconds (bounded only by the node accepting it) and then runs for as long as the node stays connected — progress, cancellation, and (if the model checkpoints) survival across a node dying are all part of this path. |

`infer()`'s hard ceiling is `GPU_POOL_INFER_TIMEOUT` (**120 seconds by default**,
operator-configurable up to `GPU_POOL_MAX_INFER_TIMEOUT`, 900s by default) — there is no
way to make a single blocking call run longer than the operator's ceiling; that's the
entire reason the job API exists. If you hit `infer_deadline_exceeded` regularly,
that's the signal to switch this call to `submit()`/`run()`, not to keep raising the
timeout.

## The job lifecycle

```
queued ──▶ running ──▶ succeeded
              │      ╲
              │       ╲──▶ failed
              │        ╲
              │         ╲─▶ canceled
              │
              │  (node dies / drains)
              ▼
          orphaned ──▶ replacing ──▶ running   (attempt += 1, same jobId, checkpoint resumed)
              │
              │  (no node has room)
              ▼
       awaiting_capacity ──▶ replacing ──▶ running   (waits up to GPU_POOL_QUEUE_TIMEOUT_SECONDS)
```

`succeeded` / `failed` / `canceled` are **terminal** — nothing after them. `orphaned`,
`replacing`, and `awaiting_capacity` are the re-placement seam: the job isn't dead, it
just has no node actively running it *right now*. **The job id never changes across any
of this** — the same poll URL (`GET /v1/jobs/{id}`) keeps working; `attempt` increments
each time it's re-placed, and `placements[]` grows with `{nodeId, deploymentId,
startedAtUnix, endedAtUnix, endReason}` for every attempt so you can see exactly what
happened (`node_lost`, `node_quota_expired`, `succeeded`, ...).

Two independent guards stop a broken job from consuming fleet capacity forever:
`maxAttempts` (default 5, `GPU_POOL_MAX_JOB_ATTEMPTS`) bounds thrash from a genuinely
flapping node; `maxColdRestarts` (default 2, `GPU_POOL_MAX_COLD_RESTARTS`) detects a job
that gets re-placed but never actually calls `ctx.save_checkpoint()`, so it would
otherwise restart from zero forever — it fails with `no_checkpoint_progress` naming the
cause instead.

`Job.wait()` on the client treats `orphaned`/`awaiting_capacity`/`replacing`
(`Job.UNPLACED`) exactly like `queued`/`running` — only a `Job.TERMINAL` state ends the
wait. A `no_capacity` failure (job waited `GPU_POOL_QUEUE_TIMEOUT_SECONDS`, 6h default,
with nowhere to run) **retains the checkpoint pointer**, so a manual resubmit with
`resumable=True` picks up where it left off even after that.

## Progress and long-polling

`Job.refresh(wait=N)` (and `GET /v1/jobs/{id}?wait=25&sinceSeq=137`) long-polls: the
call blocks server-side until the job's state changes, its progress `seq` advances past
`sinceSeq`, or `wait` seconds elapse — whichever comes first. This is what makes polling
in a loop both cheap (roughly one request per `wait` seconds) and near-real-time (you
learn about a change almost as soon as it happens, not on your own fixed polling
cadence). `Job.wait(on_progress=...)` wraps this in a loop for you:

```python
job = dep.submit({"dataset": "...", "total_steps": 20000}, max_runtime_seconds=30*3600, resumable=True)
result = job.wait(on_progress=lambda j: print(f"{j.fraction*100:.0f}% step {j.progress.get('step')}"))
```

`on_progress` only fires when the progress `seq` actually advanced (not on every poll),
so you don't need to de-duplicate identical callbacks yourself. Pass `long_poll=None` to
`wait()` to fall back to plain fixed-interval `sleep(poll_interval)` polling instead, if
your environment can't hold a 25-second-plus request open.

Any process holding a `jobId` can attach to it — it doesn't have to be the process that
submitted it: `pool.job_handle(job_id)` immediately refreshes and returns a populated
`Job`, so a separate worker, a cron job, or a debugging shell can all watch (or cancel)
the same job.

## Cancellation — three honest tiers

CPython threads cannot be forcibly killed, and a bare C/CUDA call
(`model.generate()`, a Piper `synthesize_wav`, a Whisper pipeline step) cannot be
interrupted mid-call from the outside. What cancellation actually does depends on
whether the model opted into `ctx`:

1. **Cooperative — the model actually stops.** A model that calls `ctx.check_cancel()`
   (or iterates with `ctx.iter(...)`, which calls it for you every item) raises
   `JobCancelled` at its next checkpointable point and the job goes `canceled` cleanly,
   with no thread left running.
2. **Detached — cancellation is requested but not honored; the deployment is `dirty`.**
   For the **blocking `infer()`** path specifically, no built-in `pool_models/*` model
   currently has a way to observe a cancel mid-call (that ability comes from the job
   API's `ctx`, which `infer()`'s legacy code path doesn't pass) — so today a timeout on
   `infer()` always reports `deadline.work_state == "abandoned"` and marks that
   deployment `dirty` (visible on `GET /v1/nodes`), meaning a thread may still be running
   against it and burning GPU. Nothing forcibly reclaims that VRAM on its own; see tier 3.
   For a **job** whose model doesn't check `ctx`, cancelling it doesn't stop it either —
   the worker thread runs to natural completion and *then* the job is marked `canceled`
   (or whatever it actually returned), rather than being cut off early.
3. **Forced — actually reclaim the VRAM.** `dep.teardown(force=True)` (`DELETE
   /v1/deployments/{id}?force=true`) unloads the deployment regardless of whether the
   node confirms, releasing the reservation and reporting `vramPossiblyLeaked: true`. This
   is the honest way to recycle a node stuck running an uncooperative model — recreate
   the deployment fresh afterward. There is currently no equivalent `?force=` on job
   cancellation itself (`POST /v1/jobs/{id}/cancel` takes no such parameter); forcing a
   stuck job means tearing down its deployment.

The practical takeaway: if you need real mid-run cancellation (or checkpointed
resumability, or progress), write your model against `ctx` (next section). If you don't,
cancellation is best-effort and the deployment should be assumed to need a forced
teardown to fully reclaim its VRAM.

## Writing a `pool_models/` module

A node loads a `"kind": "python"` model spec by importing `entrypoint`'s module and
calling `factory(**args)`:

```python
{"kind": "python", "entrypoint": "pool_models.my_model:build", "args": {...}}
```

`build()` must return either a plain callable `f(payload) -> result`, or an object with
an `.infer(payload)` (or `.infer(payload, ctx)`) method. The **only** thing that decides
whether a model gets progress/cancellation/checkpointing is whether its `infer` names a
parameter `ctx` (or `job_ctx`) — sniffed once via `inspect.signature` when the model
loads:

```python
def infer(self, payload):            # old style -- works forever, unchanged
    ...

def infer(self, payload, ctx):        # opt in to progress + cancel + checkpoint/resume
    for item in ctx.iter(items, every=10, phase="embed"):   # progress + cancel in one line
        ...
    return {...}
```

The whole `ctx` surface an author can use:

| member | does |
|---|---|
| `ctx.cancelled` | `bool` — true once a cancel was requested. |
| `ctx.check_cancel()` | Raises `JobCancelled` if `cancelled` — call it at safe points in a manual loop. |
| `ctx.iter(iterable, total=, every=1, phase=)` | Wraps a loop: calls `check_cancel()` and reports progress for you. |
| `ctx.progress(fraction=, step=, total=, phase=, message=, **metrics)` | Cheap to call in a tight loop — it just writes a slot under a lock; a background publisher decides when to actually send a frame. |
| `ctx.drain_requested` | `bool`, set by the framework when the node is mid-drain — "wrap up at your next safe point", distinct from `cancelled` ("stop now, don't bother saving"). |
| `ctx.should_checkpoint()` | Framework decides **when**: true once `GPU_POOL_CHECKPOINT_INTERVAL` (600s default) has elapsed, or immediately if `drain_requested`. |
| `ctx.load_checkpoint()` | Author decides **what to do with it**: returns the last saved `meta` dict, or `None` on a fresh start (no store configured, or never checkpointed before). Fetches into `ctx.checkpoint_dir` the first time; free on repeat calls. |
| `ctx.save_checkpoint(meta)` | Author decides **what** (`meta`) and **where** (files under `ctx.checkpoint_dir`); uploads blobs-first-pointer-last, prunes older versions, resets the checkpoint clock. |
| `ctx.checkpoint_dir` | `Path` — where to write files before calling `save_checkpoint`. |

Worked reference, `pool_models/resumable_batch.py` (a cursor-based batch job — the
checkpoint is a single integer, so it works even on the 8 MiB control-plane fallback
store):

```python
class ResumableBatchJob:
    def infer(self, payload, ctx):
        checkpoint = ctx.load_checkpoint()          # None on a fresh start; {"cursor": N} on resume
        cursor = checkpoint["cursor"] if checkpoint else 0
        total = payload["total"]
        while cursor < total:
            if ctx.cancelled:
                break
            cursor = self._process_batch(cursor, ...)
            ctx.progress(step=cursor, total=total, phase="batch")
            if ctx.should_checkpoint():
                ctx.save_checkpoint({"cursor": cursor})
                if ctx.drain_requested:
                    return {"status": "checkpointed", "cursor": cursor}
        return {"status": "done", "cursor": cursor}

def build():
    return ResumableBatchJob()
```

Submit it as `resumable=True` so the scheduler/replacer feel free to hand it to a
short-lived node rather than insisting on one that outlives the whole run:

```python
dep = pool.deploy({"kind": "python", "entrypoint": "pool_models.resumable_batch:build"}, mem_gib=1)
job = dep.submit({"total": 500_000, "batch": 256}, resumable=True)
```

**Model code ships by `git push`, not by editing a running node.** Every node
`git reset --hard origin/main`s its checkout of your repo before each `python`-kind
deploy (unless `GPU_POOL_AUTO_UPDATE=0`), so **a local, uncommitted change to
`pool_models/` never reaches a node** — push it to whichever repo/branch the node's
`GPU_POOL_REPO` actually points at first, then deploy. There is no notebook restart
needed for this — it's exactly what lets a bug fix land on a live pool.

## The complete error reference

Every non-2xx response is `{"detail": "<code>: <message>", "error": {...}}` — `detail`
is always a plain string starting with the code (stable across languages, so old
substring checks like `"pool_full" in str(exc)` keep working); `error` is the structured
form. `GET /v1/errors` (no auth needed) serves this exact table live from
`gpu_pool/errors.py`, so treat it as the source of truth if this ever drifts.

| code | HTTP | retryable | meaning / what to do |
|---|---|---|---|
| `invalid_api_key` | 401 | no | Missing/incorrect API key. |
| `invalid_node_token` | 401 | no | Missing/incorrect node token (a node-facing error, not yours unless you're running a node). |
| `request_body_too_large` | 413 | no | Body exceeded `GPU_POOL_MAX_REQUEST_BYTES`. Shrink the payload. |
| `invalid_request` | 422 | no | Request body failed validation. |
| `model_spec_required` | 400 | no | `modelSpec` must be a JSON object. |
| `mem_bytes_required` | 400 | no | `memBytes` must be a positive integer. |
| `mem_bytes_invalid` | 400 | no | `memBytes` wasn't a valid integer. |
| `deadline_invalid` | 400 | no | `X-Pool-Deadline-Seconds` wasn't a positive number ≤ the operator's ceiling. |
| `pool_full` | 503 | yes | No connected node currently has enough free VRAM. Retry later, or reduce `mem_gib`. |
| `no_nodes_connected` | 503 | yes | No nodes connected at all. |
| `node_draining` | 503 | yes | The specific node you targeted is draining and accepts no new work. |
| `unknown_node` | 404 | no | No node with that id is connected. |
| `unknown_deployment` | 404 | no | No deployment with that id exists (may have been evicted — check for a `node_lost`/`node_quota_expired` tombstone instead, see below). |
| `load_rejected` | 502 | no | The node rejected the model load (e.g. an exception constructing it). Check the model spec/args. |
| `module_not_allowlisted` | 403 | no | That entrypoint module isn't on the node's `GPU_POOL_ALLOWED_MODULE_PREFIXES` allowlist. An authorization decision — retrying never helps; the operator must widen the allowlist. |
| `unsupported_model_kind` | 400 | no | `model_spec.kind` isn't one of `echo`/`python`/`transformers`. |
| `node_cap_exceeded` | 503 | yes | The node's own VRAM cap would be exceeded. |
| `node_connection_missing` | 503 | yes | The node's connection dropped before the request could be sent — a transient race. |
| `deployment_node_offline` | 503 | yes | The deployment's node isn't currently connected. |
| `node_disconnected` | 502 | yes | The node's connection dropped while the request was in flight. |
| `infer_failed` | 502 | yes | The node reported a failure running inference (transport/runtime level, not the model raising). |
| `model_error` | 502 | yes | The model itself raised handling the payload — distinct from `infer_failed`: this is "your payload broke the model," not "the pool broke." Check your payload before retrying. |
| `unload_failed` | 502 | yes | The node reported a failure unloading. |
| `queue_full` | 503 | yes | The node's job queue (`GPU_POOL_NODE_JOB_QUEUE`) is full. Retry shortly. |
| `deployment_lost` | 502 | yes | The node no longer has this deployment loaded. |
| `unknown_job` | 404 | no | No job with that id exists. |
| `job_not_finished` | 409 | yes | You called `GET /v1/jobs/{id}/result` before the job reached a terminal state. Poll/wait first. |
| `job_canceled` | 409 | no | Raised specifically by `GET /v1/jobs/{id}/result` if the job ended up `canceled` — fetch `GET /v1/jobs/{id}` instead to see its state/progress, which is always 200 even for a canceled job. |
| `job_deadline_exceeded` | 504 | no | The job's total wall-clock budget (`max_runtime_seconds`) was exceeded across all attempts combined. |
| `no_capacity` | 504 | yes | No node had room for this job before `GPU_POOL_QUEUE_TIMEOUT_SECONDS` elapsed. The checkpoint pointer (if any) is retained — resubmit by hand. |
| `insufficient_runway` | 503 | yes | No connected node has enough quota left to finish this job's load+run — wait for a longer-lived node, or make the job `resumable=True`. |
| `no_checkpoint_progress` | 409 | no | The job was re-placed more than once and never called `ctx.save_checkpoint()` — it would restart from zero forever. Fix the model to checkpoint, or resubmit by hand knowing it'll restart. |
| `max_attempts_exceeded` | 504 | no | The job was re-placed the maximum allowed number of times (`GPU_POOL_MAX_JOB_ATTEMPTS`) and still failed. |
| `load_deadline_exceeded` | 504 | yes | Loading the model exceeded `GPU_POOL_LOAD_TIMEOUT`. Real models that pip-install/download weights may need this raised. |
| `infer_deadline_exceeded` | 504 | yes | A blocking `infer()` exceeded its deadline. **This is the signal to switch to a job.** |
| `unload_deadline_exceeded` | 504 | yes | Unloading exceeded `GPU_POOL_UNLOAD_TIMEOUT`; the VRAM reservation is kept (not released) — see the teardown section above. |
| `node_lost` | 504 | yes | The node stopped heartbeating and was evicted. |
| `node_quota_expired` | 504 | yes | The node reached its own self-imposed session quota and drained voluntarily — normal for free-tier notebooks. |
| `upstream_proxy_timeout` | 504 | yes | An intermediary (your reverse proxy, a CDN) timed out with no `error.code` at all — the control plane itself never got to reply. See `docs/SETUP.md`'s proxy-timeout warning. |
| `client_timeout` | *(none — client-synthesized)* | yes | Your own client gave up waiting before any reply arrived — not an HTTP status from the server at all; see `PoolClientTimeoutError` below. |
| `unknown_checkpoint` | 404 | no | (Node-facing.) No checkpoint exists under that key/version. |
| `checkpoint_too_large` | 413 | no | A checkpoint exceeded the control-plane fallback store's 8 MiB/version cap — configure HF or S3 (`docs/SETUP.md` §8). |
| `checkpoint_key_invalid` | 400 | no | (Node-facing.) The checkpoint key contains disallowed characters. |
| `internal_error` | 500 | no | Unexpected internal error — worth reporting upstream. |

### The deadline model

Every `*_deadline_exceeded` (and `node_lost`/`node_quota_expired`) error carries a
`deadline` object (`error.deadline` on the wire; `exc.deadline` — a `Deadline` — on the
client), answering "which budget did I actually blow":

| field | meaning |
|---|---|
| `layer` | Which budget: `client`, `infer`, `load`, `unload`, `queue`, `job`, `node_quota`, `platform`, or `proxy`. |
| `limitSeconds` | What actually applied (the operator's ceiling, or your own client-side timeout). |
| `elapsedSeconds` | How long it actually ran before the deadline hit. |
| `requestedSeconds` | What *you* asked for (via `timeout=`), if anything — may be larger than `limitSeconds` if the operator's ceiling capped it. |
| `budgetSource` | The env var (or `"client"`) that set the limit, e.g. `"GPU_POOL_INFER_TIMEOUT"`. |
| `workState` | `stopped` (node confirmed it actually stopped — nothing still burning GPU), `cancelling` (cancel sent, no ack yet), `abandoned` (couldn't be honored — assume it's still running), `running` (cancellation wasn't even attempted), or `unknown`. |
| `sideEffects` | `none`, `possible`, or `unknown` — whether the work may have had partial effects. |
| `resultRecoverable` | Whether retrying is automatically safe — `false` means "the work may have partially completed." |
| `nodeQuotaRemainingSeconds` | How much of the node's own session quota was left when this happened, if relevant. |

`exc.explain()` on any client-side `PoolError` turns all of this into one human-readable
paragraph — this is the direct answer to "did my work time out, and against which
budget, and is it safe to retry":

```python
try:
    text = dep.infer({"prompt": "..."}, timeout=300)
except PoolTimeoutError as e:
    print(e.explain())
    # "HTTP 504: infer_deadline_exceeded: ... The infer budget was 120.0s (source:
    #  GPU_POOL_INFER_TIMEOUT); it ran for 120.0s. You asked for 300.0s but the pool
    #  capped it. The node confirmed it stopped; nothing is still burning GPU time."
```

### Client exception hierarchy

`gpu_pool/client.py` maps every error code (and, for statuses with no more specific
mapping, the HTTP status itself) to a typed exception, all rooted at `PoolError`
(itself a `RuntimeError`, so `except PoolError` — or even plain `except Exception` —
always still works):

```
PoolError
├── PoolTimeoutError(PoolError, TimeoutError)   # also catchable via `except TimeoutError`
│   ├── PoolClientTimeoutError    # your own client gave up; no HTTP status was ever received
│   └── PoolServerTimeoutError    # the control plane answered 504: a real deadline was exceeded
├── PoolConnectionError           # DNS/refused/TLS -- never reached the pool at all
├── PoolAuthError                 # invalid_api_key / 401 / 403
├── PoolRequestError               # malformed request (4xx)
│   ├── PoolValidationError
│   └── PoolPayloadTooLargeError
├── PoolNotFoundError
│   ├── PoolUnknownDeploymentError
│   ├── PoolUnknownNodeError
│   └── PoolUnknownJobError
├── PoolConflictError              # 409: conflicts with current state
│   ├── PoolJobNotFinishedError
│   └── PoolJobCanceledError
├── PoolCapacityError              # 503: no capacity right now (.retryable tells you whether to bother)
├── PoolModelError
│   ├── PoolModelNotAllowedError   # module_not_allowlisted
│   └── PoolInferenceFailedError  # model_error / infer_failed
├── PoolNodeError                  # 502, not a deadline
└── PoolServerError                # any other 5xx
```

Every instance carries `.code`, `.status`, `.deployment_id`, `.node_id`, `.job_id`,
`.deadline`, `.details`, `.retryable`, `.retry_after_seconds`, `.request_id`, and
`.explain()`. `str(exc)` is always `"HTTP <status>: <code>: <message>"`.

## Timeout/retry recipes

- **A `PoolCapacityError` (`pool_full`, `no_nodes_connected`, `queue_full`,
  `insufficient_runway`, ...) is worth retrying with backoff** — check `.retryable` and
  `.retry_after_seconds` if present rather than hardcoding a policy per code.
- **A `PoolServerTimeoutError` on `infer()`** almost always means "use a job instead,"
  not "retry with a longer timeout" — you can't raise the timeout past
  `GPU_POOL_MAX_INFER_TIMEOUT` anyway, and `.deadline.result_recoverable` tells you
  whether the attempted call is even safe to repeat.
- **A `PoolClientTimeoutError`** means *your own* socket gave up — the pool may still be
  working on it. Don't assume it failed; for an `infer()` call this is genuinely
  ambiguous (there's no job id to re-attach to), which is itself an argument for using
  a job for anything non-trivial.
- **`no_checkpoint_progress` and `max_attempts_exceeded` are not worth retrying
  as-is** — they mean the job (or its re-placement mechanics) structurally cannot
  succeed until the underlying cause (a model that never checkpoints; a fleet that keeps
  losing this job) is fixed.
- Call `pool.limits()` (`GET /v1/limits`) once at startup and size your own `timeout=`
  values against the operator's actual effective budgets, rather than guessing — the
  client's own defaults (`deploy`: 660s, `infer`: 130s unless you pass `timeout=` to
  `PoolClient(...)`) already assume the shipped server defaults and will be wrong if the
  operator raised `GPU_POOL_LOAD_TIMEOUT`/`GPU_POOL_INFER_TIMEOUT`.

## Observability endpoints

All require the API key except `/healthz` and `/v1/errors`, which are public:

| endpoint | shows |
|---|---|
| `GET /healthz` | Liveness + connected node count. |
| `GET /v1/pool` | Aggregate + per-node free VRAM. |
| `GET /v1/nodes` / `GET /v1/nodes/{id}` | Per-node detail: GPU, VRAM, uptime, heartbeat age, deployment count, draining flag. |
| `GET /v1/accounts` | Connected Colab/Kaggle accounts grouped, with contributed/free capacity and minimum quota remaining. |
| `GET /v1/deployments` | Live deployments: kind, node, reserved VRAM, age. |
| `GET /v1/jobs` (optionally `?deploymentId=`/`?state=`) | Every known job's current state + progress. |
| `GET /v1/limits` | Every effective server-side timeout/budget, plus `serverTimeUnix` — size your own client timeouts against this instead of guessing. |
| `GET /v1/errors` | The whole error registry as data (public). |

`python backend_scripts/list_nodes.py` renders most of this in one shot if you have a
checkout of this repo handy.

## Limits and honest caveats

- **Session caps are real and platform-imposed, not a pool limitation:** roughly ~11h on
  Colab, ~8h on Kaggle, before a node voluntarily drains and disconnects (see
  `GPU_POOL_QUOTA_MINUTES` in `docs/SETUP.md`). A job that needs more wall-clock time
  than one session provides **will** be handed off to a fresh node — that's what
  `resumable=True` + checkpointing is for.
- **Handoffs need a human.** Free-tier runtime creation requires an authenticated
  browser session; there's no API to spin up a fresh Colab/Kaggle notebook
  unattended. A 30-hour job will get re-placed at least twice, and each handoff needs
  someone to open a new notebook and hit Run — this is inherent to how these platforms
  work, not something more code can fix. Kaggle's ~30h/week GPU budget means one
  account covers roughly one 30-hour job per week with no slack.
- **`mem_gib` is bookkeeping, not a VRAM ceiling** (see above) — the real ceiling is
  whatever the node's physical GPU actually has. Anything whose optimizer state runs to
  tens of GB needs tens of GB of *real* VRAM under the one-model-one-node rule; full 7B
  fine-tuning is out of scope on a single T4, while LoRA/QLoRA adapters, batch-inference
  cursors, embeddings/ETL, and full training of sub-1B models all fit comfortably.
- **This runs on free-tier infrastructure whose Terms of Service explicitly prohibit
  this kind of use.** Treat every node as ephemeral and best-effort; for anything you
  actually depend on operationally, plan for a paid fallback node (`GPU_POOL_PROVIDER`
  auto-detects as `generic` with no session cap for real hardware).
