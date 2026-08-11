"""Backend SDK -- what your application servers import to use the pool.

    from gpu_pool.client import PoolClient
    pool = PoolClient("https://pool.example.com", api_key="...")
    clip = pool.deploy({"kind": "transformers", "task": "feature-extraction",
                        "model": "openai/clip-vit-base-patch32"}, mem_gib=2)
    vec = clip.infer({"inputs": "a photo of a cat"})
    print(pool.status())

Intentionally **dependency-free** (stdlib ``urllib`` only) and synchronous, so it drops
into any Python 3 backend with nothing to install -- vendor this one file if you don't
want to depend on the rest of the repo. Payloads are whatever JSON the model expects;
use ``gpu_pool.protocol.encode_bytes`` to embed binary (audio/images) as base64.

Errors: every failure raises a :class:`PoolError` (or a more specific subclass below).
``str(exc)`` is always ``"HTTP <status>: <code>: <message>"`` so old substring checks
(``"pool_full" in str(exc)``) keep working; for anything else read the typed fields --
``exc.code``, ``exc.status``, ``exc.deadline`` (present on every timeout), and
``exc.explain()`` for a one-paragraph human explanation of what happened and whether
it's safe to retry.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

GIB = 1024 ** 3


@dataclass(frozen=True)
class Deadline:
    """Which budget was exceeded, mirrors gpu_pool.errors.DeadlineInfo on the wire."""

    layer: str | None = None
    limit_seconds: float | None = None
    elapsed_seconds: float | None = None
    requested_seconds: float | None = None
    budget_source: str | None = None
    work_state: str | None = None
    side_effects: str | None = None
    result_recoverable: bool | None = None
    node_quota_remaining_seconds: float | None = None

    @classmethod
    def from_wire(cls, raw: dict[str, Any] | None) -> "Deadline | None":
        if not raw:
            return None
        return cls(
            layer=raw.get("layer"), limit_seconds=raw.get("limitSeconds"),
            elapsed_seconds=raw.get("elapsedSeconds"), requested_seconds=raw.get("requestedSeconds"),
            budget_source=raw.get("budgetSource"), work_state=raw.get("workState"),
            side_effects=raw.get("sideEffects"), result_recoverable=raw.get("resultRecoverable"),
            node_quota_remaining_seconds=raw.get("nodeQuotaRemainingSeconds"),
        )


class PoolError(RuntimeError):
    """Base class for every error this client raises. Unchanged as the base on purpose:
    code that already does ``except PoolError`` keeps working with no changes."""

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None,
                 deployment_id: str | None = None, node_id: str | None = None,
                 job_id: str | None = None, deadline: Deadline | None = None,
                 details: dict[str, Any] | None = None, retryable: bool | None = None,
                 retry_after_seconds: float | None = None, request_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.deployment_id = deployment_id
        self.node_id = node_id
        self.job_id = job_id
        self.deadline = deadline
        self.details = details or {}
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.request_id = request_id

    def explain(self) -> str:
        """A one-paragraph, human-readable account of what happened. For a deadline
        error this is the direct answer to 'did my work time out, and against which
        budget?' -- the thing a flat error string could never say."""
        if self.deadline is None:
            return str(self)
        d = self.deadline
        parts = [str(self)]
        if d.limit_seconds is not None and d.elapsed_seconds is not None:
            parts.append(f"The {d.layer} budget was {d.limit_seconds:.1f}s "
                         f"(source: {d.budget_source or 'unknown'}); it ran for {d.elapsed_seconds:.1f}s.")
        if d.requested_seconds is not None and d.limit_seconds is not None \
                and d.requested_seconds > d.limit_seconds:
            parts.append(f"You asked for {d.requested_seconds:.1f}s but the pool capped it.")
        if d.work_state == "stopped":
            parts.append("The node confirmed it stopped; nothing is still burning GPU time.")
        elif d.work_state in ("unknown", "running", "cancelling"):
            parts.append("The node may still be running the work; this call did not cancel it.")
        if d.result_recoverable is False:
            parts.append("The work may have partially completed, so retrying is not automatically safe.")
        return " ".join(parts)


class PoolTimeoutError(PoolError, TimeoutError):
    """Any budget was exceeded -- client-side, server-side, or the node's own quota.
    Subclassing TimeoutError too means code written before this existed
    (``except TimeoutError``) still catches it."""


class PoolClientTimeoutError(PoolTimeoutError):
    """The client itself gave up waiting; no HTTP status was ever received. Distinct
    from PoolServerTimeoutError, which means the SERVER answered with a 504."""


class PoolServerTimeoutError(PoolTimeoutError):
    """The control plane returned 504: a load/infer/unload deadline was exceeded."""


class PoolConnectionError(PoolError):
    """DNS failure, connection refused, TLS error -- the request never reached the pool."""


class PoolAuthError(PoolError):
    pass


class PoolRequestError(PoolError):
    """4xx: the request itself was malformed."""


class PoolValidationError(PoolRequestError):
    pass


class PoolPayloadTooLargeError(PoolRequestError):
    pass


class PoolNotFoundError(PoolError):
    pass


class PoolUnknownDeploymentError(PoolNotFoundError):
    pass


class PoolUnknownNodeError(PoolNotFoundError):
    pass


class PoolUnknownJobError(PoolNotFoundError):
    pass


class PoolConflictError(PoolError):
    """409: the request conflicts with the resource's current state."""


class PoolJobNotFinishedError(PoolConflictError):
    """The job hasn't reached a terminal state yet -- fetching its result was premature."""


class PoolJobCanceledError(PoolConflictError):
    pass


class PoolCapacityError(PoolError):
    """503: no capacity right now. ``retryable`` tells you whether to bother."""


class PoolModelError(PoolError):
    """The model/loader itself refused the request or raised."""


class PoolModelNotAllowedError(PoolModelError):
    pass


class PoolInferenceFailedError(PoolModelError):
    pass


class PoolNodeError(PoolError):
    """502: the node or its connection failed in a way that isn't a deadline."""


class PoolServerError(PoolError):
    """5xx we don't have a more specific class for."""


_CODE_CLASS: dict[str, type[PoolError]] = {
    "invalid_api_key": PoolAuthError,
    "module_not_allowlisted": PoolModelNotAllowedError,
    "unsupported_model_kind": PoolRequestError,
    "model_error": PoolInferenceFailedError,
    "infer_failed": PoolInferenceFailedError,
    "load_rejected": PoolModelError,
    "unload_failed": PoolNodeError,
    "unknown_deployment": PoolUnknownDeploymentError,
    "unknown_node": PoolUnknownNodeError,
    "pool_full": PoolCapacityError,
    "no_nodes_connected": PoolCapacityError,
    "node_draining": PoolCapacityError,
    "node_connection_missing": PoolCapacityError,
    "deployment_node_offline": PoolCapacityError,
    "node_disconnected": PoolNodeError,
    "request_body_too_large": PoolPayloadTooLargeError,
    "invalid_request": PoolValidationError,
    "model_spec_required": PoolRequestError,
    "mem_bytes_required": PoolRequestError,
    "mem_bytes_invalid": PoolRequestError,
    "deadline_invalid": PoolRequestError,
    "queue_full": PoolCapacityError,
    "deployment_lost": PoolNodeError,
    "unknown_job": PoolUnknownJobError,
    "job_not_finished": PoolJobNotFinishedError,
    "job_canceled": PoolJobCanceledError,
}
_STATUS_CLASS: dict[int, type[PoolError]] = {
    400: PoolRequestError, 401: PoolAuthError, 403: PoolAuthError,
    404: PoolNotFoundError, 409: PoolConflictError, 413: PoolPayloadTooLargeError,
    422: PoolValidationError, 500: PoolServerError, 501: PoolServerError,
    502: PoolNodeError, 503: PoolCapacityError, 504: PoolServerTimeoutError,
}


def _exception_class(status: int | None, code: str | None) -> type[PoolError]:
    if code and code in _CODE_CLASS:
        return _CODE_CLASS[code]
    if code and code.endswith("_deadline_exceeded"):
        return PoolServerTimeoutError
    if code in ("node_lost", "node_quota_expired"):
        return PoolTimeoutError
    if status == 504:
        return PoolServerTimeoutError
    if status is not None and status in _STATUS_CLASS:
        return _STATUS_CLASS[status]
    return PoolError


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def _error_from_job(job_id: str, error: dict[str, Any] | None, *,
                    deployment_id: str | None = None, node_id: str | None = None) -> PoolError:
    """Builds the same typed exception a failed HTTP call would have raised, but from a
    job document's inline ``error`` field (GET /v1/jobs/{id} is always 200, even for a
    failed job, so there's no HTTPError to parse here)."""
    err = error or {}
    code = err.get("code")
    message = err.get("message") or code or "job failed"
    cls = _exception_class(None, code)
    return cls(f"job {job_id} failed: {message}", code=code, job_id=job_id,
              deployment_id=deployment_id, node_id=node_id, details=err.get("details"))


class Deployment:
    def __init__(self, client: "PoolClient", deployment_id: str, node_id: str,
                 model_spec: dict[str, Any]) -> None:
        self._client = client
        self.deployment_id = deployment_id
        self.node_id = node_id
        self.model_spec = model_spec

    def infer(self, payload: Any, *, timeout: float | None = None) -> Any:
        return self._client.infer(self.deployment_id, payload, timeout=timeout)

    def submit(self, payload: Any, *, max_runtime_seconds: float | None = None,
              resumable: bool = False) -> "Job":
        """Start long-running work against this deployment. Returns in milliseconds
        (the job then runs for as long as the node lives) -- poll the returned Job or
        call ``.wait()`` on it. Pass ``resumable=True`` if the model calls
        ``ctx.save_checkpoint()``/``ctx.load_checkpoint()`` -- it tells the pool the job
        can be handed off to a short-lived node instead of needing one that will live
        for the whole run."""
        return self._client.submit(self.deployment_id, payload,
                                   max_runtime_seconds=max_runtime_seconds, resumable=resumable)

    def run(self, payload: Any, *, max_runtime_seconds: float | None = None,
           resumable: bool = False, timeout: float | None = None, poll_interval: float = 2.0,
           on_progress: "Callable[[Job], None] | None" = None) -> Any:
        """submit() + wait() in one call: the job API with a blocking face. Prefer this
        over infer() for anything that might run more than a couple of minutes."""
        job = self.submit(payload, max_runtime_seconds=max_runtime_seconds, resumable=resumable)
        return job.wait(timeout=timeout, poll_interval=poll_interval, on_progress=on_progress)

    def teardown(self, *, force: bool = False) -> None:
        self._client.teardown(self.deployment_id, force=force)

    def __repr__(self) -> str:
        return f"Deployment(id={self.deployment_id!r}, node={self.node_id!r})"


class Job:
    """A handle on a submitted job. Any process holding the ``job_id`` can attach to it
    via ``PoolClient.job_handle()`` -- it doesn't have to be the one that submitted it."""

    TERMINAL = ("succeeded", "failed", "canceled")

    # A job that's temporarily unplaced (its node died, or nothing has capacity right
    # now) but not dead -- Job.wait() keeps polling through these exactly like
    # "queued"/"running"; only TERMINAL ends the wait.
    UNPLACED = ("orphaned", "awaiting_capacity", "replacing")

    def __init__(self, client: "PoolClient", job_id: str, deployment_id: str,
                 node_id: str | None, state: str = "queued") -> None:
        self._client = client
        self.job_id = job_id
        self.deployment_id = deployment_id
        self.node_id = node_id
        self.state = state
        self.progress: dict[str, Any] = {}
        self.error: dict[str, Any] | None = None
        self.attempt: int = 1
        self.placements: list[dict[str, Any]] = []
        self.pending_since: float | None = None

    @property
    def done(self) -> bool:
        return self.state in self.TERMINAL

    @property
    def fraction(self) -> float | None:
        return self.progress.get("fraction")

    def refresh(self, *, wait: float | None = None) -> "Job":
        """Pull the latest state. Pass ``wait`` (seconds) to long-poll: the call blocks
        until the state changes, progress advances, or ``wait`` elapses -- whichever
        comes first -- which is what makes polling in a loop cheap and near-real-time."""
        data = self._client.job(self.job_id, wait=wait, since_seq=self.progress.get("seq"))
        self.state = data["state"]
        self.progress = data.get("progress") or {}
        self.error = data.get("error")
        self.node_id = data.get("nodeId") or self.node_id
        self.deployment_id = data.get("deploymentId") or self.deployment_id
        self.attempt = data.get("attempt", self.attempt)
        self.placements = data.get("placements") or self.placements
        self.pending_since = data.get("pendingSinceUnix")
        return self

    def wait(self, *, timeout: float | None = None, poll_interval: float = 2.0,
             long_poll: float | None = 25.0,
             on_progress: "Callable[[Job], None] | None" = None) -> Any:
        """Block until the job reaches a terminal state, then return its result (or raise
        JobFailed/JobCanceled). Uses server-side long-polling when available (near-real-
        time, about one request per ``long_poll`` seconds); pass ``long_poll=None`` to
        fall back to plain ``sleep(poll_interval)`` polling instead."""
        deadline = None if timeout is None else time.monotonic() + timeout
        last_seq = self.progress.get("seq")
        while True:
            self.refresh(wait=long_poll)
            if on_progress is not None and self.progress.get("seq") != last_seq:
                last_seq = self.progress.get("seq")
                on_progress(self)
            if self.state == "succeeded":
                return self.result()
            if self.state == "failed":
                raise _error_from_job(self.job_id, self.error, deployment_id=self.deployment_id,
                                      node_id=self.node_id)
            if self.state == "canceled":
                raise PoolJobCanceledError(f"job {self.job_id} was canceled", code="job_canceled",
                                           job_id=self.job_id, deployment_id=self.deployment_id,
                                           node_id=self.node_id)
            if deadline is not None and time.monotonic() > deadline:
                raise PoolTimeoutError(
                    f"job {self.job_id} still {self.state} after {timeout:.1f}s of waiting "
                    f"(the job itself keeps running -- this is only YOUR wait() giving up)",
                    code="client_timeout", job_id=self.job_id, deployment_id=self.deployment_id)
            if not long_poll:
                time.sleep(poll_interval)

    def result(self) -> Any:
        return self._client.job_result(self.job_id)

    def cancel(self) -> None:
        self._client.cancel_job(self.job_id)

    def forget(self) -> None:
        self._client.forget_job(self.job_id)

    def __repr__(self) -> str:
        pct = "" if self.fraction is None else f" {self.fraction * 100:.0f}%"
        return f"Job(id={self.job_id!r}, state={self.state!r}{pct}, node={self.node_id!r})"


class PoolClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float | None = None,
                 deadline_grace_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        # If the caller passes an explicit `timeout`, it means what it always meant:
        # one number for every call, unchanged from before this existed. If they don't,
        # give deploy() the room its 600s server-side load budget actually needs --
        # the old flat 130s routinely orphaned a deployment on a real model's first load.
        self._timeout = timeout if timeout is not None else 130.0
        self._deploy_timeout = timeout if timeout is not None else 660.0
        self._grace = deadline_grace_seconds

    def close(self) -> None:
        pass  # no persistent connection; kept for API symmetry / context-manager use

    def __enter__(self) -> "PoolClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None, *,
                timeout: float | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._api_key}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        effective_timeout = timeout if timeout is not None else self._timeout
        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raw_body = exc.read()
            try:
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = {}
            err = payload.get("error") or {}
            code = err.get("code")
            detail = payload.get("detail")
            detail = detail if isinstance(detail, str) else (_stringify(detail) or code or "")
            message = f"HTTP {exc.code}: {detail}"
            cls = _exception_class(exc.code, code)
            raise cls(
                message, code=code, status=exc.code,
                deployment_id=err.get("deploymentId"), node_id=err.get("nodeId"),
                job_id=err.get("jobId"), deadline=Deadline.from_wire(err.get("deadline")),
                details=err.get("details"), retryable=err.get("retryable"),
                retry_after_seconds=err.get("retryAfterSeconds"), request_id=err.get("requestId"),
            ) from exc
        except TimeoutError as exc:
            # The read/connect timeout escape hatch: urllib raises a bare TimeoutError
            # here (a sibling of URLError, not caught by it), so without this branch a
            # client-side timeout was indistinguishable from any other failure and
            # invisible to `except PoolError`. This is the fix for that.
            raise PoolClientTimeoutError(
                f"client_timeout: no response from {method} {path} within {effective_timeout:.1f}s "
                f"(the pool may still be working on it)",
                code="client_timeout",
                deadline=Deadline(layer="client", limit_seconds=effective_timeout,
                                  budget_source="client", work_state="unknown",
                                  side_effects="possible", result_recoverable=False),
            ) from exc
        except urllib.error.URLError as exc:
            raise PoolConnectionError(f"connection error to {url}: {exc.reason}") from exc
        return json.loads(raw.decode("utf-8")) if raw else {}

    # --- deployments ---------------------------------------------------------
    def deploy(self, model_spec: dict[str, Any], *, mem_gib: float | None = None,
               mem_bytes: int | None = None, node_id: str | None = None,
               timeout: float | None = None) -> Deployment:
        """Place a model on the pool. Specify its VRAM need via ``mem_gib`` or ``mem_bytes``.
        Pass ``node_id`` to pin the deployment to a specific node (e.g. for per-node tests)."""
        if mem_bytes is None:
            if mem_gib is None:
                raise ValueError("provide mem_gib or mem_bytes")
            mem_bytes = int(mem_gib * GIB)
        body: dict[str, Any] = {"modelSpec": model_spec, "memBytes": mem_bytes}
        if node_id is not None:
            body["nodeId"] = node_id
        data = self._request("POST", "/v1/deployments", body,
                             timeout=timeout if timeout is not None else self._deploy_timeout)
        return Deployment(self, data["deploymentId"], data["nodeId"], data["modelSpec"])

    def infer(self, deployment_id: str, payload: Any, *, timeout: float | None = None) -> Any:
        """Run inference. ``timeout`` (seconds), if given, is sent to the control plane as
        the requested per-call deadline (it can only ever shorten the operator's own
        ceiling, never extend it) and also bounds how long this call itself blocks."""
        headers = {"X-Pool-Deadline-Seconds": str(timeout)} if timeout is not None else None
        call_timeout = (timeout + self._grace) if timeout is not None else self._timeout
        data = self._request("POST", f"/v1/deployments/{deployment_id}/infer",
                            {"payload": payload}, timeout=call_timeout, headers=headers)
        return data.get("result")

    def teardown(self, deployment_id: str, *, force: bool = False) -> dict[str, Any]:
        """Tears down a deployment. Returns details about whether the node actually
        confirmed the unload (``nodeConfirmed``) -- a live node that fails to unload
        keeps its VRAM reservation and raises instead of silently releasing it out from
        under it; pass ``force=True`` to release anyway (``vramPossiblyLeaked`` in that
        case means exactly what it says)."""
        suffix = "?force=true" if force else ""
        return self._request("DELETE", f"/v1/deployments/{deployment_id}{suffix}")

    # --- jobs (long-running work; use this instead of infer() for anything that might
    # take more than a couple of minutes) -------------------------------------------
    def submit(self, deployment_id: str, payload: Any, *,
              max_runtime_seconds: float | None = None, resumable: bool = False) -> Job:
        """Start a job and return immediately (bounded by the server's job-ack timeout,
        NOT by how long the job itself takes). The job then runs for as long as the node
        stays connected -- poll the returned handle, or call ``.wait()`` on it."""
        body: dict[str, Any] = {"payload": payload}
        if max_runtime_seconds is not None:
            body["maxRuntimeSeconds"] = max_runtime_seconds
        if resumable:
            body["resumable"] = True
        data = self._request("POST", f"/v1/deployments/{deployment_id}/jobs", body)
        return Job(self, data["jobId"], data["deploymentId"], data.get("nodeId"),
                  data.get("state", "queued"))

    def job_handle(self, job_id: str) -> Job:
        """Attach to a job submitted by ANY process (yours or another) -- all you need
        is its id. Immediately calls refresh() so the handle starts populated."""
        return Job(self, job_id, "", None).refresh()

    def job(self, job_id: str, *, wait: float | None = None, since_seq: Any = None) -> dict[str, Any]:
        query = []
        if wait:
            query.append(f"wait={wait}")
        if since_seq is not None:
            query.append(f"sinceSeq={since_seq}")
        suffix = ("?" + "&".join(query)) if query else ""
        call_timeout = (wait + self._grace) if wait else None
        return self._request("GET", f"/v1/jobs/{job_id}{suffix}", timeout=call_timeout)

    def jobs(self, *, deployment_id: str | None = None, state: str | None = None) -> list[dict[str, Any]]:
        query = []
        if deployment_id is not None:
            query.append(f"deploymentId={deployment_id}")
        if state is not None:
            query.append(f"state={state}")
        suffix = ("?" + "&".join(query)) if query else ""
        return self._request("GET", f"/v1/jobs{suffix}").get("jobs", [])

    def job_result(self, job_id: str) -> Any:
        return self._request("GET", f"/v1/jobs/{job_id}/result").get("result")

    def cancel_job(self, job_id: str) -> None:
        self._request("POST", f"/v1/jobs/{job_id}/cancel")

    def forget_job(self, job_id: str) -> None:
        self._request("DELETE", f"/v1/jobs/{job_id}")

    # --- observability -------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return self._request("GET", "/v1/pool")

    def nodes(self) -> list[dict[str, Any]]:
        """Detailed view of every connected node (gpu, VRAM, uptime, heartbeat age)."""
        return self._request("GET", "/v1/nodes").get("nodes", [])

    def node(self, node_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/nodes/{node_id}")

    def accounts(self) -> list[dict[str, Any]]:
        """Connected Colab/Kaggle accounts grouped, with contributed/free capacity."""
        return self._request("GET", "/v1/accounts").get("accounts", [])

    def deployments(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/deployments").get("deployments", [])

    def limits(self) -> dict[str, Any]:
        """Every effective server-side budget, so you can size your own timeouts sanely."""
        return self._request("GET", "/v1/limits")

    def error_codes(self) -> list[dict[str, Any]]:
        """The whole error registry -- useful for a non-Python caller codegenerating an
        enum, or for logging a human-readable message for a code you got back."""
        return self._request("GET", "/v1/errors").get("errors", [])
