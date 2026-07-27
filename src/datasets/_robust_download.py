"""Robust dataset-archive download.

torchvision's `download=True` uses a default User-Agent that the CIFAR host
(cs.toronto.edu) now answers with HTTP 403 Forbidden -- this breaks fresh
Colab clones (the `data/` dir is gitignored, so nothing is cached). We
pre-fetch the archive with a browser User-Agent; torchvision's downloader
then finds the verified file locally and skips the network entirely.

If every mirror fails, raise a clear error telling the user to drop the file
into `data/` manually (the function is a no-op once the file is present).
"""
from __future__ import annotations
import hashlib
import os
import shutil
import time
import urllib.request

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Per-chunk `resp.read()` calls already have a 180s idle timeout (below), but
# that only fires if the connection goes fully silent. A connection that
# dribbles in a few bytes just before each 180s window resets the idle clock
# forever -- against an overloaded/rate-limited host (cs.toronto.edu is known
# for this) that can silently "run" for hours. This wall-clock cap bounds the
# total time spent on any one mirror regardless of whether bytes are still
# trickling in.
_DEFAULT_TOTAL_TIMEOUT = 600.0  # 10 minutes per mirror


def ensure_archive(data_root: str, filename: str, urls: list[str],
                   extracted_dirname: str | None = None,
                   total_timeout: float = _DEFAULT_TOTAL_TIMEOUT,
                   md5: str | None = None) -> str:
    """Make sure `data_root/filename` exists, fetching it with a browser UA.

    Args:
        data_root: directory the archive belongs in.
        filename:  archive file name (what torchvision expects).
        urls:      candidate URLs, tried in order.
        extracted_dirname: if torchvision already extracted the archive to
            `data_root/<extracted_dirname>`, skip entirely.
        total_timeout: hard wall-clock cap (seconds) per mirror attempt, on
            top of the per-chunk idle timeout. A mirror that is technically
            alive but too slow to finish in time is abandoned in favor of
            the next one, instead of hanging indefinitely.
        md5: optional expected MD5 of the downloaded file (Step 9: the Zenodo
            mini-ImageNet caches publish one). Only checked against a FRESH
            download in this call (not an already-cached file, so existing
            CIFAR/SVHN/TinyImageNet callers that never pass this incur zero
            extra hashing cost). A mismatch deletes the file and tries the
            next mirror, same as a truncated download.

    Returns the archive path. Raises RuntimeError if all mirrors fail and the
    file is absent.
    """
    os.makedirs(data_root, exist_ok=True)
    dst = os.path.join(data_root, filename)

    if extracted_dirname and os.path.isdir(
        os.path.join(data_root, extracted_dirname)
    ):
        return dst  # already extracted by a previous run
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst  # archive already here; let torchvision verify + extract

    last_err: Exception | None = None
    for url in urls:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": _BROWSER_UA}
            )
            with urllib.request.urlopen(req, timeout=180) as resp, \
                    open(dst, "wb") as fh:
                total = resp.getheader("Content-Length")
                total = int(total) if total else None
                # Chunked copy with periodic progress so a slow/stalled
                # download is VISIBLE (silent copyfileobj on Colab's slow
                # link to the Toronto host looks identical to a hang).
                total_mb = f"{total / 1e6:.0f}" if total else "?"
                print(f"[download] {filename} <- {url}  "
                      f"({total_mb} MB)", flush=True)
                read = 0
                next_report = 16 * 1024 * 1024  # every ~16 MB
                deadline = time.monotonic() + total_timeout
                while True:
                    if time.monotonic() > deadline:
                        pct = f" ({100 * read / total:.0f}%)" if total else ""
                        raise TimeoutError(
                            f"download exceeded total_timeout={total_timeout:.0f}s "
                            f"after {read / 1e6:.0f} MB{pct} -- host is alive but "
                            f"too slow (likely overloaded/rate-limited), abandoning "
                            f"this mirror"
                        )
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    read += len(chunk)
                    if read >= next_report:
                        pct = f" ({100 * read / total:.0f}%)" if total else ""
                        print(f"[download]   {read / 1e6:.0f} MB{pct}",
                              flush=True)
                        next_report += 16 * 1024 * 1024
            # A truncated download (cancelled run, dropped connection) leaves a
            # nonzero file that would otherwise be trusted on the next call —
            # treat a size mismatch as failure and try the next mirror.
            if total is not None and os.path.getsize(dst) != total:
                raise IOError(
                    f"incomplete download: got {os.path.getsize(dst)} of "
                    f"{total} bytes"
                )
            if md5 is not None:
                hasher = hashlib.md5()
                with open(dst, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        hasher.update(chunk)
                got = hasher.hexdigest()
                if got != md5:
                    raise IOError(
                        f"MD5 mismatch for {filename}: expected {md5}, got {got}"
                    )
            if os.path.getsize(dst) > 0:
                print(f"[download] done: {filename} "
                      f"({os.path.getsize(dst) / 1e6:.0f} MB)", flush=True)
                return dst
        except Exception as e:  # noqa: BLE001 - report and try next mirror
            last_err = e
            print(f"[download] FAILED {url}: {e}", flush=True)
            if os.path.exists(dst):
                try:
                    os.remove(dst)
                except OSError:
                    pass

    raise RuntimeError(
        f"Could not download {filename!r} from any mirror "
        f"(last error: {last_err}). Manually place {filename} in "
        f"{os.path.abspath(data_root)} and re-run -- the loader will use it."
    )
