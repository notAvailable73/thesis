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
import os
import shutil
import urllib.request

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def ensure_archive(data_root: str, filename: str, urls: list[str],
                   extracted_dirname: str | None = None) -> str:
    """Make sure `data_root/filename` exists, fetching it with a browser UA.

    Args:
        data_root: directory the archive belongs in.
        filename:  archive file name (what torchvision expects).
        urls:      candidate URLs, tried in order.
        extracted_dirname: if torchvision already extracted the archive to
            `data_root/<extracted_dirname>`, skip entirely.

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
                while True:
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
