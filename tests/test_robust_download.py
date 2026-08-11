"""Regression test for the ensure_archive total_timeout wall-clock cap.

Guards against the "over an hour" hang: a connection that stays technically
alive (bytes keep trickling in) never trips the per-chunk 180s idle timeout,
so without an overall deadline a slow/overloaded host can run indefinitely.
"""
import http.server
import threading
import time

import pytest


class _SlowTrickleHandler(http.server.BaseHTTPRequestHandler):
    """Serves a large Content-Length but drips small chunks slowly forever,
    simulating an alive-but-overloaded host (never a full stall, so the
    per-chunk idle timeout alone would never fire)."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(10 * 1024 * 1024))
        self.end_headers()
        try:
            for _ in range(50):
                self.wfile.write(b"x" * (200 * 1024))
                self.wfile.flush()
                time.sleep(0.3)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client aborted once total_timeout fired -- expected

    def log_message(self, format, *args):
        pass


@pytest.fixture
def slow_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _SlowTrickleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}/archive.tar.gz"
    server.shutdown()
    thread.join(timeout=5)


def test_ensure_archive_aborts_slow_trickle_within_total_timeout(tmp_path, slow_server):
    from src.datasets._robust_download import ensure_archive

    start = time.monotonic()
    with pytest.raises(RuntimeError):
        ensure_archive(str(tmp_path), "archive.tar.gz", [slow_server],
                       total_timeout=1.0)
    elapsed = time.monotonic() - start
    # Full trickle would take ~15s; the old per-chunk idle timeout is 180s.
    # A working cap aborts within a few seconds of total_timeout=1.0s.
    assert elapsed < 10.0, (
        f"ensure_archive took {elapsed:.1f}s -- total_timeout did not abort "
        f"the slow-but-alive connection"
    )


class _UAEchoHandler(http.server.BaseHTTPRequestHandler):
    """Records the User-Agent header of the request it received, so tests can
    assert what ensure_archive actually sent without hitting a real host."""
    received_ua = None

    def do_GET(self):
        _UAEchoHandler.received_ua = self.headers.get("User-Agent")
        body = b"x" * 16
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def ua_echo_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _UAEchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}/archive.bin"
    server.shutdown()
    thread.join(timeout=5)


def test_ensure_archive_default_sends_browser_ua(tmp_path, ua_echo_server):
    """Regression guard for CIFAR's host: the DEFAULT behaviour (every
    pre-Step-9 caller) must keep sending the browser UA that unblocks
    cs.toronto.edu's 403 on urllib's own default UA."""
    from src.datasets._robust_download import ensure_archive, _BROWSER_UA
    ensure_archive(str(tmp_path), "archive.bin", [ua_echo_server])
    assert _UAEchoHandler.received_ua == _BROWSER_UA


def test_ensure_archive_user_agent_none_sends_no_custom_ua(tmp_path, ua_echo_server):
    """Regression guard for Zenodo's host (Step 9): its WAF 403s the exact
    browser UA above (verified live against zenodo.org) while allowing
    urllib's own default UA through -- user_agent=None must send that
    default, not the browser string."""
    from src.datasets._robust_download import ensure_archive
    ensure_archive(str(tmp_path), "archive.bin", [ua_echo_server], user_agent=None)
    assert _UAEchoHandler.received_ua is not None
    assert "Chrome" not in _UAEchoHandler.received_ua
