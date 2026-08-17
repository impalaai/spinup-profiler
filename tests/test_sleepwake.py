#!/usr/bin/env python3
"""Regression test for `sleepwake` sleep verification — plain python, no pytest.

Run:  python3 tests/test_sleepwake.py     (exit 0 = pass, 1 = fail)

Case 1 reproduces the field false positive: a server started WITHOUT --enable-sleep-mode answers
POST /sleep with 200 in milliseconds and never sleeps. That must FAIL, not report timings.
Case 2 is a server that really sleeps. Case 3 is --no-verify-sleep, the documented override.
"""

import io
import json
import os
import sys
import threading
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spinup_profiler.cli import build_parser  # noqa: E402
from spinup_profiler.sleepwake import cmd_sleepwake  # noqa: E402

FAILURES = []
CHECKS = [0]


def check(label, cond, detail=""):
    CHECKS[0] += 1
    if cond:
        print("  ok   %s" % label)
    else:
        print("  FAIL %s %s" % (label, detail))
        FAILURES.append(label)


class Handler(BaseHTTPRequestHandler):
    """/sleep always answers 200; whether it *sleeps* depends on server.honours_sleep."""

    def _send(self, code, body=b""):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/is_sleeping":
            return self._send(200, json.dumps({"is_sleeping": self.server.sleeping}).encode())
        self._send(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        path = self.path.split("?")[0]
        if path == "/sleep":
            if self.server.honours_sleep:
                self.server.sleeping = True
            return self._send(200, b"{}")          # 200 even without --enable-sleep-mode
        if path == "/wake_up":
            self.server.sleeping = False
            self.server.woken += 1
            return self._send(200, b"{}")
        if path == "/v1/chat/completions":
            if self.server.sleeping:
                return self._send(503, b'{"error":"sleeping"}')
            return self._send(200, json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"total_tokens": 4}}).encode())
        self._send(404)


def start_server(honours_sleep):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.honours_sleep = honours_sleep
    srv.sleeping = False
    srv.woken = 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


def run_sleepwake(base_url, extra=()):
    """Returns (rc, parsed json, raw stdout). GPU sampling off: this box has no GPU to speak for."""
    argv = ["sleepwake", "--base-url", base_url, "--model", "fake-model", "--level", "1",
            "--timeout", "10", "--first-completion-timeout", "5", "--no-verify-gpu-mem"]
    args = build_parser().parse_args(argv + list(extra))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_sleepwake(args)
    raw = buf.getvalue()
    doc = json.loads(raw[:raw.rindex("}") + 1])
    return rc, doc, raw


def main():
    print("case 1: server WITHOUT --enable-sleep-mode (/sleep 200, never sleeps)")
    srv, url = start_server(honours_sleep=False)
    try:
        rc, doc, raw = run_sleepwake(url)
        check("exits non-zero", rc != 0, "(got %r)" % rc)
        check("error names --enable-sleep-mode",
              "--enable-sleep-mode" in (doc.get("error") or ""), "(got %r)" % doc.get("error"))
        check("is_sleeping_after_sleep recorded as False",
              doc.get("is_sleeping_after_sleep") is False,
              "(got %r)" % doc.get("is_sleeping_after_sleep"))
        check("no first_completion_s is reported for a no-op sleep",
              doc.get("first_completion_s") is None)
        check("is_sleeping bodies kept in steps",
              [s for s in doc["steps"] if s["step"] == "is_sleeping_after_sleep"
               and s.get("response") == {"is_sleeping": False}])
        check("recovery wake_up was attempted", doc.get("wake_up_recovery_attempted") is True)
        check("FAILED line is printed", "FAILED:" in raw)
    finally:
        srv.shutdown()

    print("case 2: server that really sleeps")
    srv, url = start_server(honours_sleep=True)
    try:
        rc, doc, _raw = run_sleepwake(url)
        check("exits zero", rc == 0, "(got %r)" % rc)
        check("no error", doc.get("error") is None, "(got %r)" % doc.get("error"))
        check("is_sleeping_after_sleep is True", doc.get("is_sleeping_after_sleep") is True)
        for key in ("sleep_s", "wake_s", "first_completion_s", "warm_total_s"):
            check("%s is measured" % key, isinstance(doc.get(key), float),
                  "(got %r)" % doc.get(key))
        check("level is reported", doc.get("level") == 1)
        check("gpu_mem is null when sampling is off", doc.get("gpu_mem") is None)
        check("verify_sleep recorded as on", doc.get("verify_sleep") is True)
    finally:
        srv.shutdown()

    print("case 3: --no-verify-sleep overrides the check on a no-op sleep")
    srv, url = start_server(honours_sleep=False)
    try:
        rc, doc, _raw = run_sleepwake(url, extra=["--no-verify-sleep"])
        check("exits zero", rc == 0, "(got %r)" % rc)
        check("timings are reported", isinstance(doc.get("first_completion_s"), float),
              "(got %r)" % doc.get("first_completion_s"))
        check("verify_sleep recorded as off", doc.get("verify_sleep") is False)
        check("is_sleeping_after_sleep still says False",
              doc.get("is_sleeping_after_sleep") is False)
    finally:
        srv.shutdown()

    print("\n%d checks, %d failures" % (CHECKS[0], len(FAILURES)))
    if FAILURES:
        print("FAILED: %s" % ", ".join(FAILURES))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
