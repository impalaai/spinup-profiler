#!/usr/bin/env python3
"""`probe` + `--prefetch` tests — plain python, no pytest, no GPU, no real network.

Run:  python3 tests/test_probe.py     (exit 0 = pass, 1 = fail)

The disk probe runs for real on a temp dir with tiny sizes, the network probe runs against an
in-process HTTP server (so no HF traffic), and the ceilings-vs-profile comparison runs against the
committed example profile plus a synthetic probe document.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import threading
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spinup_profiler import prefetch, probe  # noqa: E402
from spinup_profiler.cli import build_parser  # noqa: E402
from spinup_profiler.parse import build_profile  # noqa: E402
from spinup_profiler.report import render  # noqa: E402

FAILURES = []
CHECKS = [0]


def check(label, cond, detail=""):
    CHECKS[0] += 1
    if cond:
        print("  ok   %s" % label)
    else:
        print("  FAIL %s %s" % (label, detail))
        FAILURES.append(label)


def near(label, got, want, tol):
    check("%s = %s (want %s +/- %s)" % (label, got, want, tol),
          got is not None and abs(got - want) <= tol,
          "" if got is not None else "(missing)")


class Payload(BaseHTTPRequestHandler):
    """Serves a compressible-but-real byte stream, honouring HEAD and Range like a CDN."""

    BLOB = b"x" * (1 << 20)

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(64 << 20))
        self.end_headers()

    def do_GET(self):
        want = 8 << 20
        self.send_response(206 if self.headers.get("Range") else 200)
        self.send_header("Content-Length", str(want))
        self.end_headers()
        sent = 0
        try:
            while sent < want:
                n = min(len(self.BLOB), want - sent)
                self.wfile.write(self.BLOB[:n])
                sent += n
        except Exception:
            pass


def start_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Payload)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/blob" % srv.server_address[1]


SYNTHETIC_PROBE = {
    "schema": "spinup-profiler/probe/v1",
    "probes": [
        {"probe": "env", "iface": "eth0", "nproc": 32, "can_drop_caches": True},
        {"probe": "disk_write_direct", "mbs": 1200.0, "size_mb": 4096, "s": 3.4},
        {"probe": "disk_read_direct_1job", "mbs": 463.0, "read_mb": 4096, "s": 8.8},
        {"probe": "disk_read_direct_4job", "mbs": 1688.0, "read_mb": 4096, "s": 2.4},
        {"probe": "disk_read_direct_8job", "mbs": 3697.0, "read_mb": 4096, "s": 1.1},
        {"probe": "net_single_stream", "mbs": 46.0, "gb": 1.1, "s": 25.0},
        {"probe": "net_parallel_4t", "mbs": 180.0, "gb": 3.6, "s": 20.0},
        {"probe": "net_parallel_16t", "mbs": 385.0, "gb": 7.7, "s": 20.0},
        {"probe": "net_parallel_32t", "mbs": 372.0, "gb": 7.4, "s": 20.0},
        {"probe": "done"},
    ],
}


def main():
    tmp = tempfile.mkdtemp(prefix="spinup-probe-")
    try:
        # ------------------------------------------------------------ prefetch
        print("prefetch")
        data = os.path.join(tmp, "weights")
        os.makedirs(data)
        total = 0
        for i in range(4):
            path = os.path.join(data, "shard-%d.safetensors" % i)
            payload = os.urandom(1 << 20) * 2  # 2 MiB each
            with open(path, "wb") as fh:
                fh.write(payload)
            total += len(payload)
        with open(os.path.join(data, "config.json"), "w") as fh:
            fh.write("{}")
        total += 2

        w = prefetch.warm(data, jobs=4)
        check("all files found", w["files"] == 5, "(got %r)" % w["files"])
        check("read every byte", w["bytes"] == total, "(got %r, want %r)" % (w["bytes"], total))
        check("jobs capped at the work available", w["jobs"] <= 4)
        check("rate reported", isinstance(w["mbs"], float) and w["mbs"] > 0, "(got %r)" % w["mbs"])
        check("no read errors", w["errors"] is None, "(got %r)" % w["errors"])
        g = prefetch.warm(os.path.join(data, "*.safetensors"), jobs=8)
        check("glob selects only the shards", g["files"] == 4, "(got %r)" % g["files"])
        check("glob byte count", g["bytes"] == total - 2, "(got %r)" % g["bytes"])
        big = os.path.join(tmp, "one-big.bin")
        with open(big, "wb") as fh:
            fh.write(b"\0" * (3 * prefetch.SEGMENT + 7))
        segs = prefetch._segments([big])
        check("one big file still splits into parallel segments", len(segs) == 4,
              "(got %d)" % len(segs))
        check("segments cover the whole file",
              sum(s[2] for s in segs) == 3 * prefetch.SEGMENT + 7)
        empty = prefetch.warm(os.path.join(tmp, "nothing-here-*"), jobs=4)
        check("a spec matching nothing is not an error",
              empty["files"] == 0 and empty["bytes"] == 0)

        # ------------------------------------------------------------ prefetch span in a profile
        print("prefetch span")
        run = os.path.join(tmp, "run1")
        os.makedirs(run)
        shutil.copy(os.path.join(ROOT, "examples", "docker.log"), run)
        meta = json.load(open(os.path.join(ROOT, "examples", "meta.json")))
        t0 = meta["t_container_start"]
        meta["t_launch"] = t0 - 0.31
        json.dump(meta, open(os.path.join(run, "meta.json"), "w"))
        with open(os.path.join(run, "events.jsonl"), "w") as fh:
            fh.write(json.dumps({"ev": "prefetch_start", "t": t0 - 8.71}) + "\n")
            fh.write(json.dumps({"ev": "prefetch_done", "t": t0 - 0.31, "bytes": 31000000000,
                                 "gb": 31.0, "mbs": 3690.5, "files": 11, "jobs": 8,
                                 "segments": 124}) + "\n")
        p = build_profile(run)
        pf = next((s for s in p["spans"] if s["name"] == "prefetch"), None)
        check("prefetch is a span", pf is not None)
        near("prefetch duration", pf["dur_s"], 8.40, 0.02)
        check("prefetch carries bytes", pf["bytes"] == 31000000000, "(got %r)" % pf.get("bytes"))
        near("prefetch rate", pf["mbs"], 3690.5, 0.1)
        # 8.40s prefetch + 0.31s launch->container sit in front of the container timeline
        near("host prefix", p["host_prefix_s"], 8.71, 0.02)
        check("prefetch counted in total_profiled_s",
              abs(p["total_profiled_s"] - (p["total_s"] + p["host_prefix_s"])) < 0.05,
              "(got %r vs %r + %r)" % (p["total_profiled_s"], p["total_s"], p["host_prefix_s"]))
        check("prefetch is a host span", pf["axis"] == "host" and pf["level"] == 0)
        check("span_prefetch flat key", p.get("span_prefetch") == pf["dur_s"])
        text = render(p)
        check("report shows the prefetch span", " prefetch" in text and "3690.5 MB/s" in text,
              "(header: %s)" % [l for l in text.splitlines() if "prefetch" in l][:2])

        # ------------------------------------------------------------ probe: disk
        print("probe disk (real dd on a temp dir, tiny sizes)")
        em = probe.Emitter(quiet=True)
        probe.probe_env(em, tmp)
        env = em.get("env")
        check("env probe reports cores", env["nproc"] == os.cpu_count())
        check("env probe reports the target", env["target"] == os.path.abspath(tmp))
        check("env probe reports whether caches can be dropped",
              isinstance(env["can_drop_caches"], bool))
        probe.probe_disk(em, tmp, size_mb=8, jobs_levels=[1, 4])
        wrote = em.get("disk_write_direct") or em.get("disk_write_buffered")
        check("disk write measured", wrote and wrote["mbs"] > 0, "(got %r)" % wrote)
        for n in (1, 4):
            r = em.get("disk_read_direct_%djob" % n)
            check("disk read %d-job measured" % n, r and r["mbs"] > 0, "(got %r)" % r)
            check("disk read %d-job had no failed jobs" % n, r and r["failed_jobs"] == 0,
                  "(got %r)" % (r or {}).get("failed_jobs"))
            check("disk read %d-job read the expected MB" % n, r and r["read_mb"] == 8)
        check("probe cleaned up after itself",
              not any(x.startswith("spinup-probe.") for x in os.listdir(tmp)),
              "(left: %r)" % os.listdir(tmp))

        # ------------------------------------------------------------ probe: net (local server)
        print("probe net (in-process HTTP server)")
        srv, url = start_server()
        try:
            em2 = probe.Emitter(quiet=True)
            probe.probe_net(em2, [url], thread_levels=[2, 4], wall_s=2)
            single = em2.get("net_single_stream")
            check("single stream measured", single and single["mbs"] > 0, "(got %r)" % single)
            for n in (2, 4):
                r = em2.get("net_parallel_%dt" % n)
                check("%d-thread level measured" % n, r and r["mbs"] > 0, "(got %r)" % r)
                check("%d-thread level records the thread count" % n, r and r["threads"] == n)
            cl = probe.ceilings(em2.records)
            check("net ceilings derived", cl["net_best_mbs"] and cl["net_single_mbs"],
                  "(got %r)" % cl)
            check("net_scales is a decision", isinstance(cl["net_scales"], bool))
        finally:
            srv.shutdown()

        em3 = probe.Emitter(quiet=True)
        probe.probe_net(em3, ["http://127.0.0.1:1/nope"], thread_levels=[4], wall_s=1)
        check("an unreachable URL is a skip, not a crash",
              em3.get("net_skipped") is not None and em3.get("net_single_stream") is None)

        # ------------------------------------------------------------ ceilings vs profile
        print("ceilings vs profile")
        cl = probe.ceilings(SYNTHETIC_PROBE["probes"])
        near("disk 1-job ceiling", cl["disk_read_1job_mbs"], 463.0, 0.1)
        near("disk best ceiling", cl["disk_read_best_mbs"], 3697.0, 0.1)
        check("disk scales", cl["disk_scales"] is True)
        near("net single ceiling", cl["net_single_mbs"], 46.0, 0.1)
        near("net best ceiling", cl["net_best_mbs"], 385.0, 0.1)
        check("net scales", cl["net_scales"] is True)

        # a profile whose weight_load pulled 250 MB/s over the network
        sysdata = {"net_rx_mbs": {"avg": 250.0, "peak": 260.0},
                   "disk_read_mbs": {"avg": 50.0, "peak": 60.0}}
        fake_profile = {"spans": [
            {"name": "weight_load", "level": 0, "dur_s": 125.75, "sys": sysdata},
            {"name": "image_pull", "level": 0, "dur_s": 141.6,
             "sys": {"net_rx_mbs": {"avg": 44.0, "peak": 46.0}}},
            {"name": "compile", "level": 1, "dur_s": 49.2,
             "sys": {"net_rx_mbs": {"avg": 0.0, "peak": 0.0}}},
        ]}
        cl2, rows = probe.compare_to_profile(SYNTHETIC_PROBE["probes"], fake_profile)
        by = {r["stage"]: r for r in rows}
        check("idle stages are dropped from the comparison", "compile" not in by)
        check("weight_load compared on the network", by["weight_load"]["subsystem"] == "network")
        near("weight_load headroom", by["weight_load"]["headroom_x"], 385.0 / 250.0, 0.02)
        check("weight_load verdict is underused-link",
              by["weight_load"]["verdict"] == "underused-link", "(got %r)" % by["weight_load"])
        check("image_pull at 44 of 385 is also underused-link",
              by["image_pull"]["verdict"] == "underused-link")

        # a stage already at the ceiling, on a link that does not scale
        flat = [r for r in SYNTHETIC_PROBE["probes"]
                if not r["probe"].startswith("net_parallel")]
        flat = flat + [{"probe": "net_parallel_16t", "mbs": 48.0}]
        cl3, rows3 = probe.compare_to_profile(flat, {"spans": [
            {"name": "weight_load", "level": 0, "dur_s": 100.0,
             "sys": {"net_rx_mbs": {"avg": 45.0, "peak": 46.0}}}]})
        check("a flat link is device-capped",
              rows3[0]["verdict"] == "device-capped", "(got %r)" % rows3[0])
        v, why = probe.verdict_for(400.0, 400.0, True)
        check("at-ceiling on a scaling subsystem", v == "at-ceiling", "(got %r)" % v)
        v, why = probe.verdict_for(10.0, None, None)
        check("no ceiling => unknown", v == "unknown")

        lines = probe.render_table(SYNTHETIC_PROBE["probes"])
        table = "\n".join(lines)
        for needle in ("RESOURCE CEILINGS", "disk_read_direct_8job", "net_parallel_16t",
                       "CEILINGS:", "scales"):
            check("table shows %r" % needle, needle in table)
        comp = "\n".join(probe.render_comparison(cl2, rows))
        for needle in ("OBSERVED PER-STAGE RATES", "HEADROOM", "VERDICT", "weight_load",
                       "underused-link"):
            check("comparison table shows %r" % needle, needle in comp)

        # ------------------------------------------------------------ probe.json feeds the report
        print("probe.json -> report hint")
        # a second run dir WITHOUT prefetch events: this is the run that should be told to prefetch
        run2 = os.path.join(tmp, "run2")
        os.makedirs(run2)
        shutil.copy(os.path.join(ROOT, "examples", "docker.log"), run2)
        shutil.copy(os.path.join(ROOT, "examples", "meta.json"), run2)
        json.dump({"schema": "spinup-profiler/probe/v1", "probes": SYNTHETIC_PROBE["probes"],
                   "ceilings": cl}, open(os.path.join(run2, "probe.json"), "w"))
        with open(os.path.join(run2, "system.jsonl"), "w") as fh:
            fh.write(json.dumps({"t": t0, "meta": {"cpu_count": 32}}) + "\n")
            rd = 0
            for i in range(472):
                # disk-bound weight_load at ~500 MB/s, nothing else doing IO
                fh.write(json.dumps({"t": t0 + i,
                                     "net": {"eth0": {"rx_bytes": 0, "tx_bytes": 0}},
                                     "disk": {"nvme0n1": {"read_bytes": int(rd),
                                                          "write_bytes": 0}},
                                     "cpu_busy_pct": 10.0}) + "\n")
                if 77.58 <= i < 203.33:
                    rd += 500 * 1000000
        p2 = build_profile(run2)
        check("ceilings land in the profile",
              (p2.get("ceilings") or {}).get("disk_read_best_mbs") == 3697.0,
              "(got %r)" % p2.get("ceilings"))
        wl = next(s for s in p2["spans"] if s["name"] == "weight_load")
        near("weight_load reads at ~500 MB/s", wl["sys"]["disk_read_mbs"]["avg"], 500.0, 6.0)
        text2 = render(p2)
        check("weight_load called DISK-bound", "weight_load is DISK-bound" in text2)
        check("prefetch is suggested", "--prefetch" in text2,
              "(hints: %s)" % [l.strip() for l in text2.splitlines() if "bound" in l])
        check("the suggestion quotes the measured ceiling",
              "3697.0 MB/s with parallel readers" in text2.replace("\n", " ").replace("    ", ""),
              "(got: %s)" % [l.strip() for l in text2.splitlines() if "measured this disk" in l])
        check("the suggestion estimates the saving",
              "would take ~17s instead of 126s" in text2.replace("\n", " ").replace("    ", ""),
              "(got: %s)" % [l.strip() for l in text2.splitlines() if "would take" in l])
        check("report header shows the measured ceilings", "measured ceilings" in text2)
        # ...but a run that ALREADY prefetches must not be told to prefetch again
        shutil.copy(os.path.join(run2, "probe.json"), run)
        shutil.copy(os.path.join(run2, "system.jsonl"), run)
        text3 = render(build_profile(run))
        check("a run that already prefetches is not told to prefetch",
              "weight_load is DISK-bound" in text3 and "--prefetch <weights dir>" not in text3)

        # ------------------------------------------------------------ CLI wiring
        print("cli")
        args = build_parser().parse_args(["probe", "--target", tmp, "--no-disk", "--no-net",
                                          "--out", os.path.join(tmp, "probe-cli.json")])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = probe.cmd_probe(args)
        check("probe --no-disk --no-net exits 0", rc == 0)
        doc = json.load(open(os.path.join(tmp, "probe-cli.json")))
        check("probe writes its JSON document", doc["schema"] == "spinup-profiler/probe/v1")
        check("probe JSON has an env probe and a done marker",
              doc["probes"][0]["probe"] == "env" and doc["probes"][-1]["probe"] == "done")
        check("JSON lines were printed", '"probe": "env"' in buf.getvalue())
        args2 = build_parser().parse_args(["probe", "--s3-prefix", "s3://x/y", "--no-disk",
                                           "--no-net"])
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            probe.cmd_probe(args2)
        check("s3 needs both flags", "both --s3-env and --s3-prefix are required" in buf2.getvalue())
        check("_levels parses a spec", probe._levels("1,4, 8") == [1, 4, 8])
        check("_levels rejects junk", probe._levels("x,-1,0") == [1])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n%d checks, %d failures" % (CHECKS[0], len(FAILURES)))
    if FAILURES:
        print("FAILED: %s" % ", ".join(FAILURES))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
