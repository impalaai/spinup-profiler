#!/usr/bin/env python3
"""Per-stage system-metrics tests — plain python, no pytest, no GPU.

Run:  python3 tests/test_system.py     (exit 0 = pass, 1 = fail)

Builds a SYNTHETIC system.jsonl whose events line up with the real example run's timeline
(examples/docker.log + meta.json), so the expected per-span rates are known exactly:

    weight_load  (t0+77.58 .. t0+203.33)   250 MB/s network in, 50 MB/s disk read,
                                           250 MB/s disk written, cpu 25 %, gpu 0 %
    imports      (t0+0     .. t0+20.99)    one core busy (100/32 = 3.13 %), no IO
    compile      (t0+204.75 .. t0+253.96)  cpu 40 %, gpu 10 %  (inductor: CPU-bound)
    warmup+graph (t0+311.45 .. t0+445.84)  gpu 90 %, no IO
"""

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spinup_profiler import sysmetrics  # noqa: E402
from spinup_profiler.parse import build_profile  # noqa: E402
from spinup_profiler.report import io_table, render  # noqa: E402

EX = os.path.join(ROOT, "examples")
CPU_COUNT = 32
MB = 1000000

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


# --- the synthetic load profile: (start_rel, end_rel, rx_mbs, disk_rd_mbs, disk_wr_mbs, cpu, gpu)
PHASES = [
    (0.0,    20.99,  0.0,   0.0,  0.0, 100.0 / CPU_COUNT, 0),   # imports: 1 core
    (20.99,  61.89,  0.0,   0.0,  0.0, 100.0 / CPU_COUNT, 0),   # spawn_workers: 1 core
    (61.89,  77.58,  1.0,   0.0,  0.0, 5.0,               0),   # pre_load
    (77.58,  203.33, 250.0, 50.0, 250.0, 25.0,            0),   # weight_load: network-bound
    (203.33, 253.96, 0.0,   0.0,  0.0, 40.0,              10),  # compile: CPU-bound, GPU idle
    (253.96, 311.45, 0.0,   0.0,  0.0, 10.0,              40),  # mem_profile_tail
    (311.45, 445.84, 0.0,   0.0,  0.0, 12.0,              90),  # warmup_kernels + graph_capture
    (445.84, 471.0,  2.0,   0.0,  0.0, 8.0,               5),   # api_startup
]


def rates_at(rel):
    for a, b, rx, rd, wr, cpu, gpu in PHASES:
        if a <= rel < b:
            return rx, rd, wr, cpu, gpu
    return 0.0, 0.0, 0.0, 1.0, 0


def write_synthetic(path, t0, interval=1.0, span_s=471.0):
    """1 Hz cumulative counters, gpu on every 2nd tick — exactly what the sampler writes."""
    net_rx = net_tx = disk_rd = disk_wr = 0
    with open(path, "w") as fh:
        fh.write(json.dumps({"t": t0, "meta": {"cpu_count": CPU_COUNT, "hostname": "synthetic",
                                               "interval_s": interval, "gpu_every": 2}}) + "\n")
        tick = 0
        rel = 0.0
        while rel <= span_s:
            rx, rd, wr, cpu, gpu = rates_at(rel)
            rec = {"t": round(t0 + rel, 3),
                   "net": {"eth0": {"rx_bytes": int(net_rx), "tx_bytes": int(net_tx)},
                           # a veth pair mirrors container traffic: must NOT be double counted
                           "veth1234": {"rx_bytes": int(net_rx), "tx_bytes": int(net_tx)}},
                   "disk": {"nvme0n1": {"read_bytes": int(disk_rd), "write_bytes": int(disk_wr)},
                            # a partition of the same disk: must NOT be double counted
                            "nvme0n1p1": {"read_bytes": int(disk_rd), "write_bytes": int(disk_wr)}}}
            if tick:
                rec["cpu_busy_pct"] = cpu
            if tick % 2 == 0:
                rec["gpu"] = [{"util_pct": gpu, "mem_mib": 30000 if rel > 203.33 else 500}]
            fh.write(json.dumps(rec) + "\n")
            # advance the cumulative counters over the interval that FOLLOWS this sample
            rx2, rd2, wr2, _c, _g = rates_at(rel + interval / 2.0)
            net_rx += rx2 * MB * interval
            net_tx += 0.05 * MB * interval
            disk_rd += rd2 * MB * interval
            disk_wr += wr2 * MB * interval
            tick += 1
            rel += interval
    return path


def main():
    meta = json.load(open(os.path.join(EX, "meta.json")))
    t0 = meta["t_container_start"]
    tmp = tempfile.mkdtemp(prefix="spinup-sys-")
    try:
        shutil.copy(os.path.join(EX, "docker.log"), os.path.join(tmp, "docker.log"))
        shutil.copy(os.path.join(EX, "meta.json"), os.path.join(tmp, "meta.json"))
        write_synthetic(os.path.join(tmp, "system.jsonl"), t0)

        # the run dir must discover system.jsonl on its own
        p = build_profile(tmp)
        spans = {s["name"]: s for s in p["spans"]}

        print("sys_summary")
        summ = p["sys_summary"]
        check("sys_summary present", bool(summ))
        check("cpu_count came from the meta line", summ["cpu_count"] == CPU_COUNT)
        near("window_s", summ["window_s"], 471.0, 1.5)
        # 250 MB/s for 125.75s of weight_load + 1 MB/s pre_load + 2 MB/s api_startup
        near("net_rx_gb", summ["net_rx_gb"], (250 * 125.75 + 1 * 15.69 + 2 * 25.2) / 1000.0, 0.6)
        near("disk_read_gb", summ["disk_read_gb"], 50 * 125.75 / 1000.0, 0.3)
        near("disk_write_gb", summ["disk_write_gb"], 250 * 125.75 / 1000.0, 0.6)
        near("peak_net_rx_mbs", summ["peak_net_rx_mbs"], 250.0, 1.0)
        check("veth interfaces excluded from the aggregate", summ["ifaces"] == ["eth0"],
              "(got %r)" % summ["ifaces"])
        check("partitions excluded from the aggregate", summ["disks"] == ["nvme0n1"],
              "(got %r)" % summ["disks"])
        check("gpu samples counted", summ["gpu_samples"] > 200, "(got %r)" % summ["gpu_samples"])

        print("per-span rates")
        near("weight_load net_rx avg", spans["weight_load"]["sys"]["net_rx_mbs"]["avg"], 250.0, 3.0)
        near("weight_load net_rx peak", spans["weight_load"]["sys"]["net_rx_mbs"]["peak"], 250.0, 1.0)
        near("weight_load disk_read avg", spans["weight_load"]["sys"]["disk_read_mbs"]["avg"], 50.0, 2.0)
        near("weight_load disk_write avg", spans["weight_load"]["sys"]["disk_write_mbs"]["avg"], 250.0, 6.0)
        near("weight_load cpu avg", spans["weight_load"]["sys"]["cpu_pct"]["avg"], 25.0, 1.0)
        near("imports cpu avg", spans["imports"]["sys"]["cpu_pct"]["avg"], 100.0 / CPU_COUNT, 0.5)
        near("imports net_rx avg", spans["imports"]["sys"]["net_rx_mbs"]["avg"], 0.0, 0.5)
        near("compile cpu avg", spans["compile"]["sys"]["cpu_pct"]["avg"], 40.0, 2.0)
        near("compile gpu avg", spans["compile"]["sys"]["gpu_util_pct"]["avg"], 10.0, 3.0)
        near("warmup_kernels gpu avg", spans["warmup_kernels"]["sys"]["gpu_util_pct"]["avg"], 90.0, 3.0)
        near("graph_capture gpu avg", spans["graph_capture"]["sys"]["gpu_util_pct"]["avg"], 90.0, 3.0)
        check("gpu mem is sampled", spans["warmup_kernels"]["sys"]["gpu_mem_mib"]["avg"] == 30000.0,
              "(got %r)" % spans["warmup_kernels"]["sys"]["gpu_mem_mib"]["avg"])
        check("every level-0 span got sys data",
              all("sys" in s for s in p["spans"] if s["level"] == 0 and s["dur_s"] > 0.5))
        check("spans carry absolute epochs",
              abs(spans["weight_load"]["t_start"] - (t0 + 77.58)) < 0.05)
        check("timing spans unchanged by the merge", p["span_weight_load"] == 125.75,
              "(got %r)" % p["span_weight_load"])

        print("report")
        rows = io_table(p)
        text = render(p)
        check("io_table renders rows", len(rows) > 5, "(got %d lines)" % len(rows))
        for needle in ("PER-STAGE SYSTEM METRICS", "NET RX MB/s", "DISK RD MB/s", "GPU %",
                       "weight_load", "totals over", "sampled: ifaces eth0"):
            check("IO table shows %r" % needle, needle in "\n".join(rows))
        check("report includes the IO table", "PER-STAGE SYSTEM METRICS" in text)
        check("IO table is suppressible", "PER-STAGE SYSTEM METRICS" not in render(p, show_io=False))

        print("attribution hints")
        net_line = [l for l in text.splitlines() if "weight_load is NETWORK-bound" in l]
        check("weight_load called NETWORK-bound", bool(net_line),
              "(bound hints: %s)" % [l.strip() for l in text.splitlines() if "bound" in l])
        if net_line:
            rate = float(net_line[0].split("NETWORK-bound at ")[1].split(" MB/s")[0])
            near("rate quoted in the network hint", rate, 250.0, 3.0)
        check("engine_init does not repeat its leaves' GPU hint",
              "engine_init is GPU compute-bound" not in text)
        check("imports called single-core CPU-bound",
              "imports is single-core CPU-bound" in text)
        check("warmup_kernels called GPU compute-bound",
              "warmup_kernels is GPU compute-bound at 90% utilisation" in text)
        check("compile called CPU-bound with an idle GPU",
              "compile is CPU-bound (inductor) with the GPU at 10%" in text)
        check("weight_load NOT called disk-bound", "weight_load is DISK-bound" not in text)

        print("robustness")
        broken = os.path.join(tmp, "broken.jsonl")
        with open(broken, "w") as fh:
            fh.write("not json\n{\"no_t\": 1}\n{\"t\": \"nan-ish\"}\n")
        p2 = build_profile(os.path.join(tmp, "docker.log"),
                          meta_path=os.path.join(tmp, "meta.json"), system_path=broken)
        check("a broken system.jsonl warns but still profiles",
              p2["total_s"] == 470.06 and p2["sys_summary"] is None
              and any("no usable samples" in w for w in p2["warnings"]),
              "(warnings: %r)" % p2["warnings"])
        p3 = build_profile(os.path.join(tmp, "docker.log"),
                          meta_path=os.path.join(tmp, "meta.json"))
        check("no system.jsonl => no sys keys at all",
              p3["sys_summary"] is None and not any("sys" in s for s in p3["spans"]))

        print("sampler (live, one tick on this box)")
        live = os.path.join(tmp, "live.jsonl")
        s = sysmetrics.Sampler(live, interval=0.2).start()
        import time as _t
        _t.sleep(0.75)
        s.stop()
        m, samples = sysmetrics.read_samples(live)
        check("sampler wrote a meta line", bool(m.get("cpu_count")), "(got %r)" % m)
        check("sampler wrote samples", len(samples) >= 2, "(got %d)" % len(samples))
        check("sampler recorded cumulative net counters",
              all("net" in x for x in samples) if sysmetrics.available() else True)
        check("sampler never raised", s.stopped_early is None, "(got %r)" % s.stopped_early)
        if len(samples) >= 2:
            series = sysmetrics.build_series(samples)
            check("live samples produce intervals", len(series["intervals"]) >= 1)
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
