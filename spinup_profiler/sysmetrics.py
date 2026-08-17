"""`system.jsonl` — 1 Hz host counters, and the rate maths that turns them into per-span numbers.

Both halves of the format live here so the writer (`run`) and the reader (`parse`) cannot drift.

File format — one JSON object per line, counters are CUMULATIVE (rates are derived at parse time):

    {"t": 1786983364.71, "meta": {"cpu_count": 32, "hostname": "gpu-1", "interval_s": 1.0}}
    {"t": 1786983365.71,
     "net":  {"eth0": {"rx_bytes": 12345, "tx_bytes": 678}},
     "disk": {"nvme0n1": {"read_bytes": 4096, "write_bytes": 8192}},
     "cpu_busy_pct": 12.5,
     "gpu": [{"util_pct": 0, "mem_mib": 3}]}

`cpu_busy_pct` is already a delta over the preceding tick (so the first line omits it); `gpu` is
sampled every 2nd tick and the key is simply absent on the other ticks. Throughput is reported in
decimal MB/s (1 MB = 1e6 bytes), matching how storage/network vendors quote it.
"""

import json
import os
import re
import threading
import time

from .shell import sh

MB = 1e6
GB = 1e9
SECTOR = 512  # /proc/diskstats counts 512-byte sectors regardless of physical block size

# The sampler never records these: they are loopback / synthetic block layers.
SKIP_IFACES = ("lo",)
SKIP_DISK_PREFIXES = ("loop", "ram", "dm-", "zram", "sr")

# Excluded when AGGREGATING (the raw per-device numbers stay in the file): container/bridge and
# overlay interfaces (wireguard, tailscale) carry traffic the physical iface has ALREADY counted,
# and a partition's bytes are also counted by its parent disk -- summing both double-counts.
AGG_SKIP_IFACE_PREFIXES = ("veth", "docker", "br-", "virbr", "tun", "tap", "cni", "flannel",
                           "wg", "tailscale", "zt", "cali")

PART_RE = re.compile(r"^(?P<parent>.+?)p?\d+$")


# --------------------------------------------------------------------- sampling side (`run`)
def read_net():
    """{iface: {rx_bytes, tx_bytes}} from /proc/net/dev."""
    out = {}
    with open("/proc/net/dev") as fh:
        for line in fh.readlines()[2:]:
            name, _, rest = line.partition(":")
            name = name.strip()
            if not name or name in SKIP_IFACES:
                continue
            f = rest.split()
            if len(f) < 9:
                continue
            out[name] = {"rx_bytes": int(f[0]), "tx_bytes": int(f[8])}
    return out


def read_disk():
    """{device: {read_bytes, write_bytes}} from /proc/diskstats (sectors x 512)."""
    out = {}
    with open("/proc/diskstats") as fh:
        for line in fh:
            f = line.split()
            if len(f) < 10:
                continue
            name = f[2]
            if name.startswith(SKIP_DISK_PREFIXES):
                continue
            out[name] = {"read_bytes": int(f[5]) * SECTOR, "write_bytes": int(f[9]) * SECTOR}
    return out


def read_cpu_totals():
    """(busy_jiffies, total_jiffies) from the aggregate line of /proc/stat."""
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("cpu "):
                v = [int(x) for x in line.split()[1:]]
                total = sum(v)
                idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
                return total - idle, total
    return None


def read_gpu():
    """[{util_pct, mem_mib}] per GPU, or None when nvidia-smi is unavailable."""
    rc, out, _err = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits"], timeout=5)
    if rc != 0:
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            gpus.append({"util_pct": int(parts[0]), "mem_mib": int(parts[1])})
    return gpus or None


def available():
    """True when this box exposes the counters at all (the auto-default for --sample-system)."""
    return os.access("/proc/net/dev", os.R_OK)


class Sampler:
    """Background 1 Hz sampler. Every tick is wrapped: it must never take the run down with it."""

    def __init__(self, path, interval=1.0, gpu_every=2, sample_gpu=True, max_errors=5):
        self.path = path
        self.interval = max(0.1, float(interval))
        self.gpu_every = max(1, int(gpu_every))
        self.sample_gpu = sample_gpu
        self.max_errors = max_errors
        self.ticks = 0
        self.errors = 0
        self.stopped_early = None
        self._stop = threading.Event()
        self._fh = None
        self._thread = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        try:
            self._fh = open(self.path, "a", buffering=1)
            self._write({"t": time.time(), "meta": {
                "cpu_count": os.cpu_count(),
                "hostname": os.uname().nodename,
                "interval_s": self.interval,
                "gpu_every": self.gpu_every,
                "writer": "spinup-profiler sysmetrics",
            }})
        except Exception as e:  # a broken sampler must not stop the measurement
            self.stopped_early = "could not open %s: %s" % (self.path, e)
            return self
        self._thread = threading.Thread(target=self._loop, name="sysmetrics", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout=5.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass
        return self

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    # -- internals ---------------------------------------------------------
    def _write(self, obj):
        self._fh.write(json.dumps(obj) + "\n")

    def _sample(self, prev_cpu, tick):
        rec = {"t": time.time()}
        try:
            rec["net"] = read_net()
        except Exception:
            pass
        try:
            rec["disk"] = read_disk()
        except Exception:
            pass
        cpu = None
        try:
            cpu = read_cpu_totals()
            if cpu and prev_cpu:
                d_busy, d_total = cpu[0] - prev_cpu[0], cpu[1] - prev_cpu[1]
                if d_total > 0:
                    rec["cpu_busy_pct"] = round(100.0 * d_busy / d_total, 2)
        except Exception:
            pass
        if self.sample_gpu and tick % self.gpu_every == 0:
            try:
                gpus = read_gpu()
                if gpus is None:
                    self.sample_gpu = False  # absent -> stop asking, silently
                else:
                    rec["gpu"] = gpus
            except Exception:
                self.sample_gpu = False
        return rec, cpu

    def _loop(self):
        prev_cpu = None
        try:
            prev_cpu = read_cpu_totals()
        except Exception:
            pass
        next_t = time.time()
        while True:
            try:
                rec, cpu = self._sample(prev_cpu, self.ticks)
                if cpu:
                    prev_cpu = cpu
                self._write(rec)
                self.ticks += 1
                self.errors = 0
            except Exception as e:
                self.errors += 1
                if self.errors >= self.max_errors:
                    self.stopped_early = "sampling failed %d times in a row: %s" % (self.errors, e)
                    return
            next_t += self.interval
            if self._stop.wait(max(0.0, next_t - time.time())):
                return


# --------------------------------------------------------------------- merge side (`parse`)
def _agg_net(net):
    rx = tx = 0
    for name, v in (net or {}).items():
        if name.startswith(AGG_SKIP_IFACE_PREFIXES):
            continue
        rx += v.get("rx_bytes") or 0
        tx += v.get("tx_bytes") or 0
    return rx, tx


def _whole_disks(names):
    """Drop partitions whose parent disk is also present, so bytes are not counted twice."""
    keep = []
    for name in names:
        m = PART_RE.match(name)
        if m and m.group("parent") in names:
            continue
        keep.append(name)
    return keep


def _agg_disk(disk):
    if not disk:
        return 0, 0
    names = _whole_disks(set(disk))
    rd = sum((disk[n].get("read_bytes") or 0) for n in names)
    wr = sum((disk[n].get("write_bytes") or 0) for n in names)
    return rd, wr


def read_samples(path):
    """(meta, samples) from a system.jsonl; unparseable lines are skipped."""
    meta, samples = {}, []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict) or "t" not in d:
                continue
            try:
                d["t"] = float(d["t"])
            except (TypeError, ValueError):
                continue
            if isinstance(d.get("meta"), dict):
                meta.update(d["meta"])
                continue
            samples.append(d)
    samples.sort(key=lambda s: s["t"])
    return meta, samples


# metric name -> (kind, unit) ; "interval" metrics describe the gap between two samples,
# "point" metrics are instantaneous gauges.
INTERVAL_METRICS = ("net_rx_mbs", "net_tx_mbs", "disk_read_mbs", "disk_write_mbs", "cpu_pct")
POINT_METRICS = ("gpu_util_pct", "gpu_mem_mib")


def build_series(samples):
    """Cumulative counters -> {intervals: [...], points: [...], totals: {...}}."""
    intervals, points = [], []
    totals = {"net_rx_bytes": 0, "net_tx_bytes": 0, "disk_read_bytes": 0, "disk_write_bytes": 0}
    ifaces, disks = set(), set()
    prev = None
    for s in samples:
        for g in s.get("gpu") or []:
            points.append({"t": s["t"], "gpu_util_pct": g.get("util_pct"),
                           "gpu_mem_mib": g.get("mem_mib")})
        ifaces.update(n for n in (s.get("net") or {}) if not n.startswith(AGG_SKIP_IFACE_PREFIXES))
        disks.update(_whole_disks(set(s.get("disk") or {})))
        if prev is not None:
            dt = s["t"] - prev["t"]
            if dt > 0:
                iv = {"t0": prev["t"], "t1": s["t"], "dt": dt}
                if s.get("net") and prev.get("net"):
                    rx0, tx0 = _agg_net(prev["net"])
                    rx1, tx1 = _agg_net(s["net"])
                    d_rx, d_tx = rx1 - rx0, tx1 - tx0
                    if d_rx >= 0 and d_tx >= 0:  # a negative delta means a counter reset
                        iv["net_rx_mbs"] = (d_rx / MB) / dt
                        iv["net_tx_mbs"] = (d_tx / MB) / dt
                        totals["net_rx_bytes"] += d_rx
                        totals["net_tx_bytes"] += d_tx
                if s.get("disk") and prev.get("disk"):
                    rd0, wr0 = _agg_disk(prev["disk"])
                    rd1, wr1 = _agg_disk(s["disk"])
                    d_rd, d_wr = rd1 - rd0, wr1 - wr0
                    if d_rd >= 0 and d_wr >= 0:
                        iv["disk_read_mbs"] = (d_rd / MB) / dt
                        iv["disk_write_mbs"] = (d_wr / MB) / dt
                        totals["disk_read_bytes"] += d_rd
                        totals["disk_write_bytes"] += d_wr
                if s.get("cpu_busy_pct") is not None:
                    iv["cpu_pct"] = float(s["cpu_busy_pct"])
                intervals.append(iv)
        prev = s
    return {"intervals": intervals, "points": points, "totals": totals,
            "ifaces": sorted(ifaces), "disks": sorted(disks),
            "t_first": samples[0]["t"] if samples else None,
            "t_last": samples[-1]["t"] if samples else None}


def window_stats(series, t_start, t_end):
    """avg (duration-weighted) + peak per metric over [t_start, t_end]; None if no coverage."""
    if t_start is None or t_end is None or t_end <= t_start:
        return None
    out, covered, n_iv = {}, 0.0, 0
    for metric in INTERVAL_METRICS:
        weighted = span = 0.0
        peak = None
        for iv in series["intervals"]:
            if metric not in iv:
                continue
            overlap = min(t_end, iv["t1"]) - max(t_start, iv["t0"])
            if overlap <= 0:
                continue
            weighted += iv[metric] * overlap
            span += overlap
            peak = iv[metric] if peak is None else max(peak, iv[metric])
        if span > 0:
            out[metric] = {"avg": round(weighted / span, 2), "peak": round(peak, 2)}
            covered = max(covered, span)
    for iv in series["intervals"]:
        if min(t_end, iv["t1"]) - max(t_start, iv["t0"]) > 0:
            n_iv += 1
    pts = {m: [] for m in POINT_METRICS}
    for p in series["points"]:
        if t_start <= p["t"] <= t_end:
            for m in POINT_METRICS:
                if p.get(m) is not None:
                    pts[m].append(float(p[m]))
    for m, vals in pts.items():
        if vals:
            out[m] = {"avg": round(sum(vals) / len(vals), 2), "peak": round(max(vals), 2)}
    if not out:
        return None
    out["samples"] = n_iv
    out["covered_s"] = round(covered, 2)
    return out


def summarize(series, meta):
    t0, t1 = series["t_first"], series["t_last"]
    tot = series["totals"]
    peaks = {}
    for metric in INTERVAL_METRICS:
        vals = [iv[metric] for iv in series["intervals"] if metric in iv]
        if vals:
            peaks["peak_" + metric] = round(max(vals), 2)
    out = {
        "source": "system.jsonl",
        "samples": len(series["intervals"]) + 1,
        "window_s": round(t1 - t0, 2) if (t0 and t1) else None,
        "net_rx_gb": round(tot["net_rx_bytes"] / GB, 3),
        "net_tx_gb": round(tot["net_tx_bytes"] / GB, 3),
        "disk_read_gb": round(tot["disk_read_bytes"] / GB, 3),
        "disk_write_gb": round(tot["disk_write_bytes"] / GB, 3),
        "ifaces": series["ifaces"],
        "disks": series["disks"],
        "gpu_samples": len(series["points"]),
        "cpu_count": meta.get("cpu_count"),
        "hostname": meta.get("hostname"),
        "interval_s": meta.get("interval_s"),
        "units": "MB/s and GB are decimal (1e6 / 1e9 bytes); cpu_pct is over ALL cores",
    }
    out.update(peaks)
    return out
