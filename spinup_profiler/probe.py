"""`probe` — measure a box's resource CEILINGS, so an observed stage rate can be judged.

Ported from the field-proven cross-cloud harness (`probe_suite.sh`, exercised on verda / nebius /
lyceum / runpod). Same probes, stdlib only, and it degrades instead of dying when a box lacks sudo,
a writable /proc, or s5cmd — all of which happened on real boxes.

Sections
  env       interface, link speed, cores, target filesystem, whether caches can be dropped
  disk      direct-IO write, then direct-IO read at 1 / 4 / 8 parallel jobs
  net       one HTTPS stream vs 4 / 16 / 32 parallel range-streams, time-boxed
  s3        optional s5cmd --numworkers sweep against an object-store prefix

With --profile it also prints each stage's OBSERVED rate (from the profiler's sampled counters)
against these ceilings, with headroom and a verdict: at-ceiling / underused-link / device-capped.
"""

import json
import os
import ssl
import subprocess
import sys
import threading
import time

from .shell import sh
from .stages import AT_CEILING, SCALES  # one definition of "at the ceiling" / "it scales"

MB = 1e6
DEFAULT_URLS = [
    "https://huggingface.co/Qwen/Qwen3.5-27B-FP8/resolve/main/"
    "model.safetensors-%05d-of-00011.safetensors" % i
    for i in range(2, 8)
]
CHUNK = 1 << 22  # 4 MiB reads, as in the field script


# --------------------------------------------------------------------------- output plumbing
class Emitter:
    """JSON lines to stdout as each probe finishes (greppable, like the shell original)."""

    def __init__(self, quiet=False):
        self.records = []
        self.quiet = quiet

    def emit(self, probe, **kw):
        rec = {"probe": probe}
        rec.update(kw)
        self.records.append(rec)
        if not self.quiet:
            print(json.dumps(rec), flush=True)
        return rec

    def get(self, probe):
        for r in self.records:
            if r["probe"] == probe:
                return r
        return None

    def all_prefixed(self, prefix):
        return [r for r in self.records if r["probe"].startswith(prefix)]


def note(msg):
    sys.stderr.write("[probe] %s\n" % msg)
    sys.stderr.flush()


# --------------------------------------------------------------------------- env
def default_iface():
    """Interface of the default route, from /proc/net/route (no `ip` dependency)."""
    try:
        with open("/proc/net/route") as fh:
            for line in fh.readlines()[1:]:
                f = line.split()
                if len(f) > 2 and f[1] == "00000000":
                    return f[0]
    except Exception:
        pass
    rc, out, _e = sh(["ip", "route", "show", "default"], timeout=10)
    if rc == 0:
        parts = out.split()
        if "dev" in parts:
            return parts[parts.index("dev") + 1]
    return None


def link_speed_mbit(iface):
    try:
        with open("/sys/class/net/%s/speed" % iface) as fh:
            return int(fh.read().strip())
    except Exception:
        return None


def can_drop_caches():
    """True when we can actually cool the page cache (root, or passwordless sudo, writable /proc)."""
    rc, _o, _e = sh(["sh", "-c",
                     "sync; { echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; } || "
                     "{ echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; }"], timeout=60)
    return rc == 0


def drop_caches():
    sh(["sh", "-c",
        "sync; { echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; } || "
        "{ echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; } || true"], timeout=60)


def probe_env(em, target):
    iface = default_iface()
    free_gb = None
    try:
        st = os.statvfs(target)
        free_gb = round(st.f_bavail * st.f_frsize / 1e9, 1)
    except Exception:
        pass
    dropped = can_drop_caches()
    if not dropped:
        note("cannot drop the page cache here (no root / read-only /proc — normal in a container); "
             "direct-IO probes are still valid, buffered numbers would not be")
    return em.emit("env", iface=iface, link_speed_mbit=link_speed_mbit(iface) if iface else None,
                   nproc=os.cpu_count(), target=os.path.abspath(target), target_free_gb=free_gb,
                   can_drop_caches=dropped, hostname=os.uname().nodename)


# --------------------------------------------------------------------------- disk
def _dd(args, timeout):
    t0 = time.time()
    rc, _out, err = sh(["dd"] + args, timeout=timeout)
    return rc, time.time() - t0, err


def probe_disk(em, target, size_mb, jobs_levels, timeout=1800):
    """Direct-IO write once, then read the same file with N parallel jobs on distinct slices.

    Deviation from the shell original, which copied the test file once per job: N copies of a
    multi-GB file needs N x size on disk (48 GB at 8 jobs), and small root disks are exactly where
    this gets run. Distinct slices of one file exercise the same device parallelism for size_mb of
    space, because iflag=direct bypasses the page cache.
    """
    if sh(["sh", "-c", "command -v dd"], timeout=20)[0] != 0:
        em.emit("disk_skipped", reason="dd not found")
        return
    d = os.path.join(target, "spinup-probe.%d" % os.getpid())
    path = os.path.join(d, "probe.bin")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception as e:
        em.emit("disk_skipped", reason="cannot write to %s: %s" % (target, e))
        return

    direct_write = True
    try:
        note("disk: writing %d MB with direct IO" % size_mb)
        rc, dt, err = _dd(["if=/dev/zero", "of=" + path, "bs=1M", "count=%d" % size_mb,
                           "oflag=direct", "conv=fdatasync"], timeout)
        if rc != 0:
            direct_write = False
            note("direct-IO write refused (%s) — retrying buffered" % (err.strip().splitlines()[-1:] or ""))
            rc, dt, err = _dd(["if=/dev/zero", "of=" + path, "bs=1M", "count=%d" % size_mb,
                               "conv=fdatasync"], timeout)
        if rc != 0:
            em.emit("disk_skipped", reason="dd write failed: %s" % err.strip()[:200])
            return
        em.emit("disk_write_direct" if direct_write else "disk_write_buffered",
                mbs=round(size_mb / dt, 1), size_mb=size_mb, s=round(dt, 2), direct=direct_write)

        for n in jobs_levels:
            drop_caches()
            slice_mb = max(1, size_mb // n)
            procs = []
            t0 = time.time()
            for i in range(n):
                argv = ["dd", "if=" + path, "of=/dev/null", "bs=1M",
                        "skip=%d" % (i * slice_mb), "count=%d" % slice_mb]
                if direct_write:
                    argv.append("iflag=direct")
                procs.append(subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL))
            failed = sum(1 for p in procs if p.wait(timeout) != 0)
            dt = time.time() - t0
            read_mb = slice_mb * n
            em.emit("disk_read_direct_%djob" % n, mbs=round(read_mb / dt, 1), jobs=n,
                    read_mb=read_mb, s=round(dt, 2), failed_jobs=failed,
                    direct=direct_write)
            note("disk: %d job(s) -> %.0f MB/s" % (n, read_mb / dt))
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
        try:
            os.rmdir(d)
        except Exception:
            pass


# --------------------------------------------------------------------------- network
def _ctx():
    return ssl.create_default_context()


def resolve(url, timeout=20):
    """Follow the CDN redirect once, so every thread streams from the final host."""
    import urllib.request
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
            return r.geturl()
    except Exception:
        return url


def timed_stream(url, max_bytes, wall_s):
    import urllib.request
    got, t0 = 0, time.time()
    try:
        with urllib.request.urlopen(url, timeout=30, context=_ctx()) as r:
            while got < max_bytes and time.time() - t0 < wall_s:
                b = r.read(CHUNK)
                if not b:
                    break
                got += len(b)
    except Exception as e:
        return got, time.time() - t0, "%s: %s" % (type(e).__name__, e)
    return got, time.time() - t0, None


def parallel_streams(urls, nthreads, wall_s):
    """N threads, each streaming a different file (or a different 1 GiB offset of one)."""
    import urllib.request
    counts = [0] * nthreads
    errors = [None] * nthreads
    picked = [urls[i % len(urls)] for i in range(nthreads)]
    stop = time.time() + wall_s

    def worker(i):
        start = (i // len(urls)) * (1 << 30)
        try:
            req = urllib.request.Request(picked[i], headers={"Range": "bytes=%d-" % start})
            with urllib.request.urlopen(req, timeout=30, context=_ctx()) as r:
                while time.time() < stop:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    counts[i] += len(b)
        except Exception as e:
            errors[i] = "%s: %s" % (type(e).__name__, e)

    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(nthreads)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join(wall_s + 30)
    dt = time.time() - t0
    return sum(counts), dt, [e for e in errors if e]


def probe_net(em, urls, thread_levels, wall_s, single_cap_gb=3):
    resolved = [resolve(u) for u in urls]
    note("net: single stream for up to %ds" % (wall_s + 5))
    got, dt, err = timed_stream(resolved[0], int(single_cap_gb * (1 << 30)), wall_s + 5)
    if got == 0:
        em.emit("net_skipped", reason=err or "no bytes from %s" % urls[0])
        note("net: no bytes (%s) — skipping the parallel levels" % err)
        return
    em.emit("net_single_stream", mbs=round(got / MB / dt, 1), gb=round(got / 1e9, 2),
            s=round(dt, 1), error=err)
    note("net: single stream -> %.0f MB/s" % (got / MB / dt))
    for n in thread_levels:
        got, dt, errs = parallel_streams(resolved, n, wall_s)
        em.emit("net_parallel_%dt" % n, mbs=round(got / MB / dt, 1), gb=round(got / 1e9, 2),
                s=round(dt, 1), threads=n, errors=errs[:3] or None)
        note("net: %d threads -> %.0f MB/s" % (n, got / MB / dt))


# --------------------------------------------------------------------------- object store
def load_env_file(path):
    """KEY=VALUE lines (what the field harness sourced with `set -a`)."""
    env = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            k, _, v = line.partition("=")
            if k:
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def probe_s3(em, s3_env, s3_prefix, worker_levels, cap_gb=8.0, dest=None):
    """s5cmd --numworkers sweep. Objects are LISTED first, deliberately.

    The shell original globbed `...0000[2-5]-of-00011.safetensors`: s5cmd does not support
    character classes, so it matched nothing and the sweep reported 0 bytes. Listing and then
    copying named objects also lets the byte cap be exact.
    """
    if not os.path.exists(s3_env):
        em.emit("s3_skipped", reason="env file not found: %s" % s3_env)
        return
    if sh(["sh", "-c", "command -v s5cmd"], timeout=20)[0] != 0:
        em.emit("s3_skipped", reason="s5cmd not installed (get it from "
                                     "github.com/peak/s5cmd/releases; not auto-installing)")
        return
    env = dict(os.environ)
    env.update(load_env_file(s3_env))
    endpoint = env.get("AWS_ENDPOINT_URL")
    base = ["s5cmd"] + (["--endpoint-url", endpoint] if endpoint else [])

    rc, out, err = _run_env(base + ["ls", s3_prefix.rstrip("/") + "/*"], env, 120)
    if rc != 0:
        em.emit("s3_skipped", reason="s5cmd ls failed: %s" % (err or out).strip()[:200])
        return
    objects = []
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 4 and f[-2].isdigit():
            objects.append((f[-1], int(f[-2])))
        elif len(f) >= 2 and f[-2].isdigit():
            objects.append((f[-1], int(f[-2])))
    if not objects:
        em.emit("s3_skipped", reason="no objects under %s" % s3_prefix)
        return
    chosen, total = [], 0
    for name, size in objects:
        if total and total + size > cap_gb * 1e9:
            break
        chosen.append(name)
        total += size
    em.emit("s3_objects", prefix=s3_prefix, objects=len(objects), chosen=len(chosen),
            gb=round(total / 1e9, 2), endpoint=endpoint)

    dest_root = dest or ("/dev/shm" if os.path.isdir("/dev/shm") else os.path.expanduser("~"))
    for nw in worker_levels:
        d = os.path.join(dest_root, "spinup-probe-s3")
        sh(["rm", "-rf", d], timeout=120)
        os.makedirs(d, exist_ok=True)
        runfile = os.path.join(d, "cmds.txt")
        with open(runfile, "w") as fh:
            for name in chosen:
                fh.write("cp %s/%s %s/\n" % (s3_prefix.rstrip("/"), name, d))
        t0 = time.time()
        rc, out, err = _run_env(base + ["--numworkers", str(nw), "run", runfile], env, 3600)
        dt = time.time() - t0
        got = 0
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f != "cmds.txt":
                    try:
                        got += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        em.emit("s3_sweep_nw%d" % nw, mbs=round(got / MB / dt, 1) if dt else None,
                gb=round(got / 1e9, 2), s=round(dt, 1), numworkers=nw,
                error=None if rc == 0 else (err or out).strip()[:200])
        note("s3: %d workers -> %.0f MB/s" % (nw, got / MB / dt if dt else 0))
        sh(["rm", "-rf", d], timeout=120)


def _run_env(argv, env, timeout):
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, env=env)
        return (p.returncode, p.stdout.decode("utf-8", "replace"),
                p.stderr.decode("utf-8", "replace"))
    except subprocess.TimeoutExpired:
        return 124, "", "timeout after %ss" % timeout
    except FileNotFoundError as e:
        return 127, "", str(e)


# --------------------------------------------------------------------------- ceilings + verdicts
def ceilings(records):
    """The comparable numbers a profile can be judged against."""
    def best(prefix):
        vals = [r["mbs"] for r in records
                if r["probe"].startswith(prefix) and isinstance(r.get("mbs"), (int, float))]
        return max(vals) if vals else None

    one = None
    for r in records:
        if r["probe"] == "disk_read_direct_1job":
            one = r.get("mbs")
    disk_best = best("disk_read_direct")
    net_single = None
    for r in records:
        if r["probe"] == "net_single_stream":
            net_single = r.get("mbs")
    net_par = best("net_parallel")
    net_best = max([v for v in (net_single, net_par) if v is not None] or [0]) or None
    out = {
        "disk_read_1job_mbs": one,
        "disk_read_best_mbs": disk_best,
        "disk_write_mbs": best("disk_write"),
        "disk_scales": (bool(one and disk_best and disk_best >= SCALES * one)
                        if (one and disk_best) else None),
        "net_single_mbs": net_single,
        "net_parallel_best_mbs": net_par,
        "net_best_mbs": net_best,
        "net_scales": (bool(net_single and net_par and net_par >= SCALES * net_single)
                       if (net_single and net_par) else None),
        "s3_best_mbs": best("s3_sweep"),
    }
    return out


def verdict_for(observed, ceiling, scales):
    """at-ceiling / device-capped / underused-link, plus the reason."""
    if not ceiling or observed is None:
        return "unknown", "no ceiling measured for this subsystem"
    ratio = observed / ceiling
    if ratio >= AT_CEILING:
        if scales:
            return "at-ceiling", "running at %.0f%% of a ceiling that scales with concurrency — " \
                                 "only a faster source will help" % (100 * ratio)
        return "device-capped", "running at %.0f%% of a ceiling that does NOT scale with " \
                                "concurrency — the device/link is the wall, change the transport " \
                                "not the parallelism" % (100 * ratio)
    if scales:
        return "underused-link", "only %.0f%% of the measured ceiling, which parallel " \
                                 "readers/streams do reach — add concurrency (prefetch, " \
                                 "runai_streamer, more streams)" % (100 * ratio)
    return "underused-link", "only %.0f%% of the measured ceiling — something other than the " \
                             "device is throttling this stage" % (100 * ratio)


def compare_to_profile(records, profile, min_dur_s=1.0, min_mbs=1.0):
    """Per-stage observed rate vs the measured ceilings."""
    cl = ceilings(records)
    rows = []
    for s in profile.get("spans") or []:
        if s.get("dur_s", 0) < min_dur_s or not s.get("sys"):
            continue
        net = (s["sys"].get("net_rx_mbs") or {}).get("avg") or 0.0
        disk = (s["sys"].get("disk_read_mbs") or {}).get("avg") or 0.0
        if max(net, disk) < min_mbs:
            continue
        if net >= disk:
            sub, observed, ceil_v, scales = "network", net, cl["net_best_mbs"], cl["net_scales"]
        else:
            sub, observed, ceil_v, scales = "disk", disk, cl["disk_read_best_mbs"], cl["disk_scales"]
        v, why = verdict_for(observed, ceil_v, scales)
        rows.append({"stage": s["name"], "level": s["level"], "subsystem": sub,
                     "observed_mbs": round(observed, 1),
                     "ceiling_mbs": ceil_v,
                     "headroom_x": round(ceil_v / observed, 2) if (ceil_v and observed) else None,
                     "verdict": v, "why": why, "dur_s": s["dur_s"]})
    return cl, rows


# --------------------------------------------------------------------------- tables
def render_table(records):
    L = ["", "=" * 78, " RESOURCE CEILINGS", "=" * 78]
    env = None
    for r in records:
        if r["probe"] == "env":
            env = r
    if env:
        L.append(" host %s   iface %s%s   %s cores   target %s%s" % (
            env.get("hostname"), env.get("iface"),
            " @%s Mbit" % env["link_speed_mbit"] if env.get("link_speed_mbit") else "",
            env.get("nproc"), env.get("target"),
            "  (%.0f GB free)" % env["target_free_gb"] if env.get("target_free_gb") else ""))
        if not env.get("can_drop_caches"):
            L.append(" note: page cache could not be dropped (container / no root) — direct-IO "
                     "numbers are still valid")
    L.append("")
    L.append(" %-28s %12s %10s %10s" % ("PROBE", "MB/s", "GB", "SECONDS"))
    L.append(" " + "-" * 28 + " " + "-" * 12 + " " + "-" * 10 + " " + "-" * 10)
    for r in records:
        if r["probe"] in ("env", "done") or r["probe"].endswith("_skipped"):
            continue
        if r.get("mbs") is None:
            continue
        L.append(" %-28s %12.1f %10s %10s" % (
            r["probe"], r["mbs"],
            "%.2f" % r["gb"] if r.get("gb") is not None else
            ("%.2f" % (r["read_mb"] / 1000.0) if r.get("read_mb") else
             "%.2f" % (r["size_mb"] / 1000.0) if r.get("size_mb") else "-"),
            "%.1f" % r["s"] if r.get("s") is not None else "-"))
    for r in records:
        if r["probe"].endswith("_skipped"):
            L.append(" %-28s SKIPPED: %s" % (r["probe"].replace("_skipped", ""), r.get("reason")))
    cl = ceilings(records)
    summary = []
    if cl["disk_read_best_mbs"]:
        summary.append("disk read %s MB/s%s" % (
            cl["disk_read_best_mbs"],
            " (1 job %s, %s with parallel readers)" % (
                cl["disk_read_1job_mbs"],
                "scales" if cl["disk_scales"] else "does NOT scale") if cl["disk_read_1job_mbs"] else ""))
    if cl["net_best_mbs"]:
        summary.append("net %s MB/s%s" % (
            cl["net_best_mbs"],
            " (single stream %s, %s with parallel streams)" % (
                cl["net_single_mbs"],
                "scales" if cl["net_scales"] else "does NOT scale") if cl["net_single_mbs"] else ""))
    if cl["s3_best_mbs"]:
        summary.append("object store %s MB/s" % cl["s3_best_mbs"])
    if summary:
        L.append("")
        L.append(" CEILINGS: %s" % "   ".join(summary))
    return L


def render_comparison(cl, rows):
    if not rows:
        return []
    L = ["", " OBSERVED PER-STAGE RATES vs THESE CEILINGS", ""]
    L.append(" %-24s %-9s %11s %11s %9s  %s" % (
        "STAGE", "SUBSYSTEM", "OBSERVED", "CEILING", "HEADROOM", "VERDICT"))
    L.append(" " + "-" * 24 + " " + "-" * 9 + " " + "-" * 11 + " " + "-" * 11 + " " + "-" * 9
             + "  " + "-" * 16)
    for r in sorted(rows, key=lambda r: -r["observed_mbs"]):
        L.append(" %-24s %-9s %8.1f MB %8.1f MB %8s  %s" % (
            r["stage"][:24], r["subsystem"], r["observed_mbs"], r["ceiling_mbs"] or 0.0,
            ("%.1fx" % r["headroom_x"]) if r["headroom_x"] else "-", r["verdict"]))
    L.append("")
    for r in sorted(rows, key=lambda r: -r["observed_mbs"]):
        L.append(_wrap_probe("%s (%s, %.0fs): %s" % (r["stage"], r["subsystem"], r["dur_s"], r["why"])))
    return L


def _wrap_probe(text, width=78, first=" * ", cont="   "):
    from .report import _wrap
    return _wrap(text, width, first, cont)


# --------------------------------------------------------------------------- entry point
def cmd_probe(args):
    em = Emitter(quiet=args.no_json)
    probe_env(em, args.target)
    if not args.no_disk:
        probe_disk(em, args.target, args.size_mb, _levels(args.jobs))
    if not args.no_net:
        urls = args.url or DEFAULT_URLS
        probe_net(em, urls, _levels(args.net_threads), args.net_seconds)
    if args.s3_env and args.s3_prefix:
        probe_s3(em, args.s3_env, args.s3_prefix, _levels(args.s3_workers), args.s3_cap_gb)
    elif args.s3_env or args.s3_prefix:
        em.emit("s3_skipped", reason="both --s3-env and --s3-prefix are required")
    em.emit("done")

    text_lines = render_table(em.records)
    comparison = None
    if args.profile:
        with open(args.profile) as fh:
            profile = json.load(fh)
        cl, rows = compare_to_profile(em.records, profile)
        comparison = {"ceilings": cl, "stages": rows}
        text_lines += render_comparison(cl, rows)
    if not args.no_table:
        print("\n".join(text_lines))

    doc = {"schema": "spinup-profiler/probe/v1", "generated_at": round(time.time(), 3),
           "probes": em.records, "ceilings": ceilings(em.records), "comparison": comparison}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=1)
            fh.write("\n")
        note("wrote %s" % args.out)
    return 0


def _levels(spec):
    """"1,4,8" -> [1, 4, 8]"""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if part.isdigit() and int(part) > 0:
            out.append(int(part))
    return out or [1]
