"""`run` — execute one instrumented vLLM spin-up on THIS box and profile it.

Mirrors the field harness (impala-spinup-research/cross-cloud/harness/run_spinup.sh)
plus the host-event JSONL from box_bootstrap.sh, in one stdlib-only command.
"""

import json
import os
import re
import sys
import time

from . import httpc, prefetch, sysmetrics
from .parse import build_profile, write_profile
from .report import render
from .shell import sh

# Injected unless --no-default-env. DEBUG level is REQUIRED for the compile /
# memory-profiling / cudagraph marks; without it those spans cannot be split.
DEFAULT_ENV = [
    "VLLM_CONFIGURE_LOGGING=1",
    "VLLM_LOGGING_LEVEL=DEBUG",
    "NO_COLOR=1",
]


class Events:
    """Append-only JSONL of host + lifecycle events, flushed on every write."""

    def __init__(self, path):
        self.path = path
        self.fh = open(path, "a", buffering=1)

    def emit(self, ev, **kw):
        rec = {"ev": ev, "t": time.time()}
        rec.update(kw)
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        return rec["t"]

    def close(self):
        self.fh.close()


def resolve_docker(explicit):
    """Return the docker argv prefix that actually works here (docker, then sudo docker)."""
    if explicit:
        return explicit.split()
    for prefix in (["docker"], ["sudo", "-n", "docker"]):
        rc, _o, _e = sh(prefix + ["info", "--format", "{{.ServerVersion}}"], timeout=30)
        if rc == 0:
            return prefix
    return ["docker"]


def log(msg):
    sys.stderr.write("[spinup-profile] %s\n" % msg)
    sys.stderr.flush()


def host_preflight(ev, docker, image, do_pull):
    """GPU / docker / image-pull events, mirroring box_bootstrap.sh."""
    info = {}
    rc, out, _e = sh(["nvidia-smi", "-L"], timeout=60)
    if rc == 0 and out.strip():
        gpus = [l for l in out.strip().splitlines() if l.strip()]
        info["gpu"] = gpus[0].replace('"', "")
        info["gpu_count"] = len(gpus)
        ev.emit("gpu_ready", gpu=info["gpu"], count=len(gpus))
    else:
        ev.emit("gpu_missing")
        log("WARNING: nvidia-smi found no GPU — continuing anyway")

    rc, out, _e = sh(docker + ["info", "--format", "{{.ServerVersion}}"], timeout=60)
    if rc == 0:
        info["docker_version"] = out.strip()
        ev.emit("docker_ready", version=out.strip())
    else:
        ev.emit("docker_missing")
        raise SystemExit("docker is not usable via %r — install it or pass --docker-bin"
                         % " ".join(docker))
    rc, out, _e = sh(docker + ["info", "--format", "{{json .Runtimes}}"], timeout=60)
    ev.emit("runtime_ready", nvidia="nvidia" in out)

    rc, _o, _e = sh(docker + ["image", "inspect", image], timeout=120)
    if rc == 0:
        ev.emit("image_already_present", image=image)
        info["image_pulled"] = False
    elif do_pull:
        log("pulling %s (this is usually 140-160s for the vLLM image)" % image)
        t0 = ev.emit("pull_start", image=image)
        rc, _o, err = sh(docker + ["pull", image], timeout=7200)
        if rc == 0:
            t1 = ev.emit("pull_done", image=image)
            info["image_pull_s"] = round(t1 - t0, 2)
            info["image_pulled"] = True
            log("pull done in %.1fs" % info["image_pull_s"])
        else:
            ev.emit("pull_failed", image=image, error=err.strip()[:400])
            raise SystemExit("docker pull %s failed: %s" % (image, err.strip()[:400]))
    else:
        ev.emit("image_absent_pull_skipped", image=image)
        log("WARNING: image absent and --no-pull given — docker run will pull implicitly, "
            "which lands inside the launch->container span")
    return info


def drop_caches(ev):
    rc, _o, err = sh(["sh", "-c", "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null"],
                     timeout=120)
    if rc == 0:
        ev.emit("drop_caches_done")
    else:
        ev.emit("drop_caches_failed", error=err.strip()[:200])
        log("WARNING: could not drop page cache (need passwordless sudo) — the run is not fully cold")


def container_started_at(docker, name):
    """Epoch of .State.StartedAt, the same anchor the research harness used."""
    rc, out, _e = sh(docker + ["inspect", name], timeout=60)
    if rc != 0:
        return None, None
    try:
        doc = json.loads(out)[0]
    except Exception:
        return None, out
    started = doc.get("State", {}).get("StartedAt")
    if not started:
        return None, out
    from .patterns import ts as parse_ts
    try:
        return parse_ts(re.sub(r"(\.\d{6})\d*", r"\1", started)), out
    except Exception:
        return None, out


def is_running(docker, name):
    rc, out, _e = sh(docker + ["inspect", "-f", "{{.State.Running}}", name], timeout=30)
    return rc == 0 and out.strip() == "true"


def build_docker_argv(args, docker):
    argv = docker + ["run", "-d", "--name", args.container]
    if args.gpus:
        argv += ["--gpus", args.gpus]
    argv += ["--network", args.network] if args.network else []
    if args.shm_size:
        argv += ["--shm-size", args.shm_size]
    if args.network != "host":
        argv += ["-p", "%d:%d" % (args.port, args.port)]
    env = [] if args.no_default_env else list(DEFAULT_ENV)
    env.append("VLLM_ENGINE_READY_TIMEOUT_S=%d" % args.timeout)
    env += list(args.env or [])
    for e in env:
        argv += ["-e", e]
    for v in args.volume or []:
        argv += ["-v", v]
    for extra in args.docker_arg or []:
        argv += extra.split(" ") if " " in extra else [extra]
    argv += [args.image] + list(args.vllm_args)
    return argv


def poll_health(args, deadline, docker, ev):
    """Poll /health at 5 Hz until 200, container death, or deadline."""
    url = "%s/health" % args.base_url
    headers = {"Authorization": "Bearer %s" % args.api_key} if args.api_key else None
    last_note = time.time()
    i = 0
    while True:
        now = time.time()
        if now > deadline:
            ev.emit("health_timeout", waited_s=round(now - (deadline - args.timeout), 1))
            return None, "timeout after %ss waiting for %s" % (args.timeout, url)
        status, _b, _err = httpc.request(url, timeout=1, headers=headers)
        if httpc.ok(status):
            return ev.emit("health_200"), None
        i += 1
        if i % 10 == 0 and not is_running(docker, args.container):
            ev.emit("container_died")
            return None, "container %s exited before becoming ready" % args.container
        if now - last_note >= 30:
            last_note = now
            rc, out, _e = sh(docker + ["logs", "--tail", "1", args.container], timeout=20)
            tail = (out.strip().splitlines() or [""])[-1][:110] if rc == 0 else ""
            log("waiting for /health ... %.0fs elapsed | %s" % (now - (deadline - args.timeout), tail))
        time.sleep(0.2)


def discover_model(args):
    status, raw, _err = httpc.request("%s/v1/models" % args.base_url, timeout=10,
                                     headers={"Authorization": "Bearer %s" % args.api_key} if args.api_key else None)
    doc = httpc.json_body(raw) if httpc.ok(status) else None
    try:
        return doc["data"][0]["id"]
    except Exception:
        return None


def sanity_completion(args, ev, model):
    """First successful chat completion after ready — the only proof the box can serve."""
    url = "%s/v1/chat/completions" % args.base_url
    headers = {"Authorization": "Bearer %s" % args.api_key} if args.api_key else {}
    body = {"model": model, "messages": [{"role": "user", "content": args.sanity_prompt}],
            "max_tokens": args.sanity_max_tokens, "temperature": 0}
    t_start = ev.emit("sanity_start", model=model)
    deadline = t_start + args.sanity_timeout
    last_err = None
    while time.time() < deadline:
        status, raw, err = httpc.request(url, method="POST", body=body,
                                         timeout=max(10, args.sanity_timeout), headers=headers)
        if httpc.ok(status):
            doc = httpc.json_body(raw) or {}
            text = ""
            try:
                text = (doc["choices"][0]["message"].get("content") or "")[:200]
            except Exception:
                pass
            t_ok = ev.emit("sanity_ok", model=model, text=text)
            return {"ok": True, "t_start": t_start, "t_ok": t_ok, "text": text,
                    "usage": doc.get("usage")}, None
        last_err = err or ("HTTP %s %s" % (status, httpc.snippet(raw)))
        time.sleep(1.0)
    ev.emit("sanity_failed", model=model, error=str(last_err)[:400])
    return {"ok": False, "t_start": t_start, "t_ok": None, "error": str(last_err)[:400]}, last_err


def cmd_run(args):
    if not args.vllm_args:
        raise SystemExit("no vLLM args given — put them after `--`, e.g. "
                         "`run --image ... -- --model Qwen/Qwen3.5-27B-FP8 --port 8200`")
    args.base_url = args.base_url or "http://%s:%d" % (args.health_host, args.port)
    out_dir = os.path.join(os.path.expanduser(args.out), args.label, "run%s" % args.run_index)
    os.makedirs(out_dir, exist_ok=True)
    docker = resolve_docker(args.docker_bin)
    ev = Events(os.path.join(out_dir, "events.jsonl"))
    log("run dir %s" % out_dir)
    ev.emit("run_start", image=args.image, port=args.port, label=args.label,
            run=args.run_index, vllm_args=" ".join(args.vllm_args),
            docker=" ".join(docker), host=os.uname().nodename)

    # System sampler. Started BEFORE the preflight pull (a superset of launch->sanity) so the
    # image_pull span gets real network numbers too; stopped once the sanity completion is done.
    sampler = None
    want_sys = args.sample_system if args.sample_system is not None else sysmetrics.available()
    if want_sys and not sysmetrics.available():
        log("WARNING: --sample-system requested but /proc/net/dev is not readable — skipping")
        want_sys = False
    if want_sys:
        sampler = sysmetrics.Sampler(os.path.join(out_dir, "system.jsonl"),
                                     interval=args.sample_interval).start()
        if sampler.stopped_early:
            log("WARNING: system sampler did not start (%s)" % sampler.stopped_early)
        else:
            ev.emit("sysmetrics_start", interval_s=args.sample_interval)

    host_info = host_preflight(ev, docker, args.image, not args.no_pull)

    rc, out, _e = sh(docker + ["rm", "-f", args.container], timeout=300)
    if rc == 0 and out.strip():
        ev.emit("removed_stale_container", name=args.container)
    if args.drop_caches:
        drop_caches(ev)

    # Page-cache warm-up: after the cache was dropped (otherwise it is pointless) and before the
    # container exists, so the engine's weight read hits RAM instead of the device.
    warmed = None
    if args.prefetch:
        log("prefetching %s with %d jobs" % (args.prefetch, args.prefetch_jobs))
        ev.emit("prefetch_start", spec=args.prefetch, jobs=args.prefetch_jobs)
        warmed = prefetch.warm(args.prefetch, jobs=args.prefetch_jobs)
        ev.emit("prefetch_done", **{k: v for k, v in warmed.items() if k != "spec"})
        if warmed["files"] == 0:
            log("WARNING: --prefetch %s matched no files" % args.prefetch)
        else:
            log("prefetched %.2f GB from %d file(s) in %.2fs (%s MB/s)"
                % (warmed["gb"], warmed["files"], warmed["s"], warmed["mbs"]))
        if warmed.get("errors"):
            log("WARNING: prefetch read errors: %s" % "; ".join(warmed["errors"]))

    argv = build_docker_argv(args, docker)
    log("docker run: %s" % " ".join(argv[len(docker):]))
    t_launch = ev.emit("docker_run_issued", argv=" ".join(argv))
    rc, out, err = sh(argv, timeout=args.timeout + 300)
    if rc != 0:
        ev.emit("docker_run_failed", rc=rc, error=err.strip()[:600])
        raise SystemExit("docker run failed (rc=%d): %s" % (rc, err.strip()[:600]))
    cid = out.strip()[:12]
    t_start, inspect_raw = container_started_at(docker, args.container)
    ev.emit("container_started", cid=cid, t_container_start=t_start)

    t_ready, ready_err = poll_health(args, t_launch + args.timeout, docker, ev)
    if ready_err:
        log("NOT READY: %s" % ready_err)

    sanity = None
    if t_ready and not args.no_sanity:
        model = args.sanity_model or discover_model(args)
        if not model:
            log("WARNING: no --sanity-model and /v1/models gave nothing — skipping sanity")
        else:
            sanity, serr = sanity_completion(args, ev, model)
            if serr:
                log("SANITY FAILED: %s" % serr)
            else:
                log("sanity completion ok in %.2fs" % (sanity["t_ok"] - sanity["t_start"]))

    if sampler is not None:
        sampler.stop()
        ev.emit("sysmetrics_stop", ticks=sampler.ticks, stopped_early=sampler.stopped_early)
        if sampler.stopped_early:
            log("WARNING: system sampler stopped early (%s)" % sampler.stopped_early)

    # capture artefacts
    log_path = os.path.join(out_dir, "docker.log")
    rc, out, err = sh(docker + ["logs", "-t", args.container], timeout=600)
    with open(log_path, "w") as fh:
        fh.write(out)
        if err:
            fh.write(err)
    ev.emit("logs_captured", bytes=os.path.getsize(log_path))
    if inspect_raw is None:
        _t, inspect_raw = container_started_at(docker, args.container)
    if inspect_raw:
        with open(os.path.join(out_dir, "inspect.json"), "w") as fh:
            fh.write(inspect_raw)

    meta = {
        "name": args.label,
        "run": args.run_index,
        "image": args.image,
        "t_launch": t_launch,
        "t_container_start": t_start or t_launch,
        "t_container_start_source": "docker inspect" if t_start else "t_launch (inspect failed)",
        "t_ready": t_ready,
        "total_launch_to_ready": (t_ready - t_launch) if t_ready else None,
        "total_start_to_ready": (t_ready - (t_start or t_launch)) if t_ready else None,
        "port": args.port,
        "vllm_args": " ".join(args.vllm_args),
        "container": args.container,
        "cid": cid,
        "docker_argv": argv,
        "drop_caches": bool(args.drop_caches),
        "timeout": args.timeout,
        "host": os.uname().nodename,
        "cpu_count": os.cpu_count(),
        "ready_error": ready_err,
        "profiler": "spinup-profiler run",
        "sysmetrics_ticks": sampler.ticks if sampler else 0,
        "prefetch": warmed,
    }
    meta.update({k: v for k, v in host_info.items()})
    if sanity:
        meta["sanity_model"] = args.sanity_model or "(discovered)"
        meta["sanity_ok"] = sanity["ok"]
        meta["t_sanity_start"] = sanity["t_start"]
        meta["t_sanity_ok"] = sanity["t_ok"]
        meta["sanity_text"] = sanity.get("text")
        meta["sanity_error"] = sanity.get("error")
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")

    if args.teardown:
        sh(docker + ["rm", "-f", args.container], timeout=300)
        ev.emit("container_removed", name=args.container)
    else:
        log("container %s left RUNNING (use --teardown to remove it, or `%s rm -f %s`)"
            % (args.container, " ".join(docker), args.container))
    ev.emit("run_done", ready=bool(t_ready))
    ev.close()

    profile = build_profile(out_dir)
    write_profile(profile, os.path.join(out_dir, "profile.json"))
    text = render(profile, width=args.width)
    with open(os.path.join(out_dir, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    artefacts = "events.jsonl,docker.log,meta.json,profile.json,report.txt"
    if sampler is not None and sampler.ticks:
        artefacts += ",system.jsonl"
    print("\nwrote %s/{%s}" % (out_dir, artefacts))
    if ready_err:
        return 1
    if sanity and not sanity["ok"]:
        return 1
    return 0
