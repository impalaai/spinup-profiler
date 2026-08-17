"""`sleepwake` — measure vLLM sleep-mode warm start against an already-running server.

The server must have been started with `VLLM_SERVER_DEV_MODE=1` (exposes /sleep,
/wake_up, /is_sleeping, /collective_rpc) **and** `--enable-sleep-mode`.

Level semantics (vLLM v1):
  level 1  weights are offloaded to host RAM, KV cache is discarded -> wake_up restores both.
  level 2  weights are DISCARDED -> the wake path must re-materialise them. Older builds need
           an explicit `reload_weights` collective RPC; newer ones do it inside /wake_up. This
           command tries the RPC and reports the outcome instead of failing.

Field findings (1xH100, vllm-openai:v0.20.2, 2026-08-17):
  * A server started WITHOUT `--enable-sleep-mode` still answers /sleep with 200 in ~19 ms and
    never sleeps. Timings from such a run are meaningless, so this command now verifies
    /is_sleeping after the sleep call and fails instead of reporting a no-op (--no-verify-sleep
    to override).
  * Level 2 is NOT validated on v0.20.2: `collective_rpc reload_weights` returns 500, and the
    first completion after a level-2 sleep succeeds instantly, i.e. the weights do not appear to
    have been discarded. Use level 1 for numbers you intend to quote.
"""

import json
import time
from urllib.parse import urlparse

from . import httpc
from .shell import gpu_mem_used_mib, have_nvidia_smi

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0")
MIN_MEM_DROP_PCT = 30.0  # a real sleep releases most of the GPU; less than this is suspicious


def _hdrs(args):
    return {"Authorization": "Bearer %s" % args.api_key} if args.api_key else {}


def _call(steps, name, url, method="POST", body=None, timeout=600, headers=None):
    t0 = time.time()
    status, raw, err = httpc.request(url, method=method, body=body, timeout=timeout,
                                     headers=headers)
    dur = round(time.time() - t0, 3)
    step = {"step": name, "url": url, "method": method, "s": dur, "status": status,
            "ok": httpc.ok(status)}
    if err:
        step["error"] = err
    elif not httpc.ok(status):
        step["error"] = "HTTP %s %s" % (status, httpc.snippet(raw))
    else:
        body_doc = httpc.json_body(raw)
        if body_doc is not None and not isinstance(body_doc, list):
            step["response"] = body_doc if len(str(body_doc)) < 300 else str(body_doc)[:300]
    steps.append(step)
    return step


def _completion(args, steps, name, timeout=None, retry_until=None):
    """Timed chat completion. With retry_until, polls until it succeeds or the deadline passes."""
    url = "%s/v1/chat/completions" % args.base_url.rstrip("/")
    body = {"model": args.model, "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens, "temperature": 0}
    t0 = time.time()
    while True:
        last = _call(steps, name, url, body=body, timeout=timeout or args.timeout,
                     headers=_hdrs(args))
        if last["ok"] or retry_until is None or time.time() > retry_until:
            break
        steps.pop()  # keep only the decisive attempt
        time.sleep(0.25)
    last["s"] = round(time.time() - t0, 3)  # wall time incl. retries
    doc = last.pop("response", None)  # a whole completion is too noisy for the record
    if isinstance(doc, dict):
        try:
            last["text"] = (doc["choices"][0]["message"].get("content") or "")[:120]
        except Exception:
            pass
        last["usage"] = doc.get("usage")
    return last


def _is_sleeping(step):
    """True / False / None(unknown) from an /is_sleeping step."""
    if not step.get("ok"):
        return None
    body = step.get("response")
    if isinstance(body, bool):
        return body
    if isinstance(body, dict):
        for key in ("is_sleeping", "sleeping", "asleep"):
            if isinstance(body.get(key), bool):
                return body[key]
    return None


def _resolve_gpu_check(args, base, notes):
    """--verify-gpu-mem defaults ON when nvidia-smi exists AND the server is on this box."""
    local = (urlparse(base).hostname or "") in LOCAL_HOSTS
    if args.verify_gpu_mem is False:
        return False
    if args.verify_gpu_mem is True:
        if not have_nvidia_smi():
            notes.append("--verify-gpu-mem requested but nvidia-smi is not usable here; skipping")
            return False
        if not local:
            notes.append("--verify-gpu-mem forced on for a non-local --base-url (%s): the sampled "
                         "GPU is THIS box, not necessarily the server's" % base)
        return True
    if not have_nvidia_smi():
        return False
    if not local:
        notes.append("GPU-memory check auto-disabled: --base-url is not local (%s), so local "
                     "nvidia-smi would measure the wrong box (force with --verify-gpu-mem)" % base)
        return False
    return True


def _emit(out, args, rc):
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    lines = ["", "sleep(level %d) %s s   wake %s s   first completion %s s   warm total %s s" % (
        args.level, out.get("sleep_s"), out.get("wake_s"),
        out.get("first_completion_s"), out.get("warm_total_s"))]
    mem = out.get("gpu_mem")
    if mem and mem.get("asleep_mib") is not None:
        lines.append("GPU mem  %s MiB awake -> %s MiB asleep -> %s MiB after wake  (dropped %s%%)"
                     % (mem.get("before_sleep_mib"), mem.get("asleep_mib"),
                        mem.get("after_wake_mib"), mem.get("drop_pct")))
    if out.get("error"):
        lines.append("FAILED: %s" % out["error"])
    for key in ("gpu_mem_note", "level2_note", "reload_weights_note",
                "reset_prefix_cache_note", "first_completion_error"):
        if out.get(key):
            lines.append("NOTE %s: %s" % (key, out[key]))
    for n in out.get("notes") or []:
        lines.append("NOTE %s" % n)
    print("\n".join(lines))
    return rc


def cmd_sleepwake(args):
    base = args.base_url.rstrip("/")
    steps, notes = [], []
    out = {"schema": "spinup-profiler/sleepwake/v1", "base_url": base, "level": args.level,
           "model": args.model, "generated_at": round(time.time(), 3),
           "verify_sleep": bool(args.verify_sleep), "steps": steps, "notes": notes}
    use_gpu = _resolve_gpu_check(args, base, notes)
    mem = {"source": "nvidia-smi (local)" if use_gpu else None,
           "before_sleep_mib": None, "asleep_mib": None, "after_wake_mib": None, "drop_pct": None}
    out["gpu_mem"] = mem if use_gpu else None

    if not args.no_precheck:
        pre = _completion(args, steps, "precheck_completion")
        out["precheck_completion_s"] = pre["s"] if pre["ok"] else None
        if not pre["ok"]:
            out["error"] = ("precheck completion failed (%s) — fix --model/--base-url before "
                            "measuring sleep/wake" % pre.get("error"))
            return _emit(out, args, 1)

    if use_gpu:
        mem["before_sleep_mib"] = gpu_mem_used_mib()

    _call(steps, "is_sleeping_before", "%s/is_sleeping" % base, method="GET", timeout=30,
          headers=_hdrs(args))
    out["is_sleeping_before"] = _is_sleeping(steps[-1])

    sleep_step = _call(steps, "sleep", "%s/sleep?level=%d" % (base, args.level),
                       timeout=args.timeout, headers=_hdrs(args))
    out["sleep_s"] = sleep_step["s"] if sleep_step["ok"] else None
    if not sleep_step["ok"]:
        out["error"] = "POST /sleep?level=%d failed: %s" % (args.level, sleep_step.get("error"))
        return _emit(out, args, 1)

    after = _call(steps, "is_sleeping_after_sleep", "%s/is_sleeping" % base, method="GET",
                  timeout=30, headers=_hdrs(args))
    slept = _is_sleeping(after)
    out["is_sleeping_after_sleep"] = slept
    if args.verify_sleep and slept is not True:
        # Never leave a server asleep because our own verification bailed out.
        _call(steps, "wake_up_recovery", "%s/wake_up" % base, timeout=args.timeout,
              headers=_hdrs(args))
        if slept is False:
            out["error"] = (
                "server did not actually sleep — was it started with --enable-sleep-mode? "
                "(/sleep returned %s in %.3fs but /is_sleeping is still false; a server without "
                "sleep mode answers /sleep 200 and does nothing)"
                % (sleep_step["status"], sleep_step["s"]))
        else:
            out["error"] = (
                "could not verify the sleep took effect (%s) — refusing to report timings that "
                "may be a no-op; pass --no-verify-sleep to override"
                % (after.get("error") or "/is_sleeping returned no is_sleeping field"))
        out["wake_up_recovery_attempted"] = True
        return _emit(out, args, 1)

    if use_gpu:
        mem["asleep_mib"] = gpu_mem_used_mib()
        if (mem["before_sleep_mib"] and mem["asleep_mib"] is not None
                and _drop_pct(mem) < MIN_MEM_DROP_PCT):
            time.sleep(1.5)  # some builds free asynchronously; re-sample once before crying wolf
            mem["asleep_mib"] = gpu_mem_used_mib()
        mem["drop_pct"] = _drop_pct(mem)
        if mem["drop_pct"] is not None and mem["drop_pct"] < MIN_MEM_DROP_PCT:
            out["gpu_mem_note"] = (
                "GPU memory only dropped %.1f%% while asleep (%s -> %s MiB); a real level-%d sleep "
                "releases most of it, so this sleep may be a partial no-op (or another process "
                "holds the memory)" % (mem["drop_pct"], mem["before_sleep_mib"],
                                       mem["asleep_mib"], args.level))
    if args.hold:
        time.sleep(args.hold)

    t_wake0 = time.time()
    wake_step = _call(steps, "wake_up", "%s/wake_up" % base, timeout=args.timeout,
                      headers=_hdrs(args))
    out["wake_s"] = wake_step["s"] if wake_step["ok"] else None
    if not wake_step["ok"]:
        out["error"] = "POST /wake_up failed: %s" % wake_step.get("error")
        return _emit(out, args, 1)

    if args.level == 2:
        # Level 2 dropped the weights. Both of these are version-dependent: report, never fail.
        rpc = _call(steps, "collective_rpc_reload_weights", "%s/collective_rpc" % base,
                    body={"method": "reload_weights"}, timeout=args.timeout, headers=_hdrs(args))
        out["reload_weights_s"] = rpc["s"] if rpc["ok"] else None
        if not rpc["ok"]:
            out["reload_weights_note"] = (
                "collective_rpc reload_weights not available/failed (%s) — known to return 500 on "
                "vllm-openai:v0.20.2; on newer builds /wake_up restores the weights itself"
                % rpc.get("error"))
        rp = _call(steps, "reset_prefix_cache", "%s/reset_prefix_cache" % base,
                   timeout=args.timeout, headers=_hdrs(args))
        out["reset_prefix_cache_s"] = rp["s"] if rp["ok"] else None
        if not rp["ok"]:
            out["reset_prefix_cache_note"] = "reset_prefix_cache failed/absent (%s)" % rp.get("error")

    first = _completion(args, steps, "first_completion_after_wake",
                        retry_until=time.time() + args.first_completion_timeout)
    out["first_completion_s"] = first["s"] if first["ok"] else None
    out["first_completion_ok"] = bool(first["ok"])
    if not first["ok"]:
        out["first_completion_error"] = first.get("error")
    out["warm_total_s"] = round(time.time() - t_wake0, 3)

    if use_gpu:
        mem["after_wake_mib"] = gpu_mem_used_mib()

    if (args.level == 2 and out.get("first_completion_s") is not None
            and out["first_completion_s"] < 1.0 and not out.get("reload_weights_s")):
        out["level2_note"] = (
            "level-2 result is NOT trustworthy on this build: reload_weights did not run and the "
            "first completion came back in %.2fs, i.e. the weights were probably never discarded. "
            "Quote level-1 numbers instead (see README)." % out["first_completion_s"])

    return _emit(out, args, 0 if out.get("first_completion_ok") else 1)


def _drop_pct(mem):
    before, asleep = mem.get("before_sleep_mib"), mem.get("asleep_mib")
    if not before or asleep is None:
        return None
    return round(100.0 * (before - asleep) / before, 1)
