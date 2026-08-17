"""`sleepwake` — measure vLLM sleep-mode warm start against an already-running server.

The server must have been started with `VLLM_SERVER_DEV_MODE=1` (exposes /sleep,
/wake_up, /is_sleeping, /collective_rpc) and `--enable-sleep-mode`.

Level semantics (vLLM v1):
  level 1  weights are offloaded to host RAM, KV cache is discarded -> wake_up restores both.
  level 2  weights are DISCARDED -> the wake path must re-materialise them. Older builds need
           an explicit `reload_weights` collective RPC; newer ones do it inside /wake_up. This
           command tries the RPC and reports the outcome instead of failing.
"""

import json
import time

from . import httpc


def _hdrs(args):
    return {"Authorization": "Bearer %s" % args.api_key} if args.api_key else {}


def _call(steps, name, url, method="POST", body=None, timeout=600, headers=None, fatal=False):
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
    if fatal and not step["ok"]:
        raise SystemExit("%s failed: %s" % (name, step.get("error")))
    return step


def _completion(args, steps, name, timeout=None, retry_until=None):
    """Timed chat completion. With retry_until, polls until it succeeds or the deadline passes."""
    url = "%s/v1/chat/completions" % args.base_url.rstrip("/")
    body = {"model": args.model, "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens, "temperature": 0}
    t0 = time.time()
    last = None
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


def cmd_sleepwake(args):
    base = args.base_url.rstrip("/")
    steps = []
    out = {"schema": "spinup-profiler/sleepwake/v1", "base_url": base, "level": args.level,
           "model": args.model, "generated_at": round(time.time(), 3)}

    if not args.no_precheck:
        pre = _completion(args, steps, "precheck_completion")
        out["precheck_completion_s"] = pre["s"] if pre["ok"] else None
        if not pre["ok"]:
            raise SystemExit("precheck completion failed (%s) — fix --model/--base-url before "
                             "measuring sleep/wake" % pre.get("error"))

    _call(steps, "is_sleeping_before", "%s/is_sleeping" % base, method="GET", timeout=30,
          headers=_hdrs(args))

    sleep_step = _call(steps, "sleep", "%s/sleep?level=%d" % (base, args.level),
                       timeout=args.timeout, headers=_hdrs(args), fatal=True)
    out["sleep_s"] = sleep_step["s"]
    _call(steps, "is_sleeping_after_sleep", "%s/is_sleeping" % base, method="GET", timeout=30,
          headers=_hdrs(args))
    if args.hold:
        time.sleep(args.hold)

    t_wake0 = time.time()
    wake_step = _call(steps, "wake_up", "%s/wake_up" % base, timeout=args.timeout,
                      headers=_hdrs(args), fatal=True)
    out["wake_s"] = wake_step["s"]

    if args.level == 2:
        # Level 2 dropped the weights. Both of these are version-dependent: report, never fail.
        rpc = _call(steps, "collective_rpc_reload_weights", "%s/collective_rpc" % base,
                    body={"method": "reload_weights"}, timeout=args.timeout, headers=_hdrs(args))
        out["reload_weights_s"] = rpc["s"] if rpc["ok"] else None
        if not rpc["ok"]:
            out["reload_weights_note"] = (
                "collective_rpc reload_weights not available/failed (%s) — on this vLLM build "
                "/wake_up likely restores weights itself" % rpc.get("error"))
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
    out["steps"] = steps

    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    lines = ["", "sleep(level %d) %s s   wake %s s   first completion %s s   warm total %s s" % (
        args.level, out.get("sleep_s"), out.get("wake_s"),
        out.get("first_completion_s"), out.get("warm_total_s"))]
    for k in ("reload_weights_note", "reset_prefix_cache_note", "first_completion_error"):
        if out.get(k):
            lines.append("NOTE %s: %s" % (k, out[k]))
    print("\n".join(lines))
    return 0 if out.get("first_completion_ok") else 1
