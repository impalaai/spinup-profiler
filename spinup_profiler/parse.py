"""Turn a captured spin-up (docker.log [+ meta / host events / provision]) into profile.json."""

import json
import os
import time

from . import stages, sysmetrics
from .patterns import MULTI_PATTERNS, PATTERNS, TS_RE, ts

SCHEMA = "spinup-profiler/profile/v1"


# --------------------------------------------------------------------------- inputs
def read_log(log_path):
    """Scan the log once. Returns (first_ts, marks_abs, values, multi, stats)."""
    first_ts = None
    marks, values = {}, {}
    multi = {name: [] for name, _rx in MULTI_PATTERNS}
    total_lines = stamped_lines = 0

    with open(log_path, errors="replace") as fh:
        for line in fh:
            total_lines += 1
            m = TS_RE.match(line)
            if not m:
                continue
            stamped_lines += 1
            t, text = ts(m.group(1)), m.group(2)
            if first_ts is None:
                first_ts = t
            for name, rx, keep_first in PATTERNS:
                mm = rx.search(text)
                if not mm:
                    continue
                if keep_first:
                    marks.setdefault(name, t)
                else:
                    marks[name] = t
                groups = [g for g in mm.groups() if g]
                if not groups:
                    continue
                if name == "compile_cache_dir":
                    values["compile_cache_dir"] = groups[-1]
                    continue
                try:
                    values[name + "_claimed"] = float(groups[-1].replace(",", ""))
                except ValueError:
                    continue
                if name == "weights_done":
                    values["weights_gib"] = float(groups[0])
                if name == "kv_size":
                    values["kv_tokens"] = int(groups[0].replace(",", ""))
                    del values["kv_size_claimed"]
                if name == "kv_mem":  # GiB, not seconds — keep it out of `claimed`
                    values["kv_mem_gib"] = values.pop("kv_mem_claimed")
            for name, rx in MULTI_PATTERNS:
                if rx.search(text):
                    multi[name].append(t)

    stats = {"log_lines": total_lines, "timestamped_lines": stamped_lines}
    return first_ts, marks, values, multi, stats


def read_host_events(path):
    """box_bootstrap.sh-style JSONL: {"ev": "...", "t": <epoch>, ...}."""
    events = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            name = d.get("ev") or d.get("event")
            if name is None or "t" not in d:
                continue
            try:
                d["t"] = float(d["t"])
            except (TypeError, ValueError):
                continue
            d["ev"] = name
            events.append(d)
    return events


PROVISION_START_KEYS = ("t_request", "t_requested", "t_provision_start", "t_create", "t_start")
PROVISION_SSH_KEYS = ("t_ssh", "t_ssh_ready", "t_ssh_up", "t_ssh_ok")


def read_provision(path):
    with open(path, errors="replace") as fh:
        d = json.load(fh)
    out = {"raw": d}
    for k in PROVISION_START_KEYS:
        if isinstance(d.get(k), (int, float)):
            out["t_request"] = float(d[k])
            break
    for k in PROVISION_SSH_KEYS:
        if isinstance(d.get(k), (int, float)):
            out["t_ssh"] = float(d[k])
            break
    for k in ("provider", "cloud", "region", "instance_type", "gpu", "notes"):
        if k in d:
            out[k] = d[k]
    return out


def resolve_inputs(target, meta=None, host_events=None, provision=None, system=None, probe=None):
    """A run dir picks up its siblings; a bare log file uses only what is passed."""
    paths = {"log": None, "meta": meta, "host_events": host_events, "provision": provision,
             "system": system, "probe": probe}
    if os.path.isdir(target):
        paths["log"] = os.path.join(target, "docker.log")
        for key, fname in (("meta", "meta.json"),
                           ("host_events", "events.jsonl"),
                           ("provision", "provision.json"),
                           ("system", "system.jsonl"),
                           ("probe", "probe.json")):
            if paths[key] is None:
                cand = os.path.join(target, fname)
                if os.path.exists(cand):
                    paths[key] = cand
        paths["out_dir"] = target
    else:
        paths["log"] = target
        paths["out_dir"] = os.path.dirname(os.path.abspath(target))
    return paths


# --------------------------------------------------------------------------- spans
def _span(name, start_s, end_s, level, parent=None, source="log", t_start=None, t_end=None):
    return {
        "name": name,
        "level": level,
        "parent": parent,
        "start_s": round(start_s, 2),
        "end_s": round(end_s, 2),
        "dur_s": round(end_s - start_s, 2),
        # absolute epochs: host spans carry their measured ones, container spans get them from t0
        "t_start": round(t_start, 3) if t_start else None,
        "t_end": round(t_end, 3) if t_end else None,
        "source": source,
        "desc": stages.describe(name),
    }


def _first(marks, names):
    for n in names:
        if marks.get(n) is not None:
            return marks[n]
    return None


def _gaps(start, end, spans, tol=0.05):
    """Intervals in [start, end] that `spans` (non-overlapping) do not cover.

    A gap means a mark was missing — usually because the log level was above
    DEBUG. Making it an explicit span keeps the waterfall summing to the total
    instead of silently shrinking it.
    """
    out, cursor = [], start
    for s in sorted(spans, key=lambda s: s["start_s"]):
        if s["start_s"] - cursor > tol:
            out.append((cursor, s["start_s"]))
        cursor = max(cursor, s["end_s"])
    if end - cursor > tol:
        out.append((cursor, end))
    return out


def _fill_gaps(spans, start, end, level, parent, warnings, label):
    """Append an `unattributed_*` span per hole, named after the span it follows."""
    filled = []
    for a, b in _gaps(start, end, spans):
        before = [s for s in spans if s["end_s"] <= a + 0.05]
        after = [s for s in spans if s["start_s"] >= b - 0.05]
        if before:
            name = "unattributed_after_" + max(before, key=lambda s: s["end_s"])["name"]
        elif after:
            name = "unattributed_before_" + min(after, key=lambda s: s["start_s"])["name"]
        else:
            name = "unattributed"
        s = _span(name, a, b, level, parent=parent, source="gap")
        s["axis"] = "container"
        filled.append(s)
        warnings.append("%.2fs of %s is unattributed (%s) — raise the log level to "
                        "VLLM_LOGGING_LEVEL=DEBUG so the missing marks appear" % (b - a, label, name))
    return filled


def build_profile(target, meta_path=None, host_events_path=None, provision_path=None,
                  system_path=None, probe_path=None):
    paths = resolve_inputs(target, meta_path, host_events_path, provision_path, system_path,
                           probe_path)
    if not os.path.exists(paths["log"]):
        raise SystemExit("no docker log at %s" % paths["log"])

    warnings = []
    first_abs, abs_marks, values, multi, stats = read_log(paths["log"])
    if first_abs is None:
        raise SystemExit(
            "no timestamped lines in %s — capture logs with `docker logs -t`" % paths["log"]
        )
    if stats["timestamped_lines"] < stats["log_lines"]:
        warnings.append("%d of %d log lines had no timestamp prefix and were skipped"
                        % (stats["log_lines"] - stats["timestamped_lines"], stats["log_lines"]))

    meta = {}
    if paths["meta"] and os.path.exists(paths["meta"]):
        with open(paths["meta"], errors="replace") as fh:
            meta = json.load(fh)
    else:
        warnings.append("no meta.json: timeline is anchored on the first log line, not container start")

    host = read_host_events(paths["host_events"]) if paths["host_events"] and os.path.exists(paths["host_events"]) else []
    prov = read_provision(paths["provision"]) if paths["provision"] and os.path.exists(paths["provision"]) else {}

    # t0 = container start (falls back to the first log line)
    t0 = meta.get("t_container_start") or first_abs
    t0_source = "meta.t_container_start" if meta.get("t_container_start") else "first_log_line"

    abs_marks["graphs_start"] = stages.derive_graphs_start(multi["cudagraph_capture"], abs_marks)
    abs_marks["first_log"] = first_abs

    marks = {}
    for name in stages.MARK_ORDER:
        v = abs_marks.get(name)
        marks[name] = round(v - t0, 2) if v is not None else None

    ready_s = marks.get("ready")
    t_ready = meta.get("t_ready")
    total_s = None
    if meta.get("total_start_to_ready"):
        total_s = round(meta["total_start_to_ready"], 2)
    elif t_ready:
        total_s = round(t_ready - t0, 2)
    elif ready_s is not None:
        total_s = ready_s
        warnings.append("no /health timestamp: total_s is the log's startup-complete mark")
    if total_s is None:
        warnings.append("spin-up never reached a ready state — spans are partial")

    spans = []

    # host prefix (absolute epoch inputs, so they are placed before t0 in order)
    def ev_full(name):
        for e in host:
            if e["ev"] == name:
                return e
        return None

    def ev(name):
        e = ev_full(name)
        return e["t"] if e else None

    host_pairs = []
    if prov.get("t_request") is not None:
        end = prov.get("t_ssh") or ev("boot_seen")
        if end is not None:
            host_pairs.append(("provision_to_ssh", prov["t_request"], end))
        else:
            warnings.append("provision.json has no SSH-ready timestamp; provision span skipped")
    host_pairs += [
        ("host_gpu_ready", ev("boot_seen"), ev("gpu_ready")),
        ("host_docker_ready", ev("gpu_ready"), ev("runtime_ready") or ev("docker_ready")),
        ("image_pull", ev("pull_start"), ev("pull_done")),
        ("prefetch", ev("prefetch_start"), ev("prefetch_done")),
        ("launch_to_container", meta.get("t_launch"), meta.get("t_container_start")),
    ]
    cursor = 0.0
    for name, a, b in host_pairs:
        if a is None or b is None or b <= a:
            continue
        dur = b - a
        s = _span(name, cursor, cursor + dur, 0, source="host", t_start=a, t_end=b)
        s["axis"] = "host"  # measured on the box, before the container timeline
        if name == "prefetch":  # carry what was warmed, not just how long it took
            payload = ev_full("prefetch_done") or {}
            for key in ("bytes", "gb", "mbs", "files", "jobs", "segments"):
                if payload.get(key) is not None:
                    s[key] = payload[key]
        spans.append(s)
        cursor += dur
    host_prefix_s = round(cursor, 2)

    # container timeline (level 0)
    marks_for_spans = dict(marks)
    marks_for_spans["t0"] = 0.0
    top_by_name = {}
    for name, frm, to, _desc in stages.TOP_SPANS:
        a, b = marks_for_spans.get(frm), marks_for_spans.get(to)
        if a is None or b is None:
            warnings.append("span %s not derivable (missing mark %s)"
                            % (name, frm if a is None else to))
            continue
        s = _span(name, a + host_prefix_s, b + host_prefix_s, 0)
        s["axis"] = "container"
        spans.append(s)
        top_by_name[name] = s

    container_spans = [s for s in spans if s.get("axis") == "container"]
    if container_spans:
        container_end = (ready_s + host_prefix_s if ready_s is not None
                         else max(s["end_s"] for s in container_spans))
        spans += _fill_gaps(container_spans, host_prefix_s, container_end, 0, None,
                            warnings, "the container timeline")

    # engine_init refinement (level 1)
    leaves = []
    if "engine_init" in top_by_name:
        parent = top_by_name["engine_init"]
        for spec in stages.ENGINE_INIT_LEAVES:
            a, b = _first(marks, spec["frm"]), _first(marks, spec["to"])
            if a is None or b is None or b < a:
                continue
            s = _span(spec["name"], a + host_prefix_s, b + host_prefix_s, 1, parent="engine_init")
            s["axis"] = "container"
            leaves.append(s)
        if leaves:
            leaves += _fill_gaps(leaves, parent["start_s"], parent["end_s"], 1, "engine_init",
                                 warnings, "engine_init")
        spans += leaves

    # tail spans (level 0, after `ready`)
    tail_cursor = None
    if ready_s is not None:
        tail_cursor = ready_s + host_prefix_s
    if tail_cursor is not None and t_ready:
        lag = round((t_ready - t0) - ready_s, 2)
        if lag > 0:
            s = _span("ready_to_health", tail_cursor, tail_cursor + lag, 0)
            s["axis"] = "container"
            spans.append(s)
            tail_cursor += lag
    first_completion_s = None
    if meta.get("t_sanity_ok") and t_ready:
        first_completion_s = round(meta["t_sanity_ok"] - t_ready, 2)
        if tail_cursor is not None and first_completion_s > 0:
            s = _span("first_completion", tail_cursor, tail_cursor + first_completion_s, 0)
            s["axis"] = "container"
            spans.append(s)

    total_profiled_s = round(sum(s["dur_s"] for s in spans if s["level"] == 0), 2)
    for s in spans:
        s["pct"] = round(100.0 * s["dur_s"] / total_profiled_s, 2) if total_profiled_s else None
        if s["t_start"] is None and s.get("axis") == "container":
            s["t_start"] = round(t0 + s["start_s"] - host_prefix_s, 3)
            s["t_end"] = round(t0 + s["end_s"] - host_prefix_s, 3)

    # per-span system metrics (network / disk / cpu / gpu), if a system.jsonl was captured
    sys_summary = None
    if paths["system"] and os.path.exists(paths["system"]):
        try:
            sys_meta, samples = sysmetrics.read_samples(paths["system"])
            if not samples:
                warnings.append("system.jsonl has no usable samples")
            else:
                if not sys_meta.get("cpu_count") and meta.get("cpu_count"):
                    sys_meta["cpu_count"] = meta["cpu_count"]
                series = sysmetrics.build_series(samples)
                for s in spans:
                    st = sysmetrics.window_stats(series, s["t_start"], s["t_end"])
                    if st:
                        s["sys"] = st
                sys_summary = sysmetrics.summarize(series, sys_meta)
                uncovered = [s["name"] for s in spans if s["level"] == 0 and "sys" not in s]
                if uncovered:
                    warnings.append("no system samples cover %s — the sampler was not running "
                                    "for that part of the timeline" % ", ".join(uncovered))
        except Exception as e:  # a bad system.jsonl must not lose the whole profile
            warnings.append("could not merge %s: %s: %s"
                            % (paths["system"], type(e).__name__, e))

    # measured resource ceilings from a `probe` run, if one was saved next to this capture
    ceilings = None
    if paths["probe"] and os.path.exists(paths["probe"]):
        try:
            with open(paths["probe"], errors="replace") as fh:
                doc = json.load(fh)
            ceilings = doc.get("ceilings") or None
            if ceilings:
                ceilings["source"] = os.path.abspath(paths["probe"])
        except Exception as e:
            warnings.append("could not read %s: %s: %s" % (paths["probe"], type(e).__name__, e))

    # cache-hit evidence
    hit = "compile_cache_hit" in abs_marks and abs_marks["compile_cache_hit"] is not None
    store = abs_marks.get("compile_cache_store") is not None
    cache_hit = bool(hit)
    if not hit and not store and abs_marks.get("compile_start") is None:
        warnings.append("no torch.compile cache evidence in the log (level above DEBUG?) — "
                        "cache_hit=false means 'unproven', not 'proven miss'")

    profile = {
        "schema": SCHEMA,
        "generated_at": round(time.time(), 3),
        "log": os.path.abspath(paths["log"]),
        "name": meta.get("name"),
        "run": meta.get("run"),
        "image": meta.get("image"),
        "vllm_args": meta.get("vllm_args"),
        "port": meta.get("port"),
        "t0": t0,
        "t0_source": t0_source,
        "total_s": total_s,
        "total_profiled_s": total_profiled_s,
        "host_prefix_s": host_prefix_s or None,
        "cache_hit": cache_hit,
        "cache_evidence": {"hit_marker": hit, "store_marker": store,
                           "cache_dir": values.get("compile_cache_dir")},
        "weights_gib": values.get("weights_gib"),
        "kv_tokens": values.get("kv_tokens"),
        "claimed": {k: v for k, v in sorted(values.items()) if k.endswith("_claimed")},
        "marks": marks,
        "spans": spans,
        "sys_summary": sys_summary,
        "ceilings": ceilings,
        "provision": {k: v for k, v in prov.items() if k != "raw"} or None,
        "host_events": [{"ev": e["ev"], "t": e["t"]} for e in host] or None,
        "sanity": ({"model": meta.get("sanity_model"), "ok": bool(meta.get("sanity_ok")),
                    "first_completion_s": first_completion_s}
                   if meta.get("sanity_model") or first_completion_s is not None else None),
        "totals": {
            "container_to_ready_s": total_s,
            "launch_to_ready_s": (round(meta["total_launch_to_ready"], 2)
                                  if meta.get("total_launch_to_ready") else None),
            "provision_to_ready_s": (round(host_prefix_s + total_s, 2)
                                     if host_prefix_s and total_s else None),
            "first_completion_s": first_completion_s,
        },
        "stats": stats,
        "warnings": warnings,
    }

    # flat convenience keys: marks, then reference-compatible span names
    profile.update({k: v for k, v in marks.items()})
    profile.update({k: v for k, v in values.items() if not k.endswith("_claimed")})
    profile.update({k: v for k, v in values.items() if k.endswith("_claimed")})
    for name, key in stages.LEGACY_SPAN_KEYS.items():
        profile[key] = top_by_name[name]["dur_s"] if name in top_by_name else None
    for s in spans:
        if s["level"] == 1 or s["source"] == "host" or s["name"] in ("ready_to_health", "first_completion"):
            profile["span_" + s["name"]] = s["dur_s"]
    return profile


def write_profile(profile, out_path):
    with open(out_path, "w") as fh:
        json.dump(profile, fh, indent=1, sort_keys=False)
        fh.write("\n")
    return out_path


def cmd_parse(args):
    profile = build_profile(args.target, args.meta, args.host_events, args.provision,
                            args.system_jsonl, args.probe_json)
    paths = resolve_inputs(args.target, args.meta, args.host_events, args.provision,
                           args.system_jsonl, args.probe_json)
    out = args.out or os.path.join(paths["out_dir"], "profile.json")
    write_profile(profile, out)
    if args.quiet:
        print(out)
    else:
        from .report import render
        print(render(profile, width=args.width))
        print("\nwrote %s" % out)
    return 0
