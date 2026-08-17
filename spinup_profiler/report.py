"""ASCII waterfall + bottleneck verdict for a profile.json."""

import json
import os

from . import stages

BAR = "#"
NAME_W = 30  # stage-name column width
SUBPIXEL = ":"


def _bar(dur, total, width):
    if not total or dur is None or dur <= 0:
        return ""
    n = int(round(width * dur / total))
    return BAR * n if n else SUBPIXEL


def _fmt(v, unit=""):
    return "n/a" if v is None else ("%.2f%s" % (v, unit))


def leaf_spans(profile):
    """The finest available partition: level-0 spans, with engine_init replaced by its leaves."""
    spans = profile.get("spans") or []
    refined = {s["parent"] for s in spans if s["level"] == 1 and s.get("parent")}
    leaves = [s for s in spans if (s["level"] == 1) or (s["level"] == 0 and s["name"] not in refined)]
    return sorted(leaves, key=lambda s: s["start_s"])


def verdict(profile, top_n=3):
    """(top_n spans by duration, bottleneck span) over the finest partition."""
    ranked = sorted(leaf_spans(profile), key=lambda s: s["dur_s"], reverse=True)
    ranked = [s for s in ranked if s["name"] != "ready_to_health"]
    return ranked[:top_n], (ranked[0] if ranked else None)


def hints_for(profile, top):
    dur = {s["name"]: s["dur_s"] for s in leaf_spans(profile)}
    out = []
    for s in top:
        h = stages.HINTS.get(s["name"])
        if h is None and s["name"].startswith(stages.UNATTRIBUTED):
            h = stages.HINTS[stages.UNATTRIBUTED]
        if h:
            out.append("%s (%.1fs, %.0f%%): %s" % (s["name"], s["dur_s"], s["pct"] or 0.0, h))
    out += stages.composite_hints(profile, dur)
    # Attribution from the sampled counters. Only the finest partition, so a refined parent
    # (engine_init) does not repeat what its leaves already said, and only stages worth naming.
    by_name = {s["name"]: s for s in leaf_spans(profile) if s["dur_s"] >= 1.0}
    out += stages.system_hints(profile, by_name)
    return out


def _pair(sysdata, metric, fmt="%.1f"):
    """'avg / peak' for one metric, or a dash when it was not sampled."""
    v = (sysdata or {}).get(metric)
    if not v:
        return "-"
    return (fmt + " / " + fmt) % (v["avg"], v["peak"])


def io_table(profile, width=None):
    """Per-stage system metrics: what the box was doing during each stage."""
    spans = [s for s in (profile.get("spans") or []) if s.get("sys")]
    if not spans:
        return []
    summary = profile.get("sys_summary") or {}
    L = ["", " PER-STAGE SYSTEM METRICS   avg / peak   (MB = 1e6 bytes; cpu%% over all %s cores)"
         % (summary.get("cpu_count") or "?")]
    head = " %-*s %15s %15s %13s %11s" % (NAME_W, "STAGE", "NET RX MB/s",
                                          "DISK RD MB/s", "CPU %", "GPU %")
    L.append(head)
    L.append(" " + "-" * NAME_W + " " + "-" * 15 + " " + "-" * 15 + " " + "-" * 13 + " " + "-" * 11)
    for s in sorted(spans, key=lambda s: (s["start_s"], s["level"])):
        label = ("  |- " if s["level"] else " ") + s["name"]
        L.append(" %-*s %15s %15s %13s %11s" % (
            NAME_W, label[:NAME_W],
            _pair(s["sys"], "net_rx_mbs"), _pair(s["sys"], "disk_read_mbs"),
            _pair(s["sys"], "cpu_pct"), _pair(s["sys"], "gpu_util_pct", "%.0f")))
    totals = []
    for key, label in (("net_rx_gb", "net in"), ("net_tx_gb", "net out"),
                       ("disk_read_gb", "disk read"), ("disk_write_gb", "disk written")):
        if summary.get(key) is not None:
            totals.append("%s %.2f GB" % (label, summary[key]))
    if totals:
        L.append(" totals over %ss of sampling: %s" % (summary.get("window_s"), "   ".join(totals)))
    where = []
    if summary.get("ifaces"):
        where.append("ifaces %s" % ",".join(summary["ifaces"]))
    if summary.get("disks"):
        where.append("disks %s" % ",".join(summary["disks"]))
    if summary.get("gpu_samples"):
        where.append("%d gpu samples" % summary["gpu_samples"])
    if where:
        L.append(" sampled: %s" % "   ".join(where))
    return L


def render(profile, width=44, top_n=3, show_hints=True, show_io=True):
    L = []
    name = profile.get("name") or os.path.basename(os.path.dirname(profile.get("log") or "")) or "run"
    run = profile.get("run")
    title = "vLLM SPIN-UP PROFILE   %s%s" % (name, "" if run is None else " run %s" % run)
    L.append("=" * 78)
    L.append(" " + title)
    L.append("=" * 78)

    total = profile.get("total_profiled_s") or profile.get("total_s")
    L.append(" image             %s" % (profile.get("image") or "?"))
    if profile.get("vllm_args"):
        L.append(" vllm args         %s" % profile["vllm_args"])
    if profile.get("provision"):
        p = profile["provision"]
        L.append(" box               %s" % " ".join(
            "%s=%s" % (k, v) for k, v in p.items() if k in ("provider", "cloud", "region", "instance_type", "gpu")))
    L.append(" container->ready  %s s   (t0 = %s)" % (_fmt(profile.get("total_s")), profile.get("t0_source")))
    if profile.get("host_prefix_s"):
        L.append(" host prefix       %s s   (host spans measured before the container timeline)"
                 % _fmt(profile["host_prefix_s"]))
    L.append(" profiled total    %s s" % _fmt(total))
    cache = "HIT" if profile.get("cache_hit") else "MISS"
    if not profile.get("cache_hit") and not (profile.get("cache_evidence") or {}).get("store_marker"):
        cache = "MISS?(no evidence)"
    L.append(" torch.compile     cache %s%s" % (
        cache,
        "   dir %s" % profile["cache_evidence"]["cache_dir"]
        if (profile.get("cache_evidence") or {}).get("cache_dir") else ""))
    extra = []
    if profile.get("weights_gib") is not None:
        extra.append("weights %.2f GiB" % profile["weights_gib"])
    if profile.get("kv_tokens") is not None:
        extra.append("KV cache %s tokens" % format(profile["kv_tokens"], ","))
    if (profile.get("sanity") or {}).get("first_completion_s") is not None:
        extra.append("first completion %.2f s" % profile["sanity"]["first_completion_s"])
    if extra:
        L.append(" model             %s" % "   ".join(extra))
    pf = next((s for s in (profile.get("spans") or []) if s["name"] == "prefetch"), None)
    if pf:
        L.append(" prefetch          %.2f GB from %s file(s) in %.2fs @ %s MB/s (%s jobs)"
                 % (pf.get("gb") or 0.0, pf.get("files"), pf["dur_s"], pf.get("mbs"),
                    pf.get("jobs")))
    if profile.get("ceilings"):
        cl = profile["ceilings"]
        bits = []
        if cl.get("disk_read_best_mbs"):
            bits.append("disk read %s MB/s%s" % (cl["disk_read_best_mbs"],
                                                 " (scales)" if cl.get("disk_scales") else ""))
        if cl.get("net_best_mbs"):
            bits.append("net %s MB/s%s" % (cl["net_best_mbs"],
                                           " (scales)" if cl.get("net_scales") else ""))
        if bits:
            L.append(" measured ceilings %s" % "   ".join(bits))
    L.append("")

    # waterfall
    L.append(" %-*s %8s %8s %6s  %s" % (NAME_W, "STAGE", "START", "DUR", "%", "SHARE OF TOTAL"))
    rule = (" " + "-" * NAME_W + " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 6 + "  " + "-" * width)
    L.append(rule)
    for s in sorted(profile.get("spans") or [], key=lambda s: (s["start_s"], s["level"])):
        label = ("  |- " if s["level"] else " ") + s["name"]
        if s["source"] == "host":
            label += " *"
        L.append(" %-*s %8.2f %8.2f %5.1f%%  %s" % (
            NAME_W, label[:NAME_W], s["start_s"], s["dur_s"], s["pct"] or 0.0,
            _bar(s["dur_s"], total, width)))
    L.append(rule)
    L.append(" %-*s %8s %8.2f %5.1f%%" % (NAME_W, "TOTAL (level-0 spans)", "", total or 0.0, 100.0))
    if profile.get("host_prefix_s"):
        L.append(" * host span: measured on the box, before the container timeline.")
    L.append("")

    top, boss = verdict(profile, top_n)
    L.append(" TOP %d STAGES (finest partition)" % len(top))
    for i, s in enumerate(top, 1):
        L.append("  %d. %-*s %8.2f s  %5.1f%%   %s" % (
            i, NAME_W, s["name"], s["dur_s"], s["pct"] or 0.0, s["desc"]))
    L.append("")
    if boss:
        L.append(" BOTTLENECK: %s (%.1fs, %.0f%% of %.1fs)" % (
            boss["name"], boss["dur_s"], boss["pct"] or 0.0, total or 0.0))
    else:
        L.append(" BOTTLENECK: undetermined (no spans derivable)")

    if profile.get("claimed"):
        L.append("")
        L.append(" vLLM's own claims: %s" % "  ".join(
            "%s=%.2fs" % (k.replace("_claimed", ""), v) for k, v in profile["claimed"].items()))

    if show_io:
        L += io_table(profile)

    if show_hints:
        hs = hints_for(profile, top)
        if hs:
            L.append("")
            L.append(" HINTS")
            for h in hs:
                L.append(_wrap(h, 78, "  * ", "    "))

    if profile.get("warnings"):
        L.append("")
        L.append(" WARNINGS")
        for w in profile["warnings"]:
            L.append(_wrap(w, 78, "  ! ", "    "))
    return "\n".join(L)


def _wrap(body, width, first_prefix, cont_prefix):
    def add(line, word):
        return line + word if (not line or line.endswith(" ")) else line + " " + word

    lines, cur = [], first_prefix
    for w in body.split():
        cand = add(cur, w)
        if len(cand) > width and cur.strip():
            lines.append(cur.rstrip())
            cur = add(cont_prefix, w)
        else:
            cur = cand
    if cur.strip():
        lines.append(cur.rstrip())
    return "\n".join(lines)


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "profile.json")
    with open(path) as fh:
        return json.load(fh)


def cmd_report(args):
    profile = load(args.profile)
    if profile.get("schema") != "spinup-profiler/profile/v1":
        print("warning: unexpected schema %r — rendering anyway" % profile.get("schema"))
    text = render(profile, width=args.width, top_n=args.top, show_hints=not args.no_hints,
                  show_io=not args.no_io)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0
