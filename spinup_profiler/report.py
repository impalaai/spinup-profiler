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
    return out


def render(profile, width=44, top_n=3, show_hints=True):
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
    text = render(profile, width=args.width, top_n=args.top, show_hints=not args.no_hints)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0
