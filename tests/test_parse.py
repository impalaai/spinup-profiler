#!/usr/bin/env python3
"""Regression test for the stage parser — plain python, no pytest.

Run:  python3 tests/test_parse.py     (exit 0 = pass, 1 = fail)

The expected numbers come from the field-verified research harness output for the
same log (impala-spinup-research/cross-cloud/results/verda/stages-base-hf-run2.json),
so a drift here means the patterns or the span derivation changed behaviour.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spinup_profiler.parse import build_profile  # noqa: E402
from spinup_profiler.report import leaf_spans, render, verdict  # noqa: E402

EX = os.path.join(ROOT, "examples")

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


def main():
    print("parsing %s" % os.path.join(EX, "docker.log"))
    p = build_profile(os.path.join(EX, "docker.log"),
                      meta_path=os.path.join(EX, "meta.json"))

    # --- headline numbers (1xH100, Qwen3.5-27B-FP8, cold, HF weights, no compile cache)
    near("total_s", p["total_s"], 470.06, 0.5)
    near("span_weight_load", p["span_weight_load"], 125.7, 1.0)

    # --- the rest of the level-0 partition, against the reference harness output
    for key, want in (("span_imports", 20.99), ("span_spawn_workers", 40.90),
                      ("span_pre_load", 15.69), ("span_engine_init", 242.54),
                      ("span_api_startup", 23.96)):
        near(key, p[key], want, 0.1)

    # --- marks
    for key, want in (("api_args", 20.99), ("engine_core_up", 61.89), ("load_start", 77.58),
                      ("stream_start", 79.61), ("weights_done", 203.33), ("compile_done", 253.96),
                      ("graphs_done", 445.84), ("init_engine_done", 445.87), ("ready", 469.83)):
        near("mark %s" % key, p["marks"][key], want, 0.1)

    # --- compile + graph capture must be present and split out
    check("compile span present", p.get("span_compile") is not None)
    check("graph_capture span present", p.get("span_graph_capture") is not None)
    check("warmup_kernels span present", p.get("span_warmup_kernels") is not None)
    near("span_compile", p.get("span_compile"), 49.21, 0.5)
    near("span_graph_capture", p.get("span_graph_capture"), 19.45, 0.5)
    near("claimed torch.compile", p["claimed"].get("compile_done_claimed"), 49.21, 0.01)
    near("claimed graph capture", p["claimed"].get("graphs_done_claimed"), 20.0, 0.01)

    # --- this run wrote its compile cache, so it is a proven MISS
    check("cache_hit is False", p["cache_hit"] is False, "(got %r)" % p["cache_hit"])
    check("cache miss is evidenced (store marker seen)", p["cache_evidence"]["store_marker"] is True)

    # --- model facts
    near("weights_gib", p["weights_gib"], 28.54, 0.01)
    check("kv_tokens = 669320", p["kv_tokens"] == 669320, "(got %r)" % p["kv_tokens"])

    # --- the partition must add up
    level0 = [s for s in p["spans"] if s["level"] == 0]
    near("level-0 spans sum to total_profiled_s",
         round(sum(s["dur_s"] for s in level0), 2), p["total_profiled_s"], 0.02)
    leaves = [s for s in p["spans"] if s["parent"] == "engine_init"]
    near("engine_init leaves sum to engine_init",
         round(sum(s["dur_s"] for s in leaves), 2), p["span_engine_init"], 0.02)
    check("no unattributed engine_init residual",
          not any(s["name"] == "engine_init_unattributed" for s in leaves))

    # --- verdict + report must render
    top, boss = verdict(p)
    check("bottleneck is weight_load", boss["name"] == "weight_load", "(got %s)" % boss["name"])
    check("top-3 has 3 entries", len(top) == 3, "(got %d)" % len(top))
    check("leaf partition is monotonic",
          all(a["start_s"] <= b["start_s"] for a, b in zip(leaf_spans(p), leaf_spans(p)[1:])))
    text = render(p)
    for needle in ("BOTTLENECK:", "weight_load", "warmup_kernels", "HINTS", "VLLM_CACHE_ROOT"):
        check("report mentions %r" % needle, needle in text)

    # --- committed example stays in sync with the parser
    committed = json.load(open(os.path.join(EX, "profile.json")))
    check("examples/profile.json is current",
          committed["total_s"] == p["total_s"]
          and committed["span_weight_load"] == p["span_weight_load"]
          and committed["marks"] == p["marks"],
          "(regenerate: python3 -m spinup_profiler parse examples/docker.log "
          "--meta examples/meta.json --out examples/profile.json)")

    print("\n%d checks, %d failures" % (CHECKS[0], len(FAILURES)))
    if FAILURES:
        print("FAILED: %s" % ", ".join(FAILURES))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
