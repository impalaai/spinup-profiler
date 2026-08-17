"""vLLM log patterns for cold-start stage extraction.

The `PATTERNS` table is lifted from the field-verified research harness
(`impala-spinup-research/harness/parse_stages.py`) and extended — the names are
part of this module's contract, do not rename them. Verified against
`vllm/vllm-openai:v0.20.x` with `VLLM_LOGGING_LEVEL=DEBUG`; the compile,
cudagraph and memory-profiling marks only appear at DEBUG level.
"""

import datetime
import re

# `docker logs -t` prefixes every line with an RFC3339Nano UTC timestamp.
TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+(?:Z|[+-]\d{2}:\d{2}))\s+(.*)$"
)


def ts(s):
    """RFC3339Nano -> epoch seconds. Nanoseconds are truncated to micros (3.10 safe)."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    s = re.sub(
        r"\.(\d{1,6})\d*(?=[+-]\d{2}:\d{2}$)",
        lambda m: "." + m.group(1).ljust(6, "0"),
        s,
    )
    return datetime.datetime.fromisoformat(s).timestamp()


# (name, regex, keep_first)
#   keep_first=True  -> first occurrence wins (stage entry)
#   keep_first=False -> last occurrence wins (stage exit / summary line)
# A capturing group means vLLM itself claims a duration/size on that line; the
# last group is stored as "<name>_claimed".
PATTERNS = [
    ("api_args",          re.compile(r"non-default args"), True),
    ("engine_core_up",    re.compile(r"Waiting for init message from front-end"), True),
    ("load_start",        re.compile(r"Starting to load model"), True),
    ("stream_start",      re.compile(r"Loading safetensors using Runai Model Streamer:\s+0%|Loading weights on cuda|Loading pt checkpoint|Loading safetensors checkpoint"), True),
    ("weights_done",      re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"), False),
    ("compile_start",     re.compile(r"Start compiling function"), True),
    ("compile_done",      re.compile(r"torch\.compile took ([\d.]+) s in total"), False),
    ("compile_cache_hit", re.compile(r"Directly load the compiled graph|Directly lo?ad the compiled|load compiled artifact|hit the cache"), True),
    ("mem_profile",       re.compile(r"Memory profiling takes ([\d.]+) seconds"), False),
    ("kv_size",           re.compile(r"GPU KV cache size: ([\d,]+) tokens"), False),
    ("graphs_done",       re.compile(r"Graph capturing finished in (\d+) secs"), False),
    ("init_engine_done",  re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"), False),
    ("ready",             re.compile(r"Application startup complete"), False),
    # --- extensions (not in the research harness) ---
    ("compile_cache_dir", re.compile(r"Using cache directory: (\S+) for vLLM's torch\.compile"), True),
    ("compile_cache_store", re.compile(r"Store the \d+-th graph for compile range"), True),
    ("kv_mem",            re.compile(r"Available KV cache memory: ([\d.]+) GiB"), False),
    ("server_listening",  re.compile(r"Starting vLLM server on http"), False),
]

# Patterns whose *every* occurrence matters, because the mark we want is
# position-dependent (see stages.derive_graphs_start): vLLM captures cudagraphs
# twice — once while estimating memory during profiling, then for real after the
# KV cache is sized.
MULTI_PATTERNS = [
    ("cudagraph_capture", re.compile(r"Capturing a cudagraph on|Capturing CUDA graph")),
]

# Marks that carry a claimed duration/size, for the README pattern table.
CLAIMED_KEYS = {
    "weights_done_claimed": "vLLM's own weight-load seconds",
    "compile_done_claimed": "vLLM's own torch.compile seconds",
    "mem_profile_claimed": "vLLM's own memory-profiling seconds",
    "graphs_done_claimed": "vLLM's own cudagraph-capture seconds",
    "init_engine_done_claimed": "vLLM's own engine-init seconds",
    "kv_mem_gib": "available KV cache memory, GiB (kept out of `claimed`)",
}
