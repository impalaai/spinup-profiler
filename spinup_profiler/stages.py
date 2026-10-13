"""The spin-up stage model: which spans exist, how they are derived, what they mean.

One place defines the timeline so `parse` and `report` cannot disagree.

Vocabulary
  mark   an instant, in seconds from t0 (the container start).
  span   a [start, end) interval named after the work happening inside it.
  level  0 = the partition of the timeline (bars sum to the total)
         1 = a refinement of a level-0 span (engine_init's internals)
"""

# --- host / provisioning prefix spans (need events.jsonl / provision.json / meta) ---
# Each is (name, description). Their timestamps are absolute epoch, resolved in parse.
HOST_SPANS = [
    ("provision_to_ssh", "API request for the box -> SSH answers (from provision.json)"),
    ("host_gpu_ready", "SSH up -> nvidia-smi answers (driver loaded)"),
    ("host_docker_ready", "GPU ready -> docker + nvidia runtime usable"),
    ("image_pull", "docker pull of the vLLM image (~32 GiB uncompressed)"),
    ("prefetch", "parallel read of the weights to warm the page cache before launch"),
    ("launch_to_container", "`docker run` issued -> container process started"),
]

# --- container timeline: the level-0 partition ---
# (name, from_mark, to_mark, description)
TOP_SPANS = [
    ("imports", "t0", "api_args",
     "container launch + python/torch imports until vLLM echoes its CLI args"),
    ("spawn_workers", "api_args", "engine_core_up",
     "API server config resolution + EngineCore process spawn (imports again in the child)"),
    ("pre_load", "engine_core_up", "load_start",
     "engine handshake, platform/attention-backend selection, tokenizer, before weights"),
    ("weight_load", "load_start", "weights_done",
     "model weight transport + placement on the GPU"),
    ("engine_init", "weights_done", "init_engine_done",
     "profile, torch.compile, KV cache creation, kernel warm-up, cudagraph capture"),
    ("api_startup", "init_engine_done", "ready",
     "chat-template / multimodal warm-up, route registration, uvicorn startup"),
]

# --- level-1 refinement of engine_init ---
# `frm`/`to` are candidate mark lists: the first mark that exists wins, so a
# missing DEBUG-only mark merges the span instead of breaking the partition.
ENGINE_INIT_LEAVES = [
    dict(name="engine_pre_compile", frm=["weights_done"], to=["compile_start"],
         desc="post-load bookkeeping before the first compiled forward"),
    dict(name="compile", frm=["compile_start"], to=["compile_done"],
         desc="torch.compile / inductor (cacheable via VLLM_CACHE_ROOT)"),
    dict(name="mem_profile_tail", frm=["compile_done"], to=["kv_size", "mem_profile"],
         desc="rest of memory profiling incl. cudagraph memory estimation, until the KV cache is sized"),
    dict(name="warmup_kernels", frm=["kv_size", "mem_profile"], to=["graphs_start"],
         desc="model warm-up after KV allocation — mostly kernel autotune/compile (cacheable)"),
    dict(name="graph_capture", frm=["graphs_start", "kv_size", "mem_profile"], to=["graphs_done"],
         desc="actual CUDA graph capture"),
    dict(name="engine_init_tail", frm=["graphs_done"], to=["init_engine_done"],
         desc="engine-init wrap-up"),
]

# --- tail spans (need meta.json) ---
TAIL_SPANS = [
    ("ready_to_health", "gap between the log's startup-complete line and the first /health 200"),
    ("first_completion", "/health 200 -> first successful chat completion (sanity)"),
]

DESCRIPTIONS = dict(
    [(n, d) for n, _f, _t, d in TOP_SPANS]
    + [(l["name"], l["desc"]) for l in ENGINE_INIT_LEAVES]
    + list(HOST_SPANS)
    + list(TAIL_SPANS)
)

# Gap spans are named after the span they follow, so they need a prefix rule.
UNATTRIBUTED = "unattributed"


def describe(name):
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    if name.startswith(UNATTRIBUTED):
        return "interval no log mark accounts for (a mark was missing, not idle time)"
    return ""

# Mark order for the absolute timeline in profile.json.
MARK_ORDER = [
    "first_log", "api_args", "engine_core_up", "load_start", "stream_start",
    "weights_done", "compile_start", "compile_done", "mem_profile", "kv_size",
    "graphs_start", "graphs_done", "init_engine_done", "server_listening", "ready",
]

# Reference-compatible flat span keys (impala-spinup-research/harness/parse_stages.py).
LEGACY_SPAN_KEYS = {
    "imports": "span_imports",
    "spawn_workers": "span_spawn_workers",
    "pre_load": "span_pre_load",
    "weight_load": "span_weight_load",
    "engine_init": "span_engine_init",
    "api_startup": "span_api_startup",
}


def derive_graphs_start(cudagraph_marks, marks):
    """First *real* cudagraph capture: the first capture line after the KV cache is sized.

    Captures also happen during memory profiling (cudagraph memory estimation),
    which is why the first occurrence overall is the wrong mark.
    """
    anchor = None
    for key in ("kv_size", "mem_profile", "compile_done"):
        if marks.get(key) is not None:
            anchor = marks[key]
            break
    if anchor is None:
        return None
    later = [t for t in cudagraph_marks if t > anchor]
    return min(later) if later else None


# --- bottleneck hints: keyed by leaf span name -----------------------------------
# Every line is a lever measured in the "Minimizing vLLM Spin-Up Time" research
# (July 2026) and the cross-cloud benchmark (Aug 2026).
HINTS = {
    "provision_to_ssh":
        "provisioning dominates: keep a warm/dark pool (stopped baked VM -> start reaches SSH in 30-95s) "
        "instead of creating a fresh instance; see the Impala spin-up research.",
    "host_gpu_ready":
        "driver init is slow on this image: use a provider image with the driver preinstalled, "
        "or persist nvidia module load across boots (nvidia-persistenced).",
    "host_docker_ready":
        "docker/nvidia-runtime install is on the critical path: bake docker + nvidia-container-toolkit "
        "into the boot disk / AMI.",
    "image_pull":
        "image pull dominates (the vLLM image is ~32 GiB): bake it into a reusable boot disk, AMI or host "
        "image cache, or pre-pull it in parallel with weight staging.",
    "prefetch":
        "the page-cache warm-up itself is now the biggest stage: lower --prefetch-jobs or prefetch only "
        "the shards the engine actually reads — it is only worth what it removes from weight_load.",
    "launch_to_container":
        "big gap between `docker run` and container start = an implicit docker pull — pre-pull or bake the image.",
    "imports":
        "python/torch import cost: keep the image layers on local NVMe (not network storage) and reuse a warm "
        "page cache; there is little else to win here (~15-25s is normal).",
    "spawn_workers":
        "engine-core spawn dominates: set VLLM_WORKER_MULTIPROC_METHOD=fork (~-8s, the child skips re-importing "
        "torch); check you are not paying a second full import per worker.",
    "pre_load":
        "pre-load overhead: usually tokenizer/config fetches from HF — pre-stage the repo locally "
        "(HF_HUB_OFFLINE=1 with a warm cache) so nothing hits the network.",
    "weight_load":
        "weight transport dominates: try runai_streamer from same-region object storage (~1.4 GB/s measured) "
        "or local NVMe; HF hub cold-CDN is the slow path; see the Impala spin-up research.",
    "compile":
        "torch.compile dominates: persist VLLM_CACHE_ROOT on reused storage — NOTE: the cache key includes the "
        "--model path string, so warm the cache with the exact --model value the replay will use.",
    "mem_profile_tail":
        "memory profiling dominates: it re-runs compiled/estimated forwards — a warm torch.compile cache "
        "(persisted VLLM_CACHE_ROOT) cut this 107s -> 22s in the research.",
    "warmup_kernels":
        "post-KV warm-up dominates: this is mostly (cacheable) kernel autotune/compile, not graph capture — "
        "persist VLLM_CACHE_ROOT; --enforce-eager will NOT remove it.",
    "graph_capture":
        "cudagraph capture dominates: reduce --cudagraph-capture-sizes / --max-num-seqs, or set --enforce-eager "
        "(costs steady-state latency); with a warm compile cache capture alone is only ~20s.",
    "engine_init_tail": "engine-init wrap-up dominates — unusual; inspect the log tail directly.",
    "engine_pre_compile": "post-load bookkeeping dominates — unusual; inspect the log around weight-load completion.",
    UNATTRIBUTED:
        "the biggest interval has no marks around it: re-run with VLLM_LOGGING_LEVEL=DEBUG (and "
        "VLLM_CONFIGURE_LOGGING=1) so the compile / profiling / capture marks are emitted — "
        "without them this tool cannot say what that time was spent on.",
    "api_startup":
        "API startup dominates: chat-template + multimodal warm-up talk to HF (404 retries cost seconds) — "
        "pre-stage the repo / set HF_HUB_OFFLINE=1, and pass --chat-template-content-format to skip detection.",
    "ready_to_health": "health-detection lag only — an artifact of the poll interval, not a real stage.",
    "first_completion":
        "first completion is slow after ready: that is prefill on a cold KV/prefix cache — send a tiny warm-up "
        "request before declaring the box in service.",
}

# --- system-metric attribution (needs a merged system.jsonl) ---------------------
# Shared with probe.py, which judges observed rates against measured ceilings with the same numbers.
AT_CEILING = 0.80           # observed/ceiling above this => the hardware is being used
SCALES = 1.30               # parallel/single above this => the subsystem rewards concurrency
NET_BOUND_FRACTION = 0.70   # stage avg rx this close to the run's peak rx => the link is the limit
GPU_BUSY_PCT = 50.0         # gpu utilisation that counts as "the GPU is the one working"
GPU_IDLE_PCT = 20.0
SINGLE_CORE_LO, SINGLE_CORE_HI = 0.70, 1.45   # multiples of (100 / cpu_count)

NET_STAGES = ("weight_load", "image_pull", "pre_load", "api_startup")
CPU_STAGES = ("imports", "spawn_workers", "pre_load", "compile")
GPU_STAGES = ("warmup_kernels", "graph_capture", "compile", "mem_profile_tail", "engine_init")


def _avg(span, metric):
    v = (span.get("sys") or {}).get(metric)
    return v["avg"] if v else None


def _prefetch_advice(profile, name, observed_mbs, spans_by_name):
    """Suggest --prefetch when a disk-bound weight_load sits below a disk that scales."""
    if name != "weight_load" or "prefetch" in spans_by_name:
        return ""
    cl = profile.get("ceilings") or {}
    best, one = cl.get("disk_read_best_mbs"), cl.get("disk_read_1job_mbs")
    if not cl.get("disk_scales") or not best or observed_mbs >= AT_CEILING * best:
        return ""
    span = spans_by_name.get(name) or {}
    saving = ""
    if span.get("dur_s") and observed_mbs:
        would_be = span["dur_s"] * observed_mbs / best
        saving = (" On the measured ceiling the same bytes would take ~%.0fs instead of %.0fs."
                  % (would_be, span["dur_s"]))
    return (" `probe` measured this disk at %s MB/s with parallel readers vs %s MB/s single-job, so "
            "the device has more to give than this stage is taking: run with "
            "`--prefetch <weights dir> --prefetch-jobs 8` to warm the page cache first (measured: "
            "31 GB warmed in 8.4s, weight_load 62.9 -> 12.5s).%s" % (best, one, saving))


def system_hints(profile, spans_by_name):
    """Attribute the expensive stages to network / disk / CPU / GPU using the sampled counters."""
    summary = profile.get("sys_summary")
    if not summary:
        return []
    out = []
    peak_rx = summary.get("peak_net_rx_mbs") or 0.0
    cpu_count = summary.get("cpu_count")
    one_core = (100.0 / cpu_count) if cpu_count else None

    for name in NET_STAGES:
        span = spans_by_name.get(name)
        rx, rd = _avg(span or {}, "net_rx_mbs"), _avg(span or {}, "disk_read_mbs")
        if span is None or rx is None:
            continue
        if rd is not None and rd > max(rx, 1.0):
            msg = ("%s is DISK-bound: %.0f MB/s read from disk vs %.0f MB/s off the network — "
                   "the local device is the limit, not the link." % (name, rd, rx))
            msg += _prefetch_advice(profile, name, rd, spans_by_name)
            out.append(msg)
        elif peak_rx and rx >= NET_BOUND_FRACTION * peak_rx and rx > 1.0:
            out.append("%s is NETWORK-bound at %.0f MB/s (%.0f%% of the %.0f MB/s peak this run "
                       "ever reached) — the transport is the limit, so a faster source (same-region "
                       "object storage, runai_streamer) is the only lever."
                       % (name, rx, 100.0 * rx / peak_rx, peak_rx))

    if one_core:
        for name in CPU_STAGES:
            span = spans_by_name.get(name)
            cpu, gpu = _avg(span or {}, "cpu_pct"), _avg(span or {}, "gpu_util_pct")
            if span is None or cpu is None:
                continue
            if (SINGLE_CORE_LO * one_core <= cpu <= SINGLE_CORE_HI * one_core
                    and (gpu is None or gpu < GPU_IDLE_PCT)):
                out.append("%s is single-core CPU-bound: %.1f%% of %d cores is ~1 core saturated "
                           "(1 core = %.1f%%) with the GPU idle — this work does not parallelise, "
                           "so only a faster core or less work (warm cache, fewer imports) helps."
                           % (name, cpu, cpu_count, one_core))

    for name in GPU_STAGES:
        span = spans_by_name.get(name)
        gpu = _avg(span or {}, "gpu_util_pct")
        if span is None or gpu is None:
            continue
        if gpu >= GPU_BUSY_PCT:
            out.append("%s is GPU compute-bound at %.0f%% utilisation — expected for kernel "
                       "autotune/capture; the lever is doing it once and caching it, not faster IO."
                       % (name, gpu))
        elif name == "compile" and gpu < GPU_IDLE_PCT:
            out.append("compile is CPU-bound (inductor) with the GPU at %.0f%% — more/faster cores "
                       "or a warm VLLM_CACHE_ROOT are the only levers." % gpu)
    return out


# Composite rules: (predicate(profile, by_name) -> bool, hint)
def composite_hints(profile, dur):
    """Extra hints that depend on more than one span. `dur` maps span name -> seconds."""
    out = []
    compile_family = sum(dur.get(k, 0.0) for k in ("compile", "mem_profile_tail",
                                                   "warmup_kernels", "graph_capture"))
    total = profile.get("total_profiled_s") or profile.get("total_s") or 0.0
    if total and compile_family / total >= 0.30:
        if not profile.get("cache_hit"):
            out.append(
                "compile+profile+capture is %.0f%% of the spin-up and the torch.compile cache MISSED: persist "
                "VLLM_CACHE_ROOT on reused storage (disk/volume/bucket) — NOTE: the cache key includes the "
                "--model path string, so warm it with the exact --model value the replay uses. "
                "Measured effect: compile 49->7s, warm-up+capture 192->20s, profiling 107->22s."
                % (100.0 * compile_family / total)
            )
        else:
            out.append(
                "compile+profile+capture is %.0f%% of the spin-up even though the compile cache HIT: check the "
                "cache is complete for this exact --model/--max-model-len/cudagraph-sizes combination."
                % (100.0 * compile_family / total)
            )
    if dur.get("image_pull") and dur.get("weight_load"):
        out.append(
            "image pull and weight load are both on the critical path and independent — overlap them "
            "(pull while hydrating weights) to hide the smaller one (%.0fs)."
            % min(dur["image_pull"], dur["weight_load"])
        )
    if profile.get("cache_hit") is False and "compile" not in dur:
        out.append(
            "no torch.compile marks were found: the log level is probably above DEBUG, so cache_hit=false "
            "means 'no evidence', not 'proven miss'. Re-run with VLLM_LOGGING_LEVEL=DEBUG."
        )
    return out
