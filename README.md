# spinup-profiler

Stage-level profiler for **vLLM cold spin-up**. It runs one instrumented spin-up on a GPU box,
decomposes it into the stages that actually cost time (image pull, imports, engine spawn, weight
load, torch.compile, kernel warm-up, cudagraph capture, API startup), and tells you which one is
the bottleneck — plus the lever that shortened it in the IMP-2773 research.

* **Stdlib only.** Python 3.10+, no pip install, nothing to vendor: target boxes have a bare
  `python3`. `urllib` + `argparse` + `json` + `re` + `subprocess`, that's it.
* **Field-verified patterns.** The log regexes are lifted from the research harness that produced
  the cross-cloud spin-up benchmark, verified against `vllm/vllm-openai:v0.20.x` at
  `VLLM_LOGGING_LEVEL=DEBUG`. The parser reproduces that harness's numbers on the same log
  (see `tests/test_parse.py`).
* **Four subcommands.** `run` (measure), `parse` (derive `profile.json`), `report` (waterfall +
  verdict), `sleepwake` (measure sleep-mode warm start instead of a cold start).
* **Per-stage system metrics.** `run` samples host network / disk / CPU / GPU counters at 1 Hz, and
  the report attributes each stage to what actually limited it — network-bound, disk-bound,
  single-core CPU-bound or GPU compute-bound.

```
git clone git@github.com:impalaai/spinup-profiler.git
cd spinup-profiler
python3 -m spinup_profiler report examples/profile.json     # or: bin/spinup-profile report ...
```

Copy the repo to the box (`scp -r`, `git clone`) and call `bin/spinup-profile`; it sets
`PYTHONPATH` for you. `python3 -m spinup_profiler <cmd>` from the repo root is equivalent.

---

## Quick start

### 1. Measure a cold spin-up on the box

```sh
bin/spinup-profile run \
    --image vllm/vllm-openai:v0.20.2 --port 8200 --gpus all --shm-size 48g \
    --drop-caches --label base-hf --run-index 2 \
    --out ~/runs --timeout 1800 \
    --sanity-model Qwen/Qwen3.5-27B-FP8 \
    -- --port 8200 --model Qwen/Qwen3.5-27B-FP8 --max-model-len 262144 \
       --reasoning-parser qwen3 --max-num-seqs 256
```

Everything after `--` is passed to the container verbatim. What `run` does, in order:

1. host preflight → `nvidia-smi -L`, `docker info`, nvidia runtime, and a **timed** `docker pull`
   if the image is absent (`--no-pull` to skip it and let `docker run` pull implicitly instead);
2. `--drop-caches` → `sync` + `echo 3 > /proc/sys/vm/drop_caches` (needs passwordless sudo; if it
   fails you get a warning, not a lie about the run being cold);
3. `docker run -d` with `VLLM_CONFIGURE_LOGGING=1 VLLM_LOGGING_LEVEL=DEBUG NO_COLOR=1` injected
   (`--no-default-env` to opt out — but then the compile/profile/capture spans disappear),
   repeatable `--env K=V` / `--volume SRC:DST` / `--docker-arg`;
4. polls `/health` at 5 Hz until 200, the container dies, or `--timeout` (progress every 30 s
   includes the container's last log line);
5. runs a sanity chat completion (`--sanity-model`, or the first id from `/v1/models`);
6. captures `docker logs -t` + `docker inspect`, writes the run dir, then parses and reports.

Run dir (`<out>/<label>/run<N>/`): `events.jsonl`, `docker.log`, `inspect.json`, `meta.json`,
`system.jsonl`, `profile.json`, `report.txt`. The container is left **running** unless you pass
`--teardown`, so you can follow up with `sleepwake` or a throughput probe. Exit code is non-zero if
the server never became ready or the sanity completion failed.

While all that happens a background sampler writes host counters to `system.jsonl` at 1 Hz
(`--sample-system` / `--no-sample-system`, default **on** when `/proc/net/dev` is readable;
`--sample-interval` to change the rate). It starts before the image pull and stops once the sanity
completion is done, so every stage — including `image_pull` — gets network/disk/CPU/GPU numbers.
Every tick is wrapped in `try/except`: a sampler that cannot read a counter drops that field, and
one that fails repeatedly stops itself and records why. It can never fail the run.

### 2. Re-parse a capture you already have

```sh
bin/spinup-profile parse ~/runs/base-hf/run2                       # run dir: picks up siblings
bin/spinup-profile parse examples/docker.log --meta examples/meta.json --out /tmp/profile.json
bin/spinup-profile parse ~/runs/base-hf/run2 \
    --host-events ~/runs/bootstrap.jsonl --provision ~/runs/provision.json
```

A run dir automatically uses `docker.log`, `meta.json`, `events.jsonl` and `provision.json` if
present; a bare log file uses only what you pass. Without `meta.json` the timeline is anchored on
the first log line instead of container start (you get a warning saying so).

### 3. Render the waterfall and the verdict

```sh
bin/spinup-profile report examples/profile.json            # --width 60 --top 5 --no-hints --out r.txt
```

See `examples/report.txt` for the full output. Abridged:

```
 STAGE                             START      DUR      %  SHARE OF TOTAL
 ------------------------------ -------- -------- ------  --------------------------------------------
  imports                           0.31    20.99   4.5%  ##
  spawn_workers                    21.30    40.90   8.7%  ####
  pre_load                         62.20    15.69   3.3%  #
  weight_load                      77.89   125.75  26.7%  ############
  engine_init                     203.64   242.54  51.6%  #######################
   |- compile                      205.06    49.21  10.5%  #####
   |- mem_profile_tail             254.27    57.49  12.2%  #####
   |- warmup_kernels               311.76   114.94  24.4%  ###########
   |- graph_capture                426.70    19.45   4.1%  ##
  api_startup                     446.18    23.96   5.1%  ##

 BOTTLENECK: weight_load (125.8s, 27% of 470.4s)
```

`engine_init` is shown twice: once as the level-0 span it is, then broken into the level-1 leaves
that partition it. The verdict ranks the **finest** partition, so `engine_init` never wins as a
lump — you get told which part of it is the problem.

When a `system.jsonl` was captured, the report also prints what the box was *doing* during each
stage and attributes the bottleneck to a subsystem (`--no-io` to omit the table):

```
 PER-STAGE SYSTEM METRICS   avg / peak   (MB = 1e6 bytes; cpu% over all 32 cores)
 STAGE                              NET RX MB/s    DISK RD MB/s         CPU %       GPU %
 ------------------------------ --------------- --------------- ------------- -----------
  imports                             0.0 / 0.0       0.0 / 0.0     3.1 / 3.1      0 / 0
  weight_load                     248.5 / 250.0     49.7 / 50.0   25.0 / 40.0      0 / 0
   |- compile                         0.0 / 0.0       0.0 / 0.0   39.4 / 40.0    10 / 10
   |- warmup_kernels                  0.0 / 0.0       0.0 / 0.0   12.0 / 12.0    90 / 90
 totals over 471.0s of sampling: net in 31.32 GB   net out 0.02 GB   disk read 6.25 GB   disk written 31.25 GB
 sampled: ifaces eth0   disks nvme0n1   236 gpu samples

  * weight_load is NETWORK-bound at 249 MB/s (99% of the 250 MB/s peak this run ever reached) —
    the transport is the limit, so a faster source (same-region object storage, runai_streamer)
    is the only lever.
  * imports is single-core CPU-bound: 3.1% of 32 cores is ~1 core saturated (1 core = 3.1%) with
    the GPU idle — this work does not parallelise, ...
  * warmup_kernels is GPU compute-bound at 90% utilisation — expected for kernel autotune/capture;
    the lever is doing it once and caching it, not faster IO.
  * compile is CPU-bound (inductor) with the GPU at 10% — more/faster cores or a warm
    VLLM_CACHE_ROOT are the only levers.
```

Attribution rules: a stage is **network-bound** when its average receive rate is within 70 % of the
fastest rate anywhere in the run, **disk-bound** when it reads from disk faster than from the
network, **single-core CPU-bound** when CPU sits near `100 / nproc` % with the GPU idle, and **GPU
compute-bound** at ≥ 50 % GPU utilisation. Network transmit and disk write are in the JSON and the
totals line rather than the table, to keep it readable.

### 4. Measure sleep-mode warm start (instead of a cold start)

Requires a **running** server started with `VLLM_SERVER_DEV_MODE=1` **and** `--enable-sleep-mode`.

```sh
bin/spinup-profile sleepwake --base-url http://127.0.0.1:8200 --level 1 \
    --model Qwen/Qwen3.5-27B-FP8 --out ~/runs/sleepwake-l1.json
```

Emits `{"sleep_s": …, "wake_s": …, "first_completion_s": …, "level": 1, …}` plus a `steps[]` record
of every call. It pre-checks with one completion (so a wrong `--model` fails *before* you put the
server to sleep), times `POST /sleep?level=N` and `POST /wake_up`, then times the first successful
completion after wake.

**It verifies the sleep actually happened.** A server started *without* `--enable-sleep-mode` still
answers `POST /sleep` with **200 in ~19 ms** and stays awake — measured live on a Nebius H100 — so
the timings from such a run are a fiction. After the sleep call this command reads
`GET /is_sleeping` and, unless it reports `true`, wakes the server back up (so its own check never
leaves a box asleep) and exits non-zero:

```
FAILED: server did not actually sleep — was it started with --enable-sleep-mode?
        (/sleep returned 200 in 0.019s but /is_sleeping is still false; ...)
```

`--no-verify-sleep` overrides that if you need the raw numbers anyway. The `/is_sleeping` bodies are
kept in `steps[]` either way.

**GPU memory is the second, independent check.** `--verify-gpu-mem` samples
`nvidia-smi --query-gpu=memory.used` before sleep, while asleep and after wake into
`gpu_mem: {before_sleep_mib, asleep_mib, after_wake_mib, drop_pct}`, and warns via `gpu_mem_note`
if the asleep sample did not drop by at least 30 % (it re-samples once after 1.5 s first, since some
builds free asynchronously). It is **on by default when `nvidia-smi` works and `--base-url` is
local**, and auto-disables for a remote URL — sampling this box's GPUs would describe the wrong
machine. `--no-verify-gpu-mem` turns it off; passing `--verify-gpu-mem` explicitly forces it on.

For level 2 the command also tries `POST /collective_rpc {"method": "reload_weights"}` and
`POST /reset_prefix_cache`; both are version-dependent, so a failure is **reported, not fatal**.

> ⚠️ **Level 2 is NOT validated on `vllm-openai:v0.20.2`.** On that build `collective_rpc
> reload_weights` returns **500**, and the first completion after a level-2 sleep still succeeds
> instantly — i.e. the weights were apparently never discarded, so a "level 2" number there is
> measuring something other than a weight reload. The command flags this as `level2_note`. Quote
> **level 1** for anything that matters until level 2 is re-verified on a newer vLLM.

Verified reference numbers — **level 1**, 1×H100 80 GB, `Qwen/Qwen3.5-27B-FP8` (31 GB weights),
`vllm-openai:v0.20.2`, sleep state and GPU memory both confirmed:

| Measurement | Value |
|---|---|
| `sleep_s`, first cycle | 15.0 s |
| `sleep_s`, subsequent cycles | 0.65 s |
| `wake_s` | 0.86 s |
| `first_completion_s` | 0.12 s |
| **wake → serving** | **≈ 1 s** |
| GPU memory awake → asleep → awake | 77.9 GB → 3.0 GB → 77.5 GB |

Compare that against ~470 s for the cold start in the table above: sleep mode is the difference
between a cold box and a warm one, provided you can afford to hold the GPU.

---

## Stage glossary

| Stage | From → to | What it is | Lever that worked |
|---|---|---|---|
| `provision_to_ssh` * | provision.json `t_request` → `t_ssh` | cloud API call until SSH answers | warm/dark pool: a stopped baked VM reaches SSH in 30–95 s |
| `host_gpu_ready` * | `boot_seen` → `gpu_ready` | driver up, `nvidia-smi` answers | provider image with the driver preinstalled |
| `host_docker_ready` * | `gpu_ready` → `runtime_ready` | docker + nvidia container runtime usable | bake docker + nvidia-container-toolkit into the disk |
| `image_pull` * | `pull_start` → `pull_done` | pulling the ~32 GiB vLLM image | bake the image into a boot disk / AMI / host cache |
| `launch_to_container` * | `t_launch` → container `StartedAt` | `docker run` accepted until the process exists | if this is big, an implicit pull is hiding in it |
| `imports` | container start → `non-default args` | container launch + python/torch imports | image layers on local NVMe, warm page cache (~15–25 s is normal) |
| `spawn_workers` | `non-default args` → engine-core wait | config resolution + EngineCore spawn | `VLLM_WORKER_MULTIPROC_METHOD=fork` (≈ −8 s) |
| `pre_load` | engine-core wait → `Starting to load model` | handshake, backend selection, tokenizer | pre-stage the HF repo locally |
| `weight_load` | `Starting to load model` → `Model loading took …` | weight transport + placement | runai_streamer from same-region object storage (~1.4 GB/s) or local NVMe |
| `engine_init` | `Model loading took …` → `init engine … took` | everything between weights and a usable engine | see its four leaves below |
| ├ `compile` | `Start compiling function` → `torch.compile took` | torch.compile / inductor | persist `VLLM_CACHE_ROOT` (49 s → 7 s) |
| ├ `mem_profile_tail` | `torch.compile took` → `GPU KV cache size` | rest of memory profiling incl. cudagraph estimation | warm compile cache (107 s → 22 s) |
| ├ `warmup_kernels` | `GPU KV cache size` → first real capture | post-KV warm-up: mostly kernel autotune/compile | warm compile cache; `--enforce-eager` does **not** remove it |
| ├ `graph_capture` | first real capture → `Graph capturing finished` | actual CUDA graph capture | fewer capture sizes / `--max-num-seqs`, or `--enforce-eager` |
| `api_startup` | `init engine … took` → `Application startup complete` | chat-template + multimodal warm-up, routes, uvicorn | pre-stage the repo / `HF_HUB_OFFLINE=1`, `--chat-template-content-format` |
| `ready_to_health` | log ready line → first `/health` 200 | poll-interval artefact, not a stage | — |
| `first_completion` | `/health` 200 → first completion | prefill on a cold KV/prefix cache | send a warm-up request before declaring the box in service |

`*` host spans; they only appear when you give `parse` an `events.jsonl` / `provision.json`, or
when `run` measured them. They are laid out before the container timeline on their own axis.

Missing marks never silently shrink the total: a hole becomes an explicit
`unattributed_after_<span>` row, with a warning telling you to raise the log level.

---

## Known stage budget — 1×H100 80 GB, Qwen3.5-27B-FP8, naive cold start

From `examples/` (Verda 1×H100, `vllm/vllm-openai:v0.20.2`, FP8 31 GB weights from the HF hub,
262 k context, **no** persisted torch.compile cache, page cache dropped):

| Stage | Seconds | % of 470 s |
|---|---|---|
| imports | 20.99 | 4.5 % |
| spawn_workers | 40.90 | 8.7 % |
| pre_load | 15.69 | 3.3 % |
| **weight_load** | **125.75** | **26.7 %** |
| engine_init | 242.54 | 51.6 % |
| ├ engine_pre_compile | 1.42 | 0.3 % |
| ├ compile | 49.21 | 10.5 % |
| ├ mem_profile_tail | 57.49 | 12.2 % |
| ├ **warmup_kernels** | **114.94** | **24.4 %** |
| ├ graph_capture | 19.45 | 4.1 % |
| ├ engine_init_tail | 0.03 | 0.0 % |
| api_startup | 23.96 | 5.1 % |
| **container start → /health 200** | **470.06** | 100 % |

vLLM's own claims on the same run: weight load 124.86 s, torch.compile 49.21 s, memory profiling
107.80 s, graph capture 20 s, engine init 242.53 s. Note that vLLM's "graph capture 20 s" hides
the 114.94 s of kernel warm-up that precedes it — that gap is why this tool splits
`warmup_kernels` out.

On the same box family the host side added roughly: provision → SSH 89 s, driver + docker < 1 s,
image pull 142 s (≈ 700 s request → serving, matching the research's Verda baseline). With every
mitigation applied (baked disk, same-region streamed weights, persisted compile cache, `fork`)
the same recipe reached ~220–290 s end-to-end.

---

## `profile.json` schema (`spinup-profiler/profile/v1`)

| Key | Type | Meaning |
|---|---|---|
| `schema`, `generated_at`, `log` | str/float/str | format id, epoch of the parse, absolute log path |
| `name`, `run`, `image`, `vllm_args`, `port` | from `meta.json` | run identity |
| `t0`, `t0_source` | float, str | timeline origin (`meta.t_container_start` or `first_log_line`) |
| `total_s` | float | container start → first `/health` 200 (the headline number) |
| `total_profiled_s` | float | sum of every level-0 span (host prefix + container + tail) |
| `host_prefix_s` | float/null | total of the host spans, or null |
| `marks` | {str: float\|null} | absolute timeline, seconds from `t0`, one entry per pattern |
| `spans` | list | `{name, level, parent, start_s, end_s, dur_s, pct, t_start, t_end, source, axis, desc, sys}` |
| `spans[].t_start/t_end` | float | absolute epochs, so a span can be lined up with any other time series |
| `spans[].sys` | object | per-metric `{avg, peak}` for `net_rx_mbs`, `net_tx_mbs`, `disk_read_mbs`, `disk_write_mbs`, `cpu_pct`, `gpu_util_pct`, `gpu_mem_mib`, plus `samples` / `covered_s` (absent when nothing was sampled in that window) |
| `sys_summary` | object/null | `samples`, `window_s`, `net_rx_gb`, `net_tx_gb`, `disk_read_gb`, `disk_write_gb`, `peak_*`, `ifaces`, `disks`, `gpu_samples`, `cpu_count`, `hostname` |
| `cache_hit` | bool | a torch.compile cache-hit marker was seen |
| `cache_evidence` | object | `{hit_marker, store_marker, cache_dir}` — `store_marker` proves a miss |
| `weights_gib`, `kv_tokens`, `kv_mem_gib` | float/int | model facts vLLM reported |
| `claimed` | {str: float} | vLLM's own durations (`*_claimed`), for cross-checking the spans |
| `provision`, `host_events` | object/list | what was read from `provision.json` / `events.jsonl` |
| `sanity` | object/null | `{model, ok, first_completion_s}` |
| `totals` | object | `container_to_ready_s`, `launch_to_ready_s`, `provision_to_ready_s`, `first_completion_s` |
| `stats`, `warnings` | object, list | log line counts; everything the parser could not do |
| `span_<name>` | float | flat duration per span, e.g. `span_weight_load`, `span_compile` |
| `<mark>` | float | flat mark per mark name, e.g. `weights_done`, `ready` |

The flat `span_*` / mark keys are deliberately reference-compatible with the research harness's
`stages-*.json`, so old and new outputs diff directly.

### Input files `parse` understands

* `meta.json` — needs `t_container_start` (timeline origin) and `t_ready` (total); `t_launch`,
  `t_sanity_ok`, `name`, `run`, `image`, `vllm_args`, `port` are used when present. `run` writes it.
* `events.jsonl` — one JSON object per line, `{"ev": "<name>", "t": <epoch>}`. Recognised names:
  `boot_seen`, `gpu_ready`, `docker_ready`, `runtime_ready`, `pull_start`, `pull_done`. Compatible
  with the research harness's `box_bootstrap.sh` output.
* `provision.json` — `{"t_request": <epoch>, "t_ssh": <epoch>, "provider": "verda", "region": …,
  "instance_type": …}`. `t_requested`/`t_provision_start`/`t_create`/`t_start` and
  `t_ssh_ready`/`t_ssh_up`/`t_ssh_ok` are accepted as aliases. Write it from whatever provisions
  the box.
* `system.jsonl` — 1 Hz host counters (`--system-jsonl` to point elsewhere). One JSON object per
  line, **counters cumulative**, rates derived at parse time:

  ```json
  {"t": 1786983364.71, "meta": {"cpu_count": 32, "hostname": "gpu-1", "interval_s": 1.0}}
  {"t": 1786983365.71,
   "net":  {"eth0": {"rx_bytes": 12345, "tx_bytes": 678}},
   "disk": {"nvme0n1": {"read_bytes": 4096, "write_bytes": 8192}},
   "cpu_busy_pct": 12.5,
   "gpu": [{"util_pct": 0, "mem_mib": 3}]}
  ```

  Sources are `/proc/net/dev` (skipping `lo`), `/proc/diskstats` (sectors × 512, skipping
  `loop*`/`ram*`/`dm-*`/`zram*`), `/proc/stat` (busy/total delta, so the first line has no
  `cpu_busy_pct`) and `nvidia-smi --query-gpu=utilization.gpu,memory.used` every 2nd tick — absent
  `nvidia-smi` simply drops the `gpu` key. When **aggregating**, `parse` additionally excludes
  virtual interfaces (`veth*`, `docker*`, `br-*`, `tun*`, `wg*`, `tailscale*`, …) and partitions
  whose parent disk is present, because both would double-count bytes the physical device already
  reported. The raw per-device numbers stay in the file.

---

## Log-pattern caveat

Everything is derived from `docker logs -t` text. **Verified against `vllm/vllm-openai:v0.20.x`
with `VLLM_LOGGING_LEVEL=DEBUG`** (and `VLLM_CONFIGURE_LOGGING=1`, `NO_COLOR=1`). Other versions
move log lines; if a mark disappears the affected span turns into an `unattributed_*` row plus a
warning rather than a wrong number. The DEBUG-only marks are flagged below — at INFO level you
still get imports / weight_load / engine_init / api_startup, but not the compile split.

| Mark | Pattern (regex, abridged) | Level | Note |
|---|---|---|---|
| `api_args` | `non-default args` | INFO | first occurrence |
| `engine_core_up` | `Waiting for init message from front-end` | **DEBUG** | first |
| `load_start` | `Starting to load model` | INFO | first |
| `stream_start` | `Loading safetensors using Runai Model Streamer: 0%` \| `Loading weights on cuda` \| `Loading pt checkpoint` \| `Loading safetensors checkpoint` | INFO/DEBUG | first; informational (start of the actual transfer) |
| `weights_done` | `Model loading took (…) GiB memory and (…) seconds` | INFO | last; gives `weights_gib` + claimed seconds |
| `compile_start` | `Start compiling function` | **DEBUG** | first |
| `compile_done` | `torch\.compile took (…) s in total` | INFO | last |
| `compile_cache_hit` | `Directly load the compiled graph` \| `load compiled artifact` \| `hit the cache` | **DEBUG** | sets `cache_hit` |
| `compile_cache_store` | `Store the \d+-th graph for compile range` | **DEBUG** | proves a cache **miss** |
| `compile_cache_dir` | `Using cache directory: (…) for vLLM's torch\.compile` | INFO | reported so you can check the cache key |
| `mem_profile` | `Memory profiling takes (…) seconds` | **DEBUG** | last |
| `kv_size` | `GPU KV cache size: ([\d,]+) tokens` | INFO | last; gives `kv_tokens` |
| `kv_mem` | `Available KV cache memory: (…) GiB` | INFO | GiB, not seconds |
| `graphs_start` | *derived*: first `Capturing a cudagraph on` **after** `kv_size` | **DEBUG** | captures also happen during profiling, so the first one overall is the wrong mark |
| `graphs_done` | `Graph capturing finished in (\d+) secs` | INFO | last |
| `init_engine_done` | `init engine \(profile, create kv cache, warmup model\) took (…) s` | INFO | last |
| `server_listening` | `Starting vLLM server on http` | INFO | informational |
| `ready` | `Application startup complete` | INFO | last |

Patterns live in `spinup_profiler/patterns.py`; the span/leaf/hint model lives in
`spinup_profiler/stages.py`. Extend those two files — the names above are contract.

---

## Tests

```sh
python3 tests/test_parse.py
python3 tests/test_sleepwake.py
python3 tests/test_system.py
```

All three are plain python, no pytest, and exit non-zero on failure.

`test_system.py` writes a synthetic `system.jsonl` whose phases line up with the real example run's
timeline (250 MB/s of network during `weight_load`, one busy core during `imports`, 90 % GPU during
`warmup_kernels`, …), then asserts the derived per-span rates, that virtual interfaces and
partitions are excluded from the aggregate, that the IO table renders, and that each stage gets the
right bottleneck attribution. It also runs the real sampler for a moment to prove it writes what the
reader expects. No GPU needed.

`test_sleepwake.py` stands up three throwaway HTTP servers in-process and asserts the sleep
verification behaves: a server that answers `/sleep` 200 without sleeping must fail with an error
naming `--enable-sleep-mode` (and must attempt the recovery wake), a server that really sleeps must
report all four timings, and `--no-verify-sleep` must override the check. No GPU needed.

`test_parse.py` parses `examples/docker.log` (a real
`docker logs -t` capture) and asserts the headline numbers against the field-verified harness
output for the same log: `total_s ≈ 470.06`, `weight_load ≈ 125.7`, the rest of the level-0
partition and every mark within 0.1 s, compile/graph spans present, `cache_hit` false with the
store-marker evidence, and that both partitions sum to their parents. It also checks the committed
`examples/profile.json` is still what the parser produces — regenerate it with:

```sh
python3 -m spinup_profiler parse examples/docker.log --meta examples/meta.json --out examples/profile.json --quiet
python3 -m spinup_profiler report examples/profile.json --out examples/report.txt
```

---

## Layout

```
spinup_profiler/
  cli.py         argparse + dispatch          patterns.py   log regexes (field-verified)
  runner.py      `run`  — instrumented docker run, health poll, sanity, capture
  parse.py       `parse` — marks -> spans -> profile.json
  report.py      `report` — ASCII waterfall, top-3, BOTTLENECK line, hints
  sleepwake.py   `sleepwake` — sleep/wake/first-completion timing
  stages.py      the stage model: spans, leaves, descriptions, hints, attribution rules
  sysmetrics.py  system.jsonl: the 1 Hz sampler AND the rate maths that reads it back
  httpc.py       urllib wrapper that never raises
  shell.py       subprocess + nvidia-smi helpers shared by run/sleepwake
bin/spinup-profile      shim: PYTHONPATH + python3 -m spinup_profiler
examples/               real docker.log + meta.json + generated profile.json + report.txt
tests/test_parse.py     parser regression test against the example
tests/test_sleepwake.py sleep-verification test against in-process fake servers
tests/test_system.py    per-stage system metrics against a synthetic system.jsonl
```

## Gotchas

* **DEBUG logging is not optional** for the compile/warm-up/capture split. `run` injects it;
  if you capture logs yourself, set `VLLM_CONFIGURE_LOGGING=1 VLLM_LOGGING_LEVEL=DEBUG`.
* **`docker logs -t`** is required — without the RFC3339 timestamp prefix there is no timeline and
  `parse` refuses the file.
* **The torch.compile cache key includes the `--model` path string.** A cache warmed with an HF id
  will not be hit by a run using `/models/...` or `s3://...`. This is the single most common reason
  a "persisted cache" does nothing.
* **`--drop-caches` needs passwordless sudo.** Without it the run is warm and the report says so.
* `run` leaves the container running on purpose. `--teardown` removes it.
* `sleepwake` talks to endpoints that only exist under `VLLM_SERVER_DEV_MODE=1`, and needs
  `--enable-sleep-mode` on top of that — **without it `/sleep` returns 200 and does nothing**,
  which is exactly the false positive the `/is_sleeping` verification now catches.
* **Sleep level 2 is unvalidated on v0.20.2** (`reload_weights` → 500, weights apparently not
  discarded). Level 1 is the number to quote.
* **`cpu_busy_pct` is over all cores.** On a 32-core box one saturated core reads as ~3 %, which is
  why the single-core hint compares against `100 / nproc` rather than looking for 100 %.
* **System metrics are host-wide, not container-scoped.** Anything else running on the box lands in
  the same counters. On a dedicated GPU box during a spin-up that is what you want; on a shared box,
  read the per-device breakdown in `system.jsonl` before trusting the attribution.

Research background: "Minimizing vLLM Spin-Up Time" (July 2026) and the cross-cloud spin-up
benchmark (August 2026). Ticket: **IMP-2773**.
