"""Argument parsing / dispatch for `python3 -m spinup_profiler`."""

import argparse

from . import __version__

EPILOG = """examples:
  # profile a cold spin-up on this box (vLLM args go after `--`)
  spinup-profile run --image vllm/vllm-openai:v0.20.2 --port 8200 --gpus all \\
      --shm-size 48g --drop-caches --label base-hf --run-index 2 \\
      --sanity-model Qwen/Qwen3.5-27B-FP8 \\
      -- --port 8200 --model Qwen/Qwen3.5-27B-FP8 --max-model-len 262144 \\
         --reasoning-parser qwen3 --max-num-seqs 256

  # re-parse an existing capture
  spinup-profile parse ~/runs/base-hf/run2
  spinup-profile parse examples/docker.log --meta examples/meta.json --out /tmp/p.json

  # render the waterfall + verdict
  spinup-profile report examples/profile.json

  # sleep-mode warm start against a running dev-mode server
  spinup-profile sleepwake --base-url http://127.0.0.1:8200 --level 2 \\
      --model Qwen/Qwen3.5-27B-FP8
"""


def build_parser():
    p = argparse.ArgumentParser(
        prog="spinup-profile",
        description="Stage-level vLLM spin-up profiler — find the cold-start bottleneck (IMP-2773).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version="spinup-profiler %s" % __version__)
    sub = p.add_subparsers(dest="cmd", metavar="<cmd>")

    # ---------------------------------------------------------------- run
    r = sub.add_parser("run", help="execute one instrumented spin-up on this box, then profile it",
                       epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    r.add_argument("--image", required=True, help="vLLM container image")
    r.add_argument("--port", type=int, default=8000, help="port vLLM serves on (default 8000)")
    r.add_argument("--gpus", default="all", help='--gpus value for docker run (default all, e.g. \'"device=0,1"\')')
    r.add_argument("--shm-size", default="48g", help="docker --shm-size (default 48g)")
    r.add_argument("--network", default="host", help="docker --network (default host; non-host publishes --port)")
    r.add_argument("--env", "-e", action="append", metavar="K=V",
                   help="env var for the container, repeatable (wins over the defaults)")
    r.add_argument("--volume", "-v", action="append", metavar="SRC:DST",
                   help="bind mount, repeatable (e.g. /mnt/nvme/cache:/root/.cache/vllm)")
    r.add_argument("--docker-arg", action="append", metavar="ARG",
                   help="extra raw docker-run arg, repeatable")
    r.add_argument("--no-default-env", action="store_true",
                   help="do not inject VLLM_LOGGING_LEVEL=DEBUG etc (the compile/graph marks then disappear)")
    r.add_argument("--drop-caches", action="store_true",
                   help="sync + drop the host page cache before launch (needs passwordless sudo)")
    r.add_argument("--no-pull", action="store_true",
                   help="skip the timed pre-pull; docker run will pull implicitly instead")
    r.add_argument("--timeout", type=int, default=1800, help="seconds to wait for /health (default 1800)")
    r.add_argument("--container", default="vllm-spinup", help="container name (default vllm-spinup)")
    r.add_argument("--label", default="spinup", help="experiment name -> <out>/<label>/run<N> (default spinup)")
    r.add_argument("--run-index", default="1", help="run index (default 1)")
    r.add_argument("--out", default="~/runs", help="run-dir root (default ~/runs)")
    r.add_argument("--health-host", default="127.0.0.1", help="host to poll (default 127.0.0.1)")
    r.add_argument("--base-url", default=None, help="override the polled base URL entirely")
    r.add_argument("--api-key", default=None, help="bearer token, if the server requires one")
    r.add_argument("--sanity-model", default=None,
                   help="served model name for the sanity completion (default: first /v1/models id)")
    r.add_argument("--sanity-prompt", default="hello", help='sanity prompt (default "hello")')
    r.add_argument("--sanity-max-tokens", type=int, default=8, help="sanity max_tokens (default 8)")
    r.add_argument("--sanity-timeout", type=int, default=300,
                   help="seconds to keep retrying the sanity completion (default 300)")
    r.add_argument("--no-sanity", action="store_true", help="skip the sanity completion")
    r.add_argument("--teardown", action="store_true",
                   help="docker rm -f the container at the end (default: leave it running)")
    r.add_argument("--docker-bin", default=None, help='docker command (default: docker, else "sudo -n docker")')
    r.add_argument("--width", type=int, default=44, help="waterfall bar columns (default 44)")
    r.add_argument("vllm_args", nargs=argparse.REMAINDER,
                   help="everything after `--` is passed to the container verbatim")

    # ---------------------------------------------------------------- parse
    q = sub.add_parser("parse", help="derive profile.json from a run dir or a docker.log")
    q.add_argument("target", help="run dir (docker.log + meta.json + events.jsonl) or a docker.log path")
    q.add_argument("--meta", default=None, help="meta.json (default <run-dir>/meta.json)")
    q.add_argument("--host-events", default=None, help="events.jsonl (default <run-dir>/events.jsonl)")
    q.add_argument("--provision", default=None, help="provision.json (default <run-dir>/provision.json)")
    q.add_argument("--out", default=None, help="output path (default <run-dir>/profile.json)")
    q.add_argument("--quiet", action="store_true", help="print only the output path, no report")
    q.add_argument("--width", type=int, default=44, help="waterfall bar columns (default 44)")

    # ---------------------------------------------------------------- report
    o = sub.add_parser("report", help="render the waterfall + bottleneck verdict")
    o.add_argument("profile", help="profile.json (or a run dir containing one)")
    o.add_argument("--width", type=int, default=44, help="waterfall bar columns (default 44)")
    o.add_argument("--top", type=int, default=3, help="how many stages to rank (default 3)")
    o.add_argument("--no-hints", action="store_true", help="omit the per-stage hints")
    o.add_argument("--out", default=None, help="also write the rendered report here")

    # ---------------------------------------------------------------- sleepwake
    s = sub.add_parser("sleepwake", help="measure sleep-mode warm start on a RUNNING dev-mode server")
    s.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8200")
    s.add_argument("--level", type=int, choices=(1, 2), default=1, help="sleep level (default 1)")
    s.add_argument("--model", required=True, help="served model name for the completions")
    s.add_argument("--prompt", default="hello", help='prompt (default "hello")')
    s.add_argument("--max-tokens", type=int, default=8, help="max_tokens (default 8)")
    s.add_argument("--timeout", type=int, default=600, help="per-call timeout seconds (default 600)")
    s.add_argument("--first-completion-timeout", type=int, default=600,
                   help="seconds to keep retrying the post-wake completion (default 600)")
    s.add_argument("--hold", type=float, default=0.0, help="seconds to stay asleep before waking (default 0)")
    s.add_argument("--api-key", default=None, help="bearer token, if the server requires one")
    s.add_argument("--no-precheck", action="store_true", help="skip the pre-sleep completion check")
    s.add_argument("--verify-sleep", dest="verify_sleep", action="store_true", default=True,
                   help="check GET /is_sleeping after the sleep call and fail if the server did "
                        "not actually sleep (default on)")
    s.add_argument("--no-verify-sleep", dest="verify_sleep", action="store_false",
                   help="report the timings even if /is_sleeping says the sleep was a no-op")
    s.add_argument("--verify-gpu-mem", dest="verify_gpu_mem", action="store_true", default=None,
                   help="sample nvidia-smi memory.used before sleep / asleep / after wake "
                        "(default: on when nvidia-smi exists and --base-url is local)")
    s.add_argument("--no-verify-gpu-mem", dest="verify_gpu_mem", action="store_false",
                   help="skip the GPU-memory sampling")
    s.add_argument("--out", default=None, help="write the JSON result here as well")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 2
    if args.cmd == "run":
        from .runner import cmd_run
        args.vllm_args = list(args.vllm_args or [])
        if args.vllm_args and args.vllm_args[0] == "--":  # argparse.REMAINDER keeps the separator
            args.vllm_args.pop(0)
        return cmd_run(args)
    if args.cmd == "parse":
        from .parse import cmd_parse
        return cmd_parse(args)
    if args.cmd == "report":
        from .report import cmd_report
        return cmd_report(args)
    if args.cmd == "sleepwake":
        from .sleepwake import cmd_sleepwake
        return cmd_sleepwake(args)
    parser.print_help()
    return 2
