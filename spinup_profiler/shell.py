"""Subprocess + nvidia-smi helpers shared by `run` and `sleepwake`."""

import subprocess


def sh(cmd, timeout=60):
    """Run a command, never raise. Returns (rc, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "", "timeout after %ss: %s" % (timeout, " ".join(cmd))
    except FileNotFoundError as e:
        return 127, "", str(e)


def have_nvidia_smi():
    return sh(["nvidia-smi", "-L"], timeout=30)[0] == 0


def gpu_mem_used_mib():
    """Total GPU memory in use across all LOCAL GPUs, in MiB (None if unavailable)."""
    rc, out, _err = sh(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], timeout=30)
    if rc != 0:
        return None
    vals = [int(tok) for tok in out.replace(",", " ").split() if tok.isdigit()]
    return sum(vals) if vals else None
