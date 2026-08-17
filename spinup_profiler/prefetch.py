"""Warm the OS page cache before a spin-up, so the engine's first read is not the cold one.

Measured effect (Verda 2xH100, 31 GB of FP8 shards on local NVMe): 8 jobs warm 31 GB in 8.4 s at
3.7 GB/s, weight_load then drops 62.9 s -> 12.5 s. It only pays when the disk rewards parallel
readers -- `probe` measures whether yours does.
"""

import glob
import os
import threading
import time

CHUNK = 8 << 20        # 8 MiB reads
SEGMENT = 256 << 20    # split files into 256 MiB units of work so one huge file still parallelises


def expand(spec):
    """A directory (recursive), a glob, or a file -> sorted list of regular files."""
    if os.path.isdir(spec):
        found = []
        for root, _dirs, files in os.walk(spec):
            for f in files:
                found.append(os.path.join(root, f))
        return sorted(found)
    files = []
    for m in sorted(glob.glob(spec)):
        if os.path.isdir(m):
            files += expand(m)
        elif os.path.isfile(m):
            files.append(m)
    return files


def _segments(files, segment=SEGMENT):
    work = []
    for path in files:
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size <= 0:
            continue
        offset = 0
        while offset < size:
            work.append((path, offset, min(segment, size - offset)))
            offset += segment
    return work


def warm(spec, jobs=8, chunk=CHUNK):
    """Parallel-read everything matched by `spec`. Returns a summary dict; never raises."""
    t0 = time.time()
    files = expand(spec)
    work = _segments(files)
    read = [0]
    errors = []
    lock = threading.Lock()
    index = [0]

    def worker():
        while True:
            with lock:
                if index[0] >= len(work):
                    return
                path, offset, length = work[index[0]]
                index[0] += 1
            got = 0
            try:
                with open(path, "rb") as fh:
                    fh.seek(offset)
                    while got < length:
                        b = fh.read(min(chunk, length - got))
                        if not b:
                            break
                        got += len(b)
            except Exception as e:
                with lock:
                    if len(errors) < 5:
                        errors.append("%s: %s: %s" % (path, type(e).__name__, e))
            with lock:
                read[0] += got

    n = max(1, min(int(jobs), len(work) or 1))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.time() - t0
    return {
        "spec": spec,
        "files": len(files),
        "segments": len(work),
        "jobs": n,
        "bytes": read[0],
        "gb": round(read[0] / 1e9, 3),
        "s": round(dt, 2),
        "mbs": round(read[0] / 1e6 / dt, 1) if dt > 0 else None,
        "errors": errors or None,
    }
