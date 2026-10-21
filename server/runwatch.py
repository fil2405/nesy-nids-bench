"""Lancia un comando e registra tempo, CPU, RAM di picco e GPU (campionata ogni 2 s).

    python server/runwatch.py --out watch.json -- python -m nesybench.run ...
"""
import argparse
import json
import resource
import subprocess
import sys
import threading
import time


def gpu_sample():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
        rows = [[float(v) for v in line.split(",")] for line in out.strip().splitlines()]
        return max(r[0] for r in rows), sum(r[1] for r in rows)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=float, default=2.0)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error("manca il comando")

    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            s = gpu_sample()
            if s:
                samples.append(s)
            stop.wait(args.every)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    start = time.time()
    code = subprocess.call(cmd)
    end = time.time()
    stop.set()
    thread.join()

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = usage.ru_utime + usage.ru_stime
    wall = end - start
    record = {
        "cmd": cmd, "start": start, "end": end, "wall_s": wall, "exit_code": code,
        "cpu_s": cpu, "avg_cores": cpu / wall if wall else None,
        "peak_rss_gb": usage.ru_maxrss / 2**20,   # Linux: KB
        "gpu_util_mean": sum(s[0] for s in samples) / len(samples) if samples else None,
        "gpu_mem_max_mib": max((s[1] for s in samples), default=None),
        "gpu_samples": len(samples),
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    sys.exit(code)


if __name__ == "__main__":
    main()
