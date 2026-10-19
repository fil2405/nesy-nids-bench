"""Size of CIC-DDoS2019: rows per file and per label, read speed (plan M5, not a result).

    python analysis/measure_ddos2019_scan.py --source <dir or zips> --workdir <dir> --out scan.txt

Only the Label column is read, in blocks.
"""
import argparse
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

CHUNK = 500_000


class Report:
    def __init__(self, path):
        self.file = open(path, "w")

    def __call__(self, line=""):
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()


def unpack(source: Path, workdir: Path, say) -> None:
    archives = [source] if source.suffix == ".zip" else sorted(source.rglob("*.zip"))
    while archives:
        archive = archives.pop(0)
        target = workdir / archive.stem
        t0 = time.time()
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        say(f"unzip {archive.name}: {size / 1e6:.0f} MB in {time.time() - t0:.0f} s")
        archives += sorted(target.rglob("*.zip"))


def scan_file(path: Path):
    header = pd.read_csv(path, nrows=0, encoding="latin-1").columns
    label = next(c for c in header if c.strip().lower() == "label")
    counts = Counter()
    for chunk in pd.read_csv(path, usecols=[label], chunksize=CHUNK, encoding="latin-1"):
        counts.update(chunk[label].astype(str).str.strip().value_counts().to_dict())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--out", default="ddos2019_scan.txt")
    args = parser.parse_args()
    say = Report(args.out)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    say(subprocess.run(["df", "-h", str(workdir), args.source], capture_output=True, text=True).stdout)
    unpack(Path(args.source), workdir, say)
    say(subprocess.run(["df", "-h", str(workdir)], capture_output=True, text=True).stdout)

    total, per_label = 0, Counter()
    for path in sorted(workdir.rglob("*.csv")):
        mb = path.stat().st_size / 1e6
        t0 = time.time()
        counts = scan_file(path)
        dt = time.time() - t0
        say(f"{path.relative_to(workdir)}: {sum(counts.values())} rows, {mb:.0f} MB, {mb / dt:.0f} MB/s")
        for label, n in sorted(counts.items()):
            say(f"    {label}: {n}")
        total += sum(counts.values())
        per_label.update(counts)
    say(f"\nTOTAL rows: {total}")
    for label, n in per_label.most_common():
        say(f"  {label}: {n}")
    free = shutil.disk_usage(workdir).free / 1e9
    say(f"free disk: {free:.1f} GB")


if __name__ == "__main__":
    main()
