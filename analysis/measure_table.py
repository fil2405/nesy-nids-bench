"""Table of wall time, peak RAM and GPU use from the measurement logs.

    python analysis/measure_table.py ~/tmp_results/measure
"""
import re
import sys
from pathlib import Path


def parse_time(path: Path) -> dict:
    text = path.read_text() if path.exists() else ""
    exit_code = re.search(r"Exit status: (\d+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    wall = re.search(r"Elapsed \(wall clock\) time.*: ([\d:.]+)$", text, re.M)
    seconds = None
    if wall:
        seconds = 0.0
        for part in wall.group(1).split(":"):
            seconds = seconds * 60 + float(part)
    return {
        "exit": int(exit_code.group(1)) if exit_code else None,
        "ram_gb": int(rss.group(1)) / 1e6 if rss else None,
        "wall_s": seconds,
    }


def parse_dmon(path: Path) -> dict:
    if not path.exists():
        return {}
    columns, rows = None, []
    for line in path.read_text().splitlines():
        parts = line.split()
        if line.startswith("#"):
            if "sm" in parts:
                columns = parts[1:]
            continue
        if columns and len(parts) == len(columns):
            rows.append(dict(zip(columns, parts)))
    def values(key):
        return [float(r[key]) for r in rows if r.get(key, "-") not in ("-", "")]
    sm, fb = values("sm"), values("fb")
    return {
        "sm_mean": sum(sm) / len(sm) if sm else None,
        "sm_max": max(sm) if sm else None,
        "fb_max_mb": max(fb) if fb else None,
    }


def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    header = f"{'measure':<16}{'exit':>5}{'wall_s':>9}{'ram_gb':>8}{'gpu_sm%':>9}{'sm_max%':>9}{'vram_mb':>9}"
    print(header)
    for time_file in sorted(out.glob("*.time")):
        name = time_file.stem
        t, g = parse_time(time_file), parse_dmon(out / f"{name}.dmon")
        print(f"{name:<16}{fmt(t['exit'], 'd'):>5}{fmt(t['wall_s'], '.0f'):>9}"
              f"{fmt(t['ram_gb'], '.2f'):>8}{fmt(g.get('sm_mean'), '.0f'):>9}"
              f"{fmt(g.get('sm_max'), '.0f'):>9}{fmt(g.get('fb_max_mb'), '.0f'):>9}")


if __name__ == "__main__":
    main(sys.argv[1])
