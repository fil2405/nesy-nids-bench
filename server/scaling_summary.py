"""Tabella del test di scaling: python server/scaling_summary.py results/server/scaling/<config>"""
import glob
import json
import re
import sys
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    base = Path(sys.argv[1])
    levels = sorted((int(re.sub(r"\D", "", d.name)), d) for d in base.glob("P*") if d.is_dir())
    rows, wall1, fit1 = [], None, None
    for p, d in levels:
        watch = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{d}/watch_*.json"))]
        runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{d}/results/*_seed*.json"))]
        if not watch:
            continue
        ok = sum(w["exit_code"] == 0 for w in watch)
        wall = max(w["end"] for w in watch) - min(w["start"] for w in watch)
        fit = mean([r["timings"]["fit_s"] for r in runs])
        epochs = mean([(r.get("training") or {}).get("epochs_run") for r in runs])
        if p == 1:
            wall1, fit1 = wall, fit
        rows.append({
            "P": p, "ok": f"{ok}/{len(watch)}", "wall_s": wall,
            "speedup": p * wall1 / wall if wall1 and ok == len(watch) else None,
            "fit_s": fit, "fit_vs_P1": fit / fit1 if fit and fit1 else None, "epochs": epochs,
            "rss_max_gb": max(w["peak_rss_gb"] for w in watch),
            "rss_sum_gb": sum(w["peak_rss_gb"] for w in watch),
            "cores": sum(w["avg_cores"] or 0 for w in watch),
            "gpu_util": mean([w["gpu_util_mean"] for w in watch]),
            "gpu_mem_mib": max((w["gpu_mem_max_mib"] or 0) for w in watch),
        })
    head = ["P", "ok", "wall_s", "speedup", "fit_s", "fit_vs_P1", "epochs",
            "rss_max_gb", "rss_sum_gb", "cores", "gpu_util", "gpu_mem_mib"]
    lines = ["\t".join(head)] + ["\t".join(
        r[k] if isinstance(r[k], str) else (str(r[k]) if k == "P" else fmt(r[k], 2))
        for k in head) for r in rows]
    (base / "summary.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nspeedup = P * wall(P=1) / wall(P): vale solo se le epoche per processo sono simili (colonna epochs).")


if __name__ == "__main__":
    main()
