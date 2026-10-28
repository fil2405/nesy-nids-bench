"""E7/E8: score against the real train size, and the knowledge effect per level.

    python analysis/stress_curves.py --exp e7 --dataset nslkdd \
        [--metric macro_f1] [--part test] [--class R2L] [--tex out.tex] [--plot out.pdf]

Reads <dataset>_<arm>_tuned_<exp>_f<level> and the finals <dataset>_<arm>_tuned
(level 1). Knowledge effect: ltn_full minus ltn_sup, paired by seed, no Holm.
The finals have no split_sizes; their rows per class are estimated from the run that kept
the largest share of each class (classes at the floor give none).
With --class the plot uses that class's train rows as x (E8), else the total (E7).
--ltn-full no_A3: ltn_full read from the runs with that edit (<exp>_no_A3_f<level>), level 1
from the E3 run e3_no_A3; the other arms are unchanged.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_runs import metric_value  # noqa: E402
from summarize_runs import load_runs  # noqa: E402

from nesybench.compare import paired  # noqa: E402

ARMS = ("mlp_noweights", "mlp_weights", "rf", "xgb", "ltn_sup", "ltn_full")
RUN_ID = re.compile(r"^(?P<ds>[a-z0-9]+)_(?P<arm>" + "|".join(ARMS)
                    + r")_tuned_(?P<exp>e\d+)(?:_(?P<var>no_\w+?))?_f(?P<level>\d+(?:p\d+)?)$")


def level_value(tag):
    return float(tag.replace("p", ".")) / 100  # f5 -> 0.05, f0p1 -> 0.001


def collect(runs, dataset, exp, ltn_full=None):
    """{arm: {level: run_id}}, the finals as level 1; ltn_full: an edit tag such as no_A3."""
    found = {arm: {} for arm in ARMS}
    for run_id in runs:
        m = RUN_ID.match(run_id)
        if not m or m["ds"] != dataset or m["exp"] != exp:
            continue
        want = ltn_full if m["arm"] == "ltn_full" else None
        if m["var"] == want:
            found[m["arm"]][level_value(m["level"])] = run_id
    for arm in ARMS:
        final = f"{dataset}_{arm}_tuned"
        if arm == "ltn_full" and ltn_full:
            final = f"{final}_e3_{ltn_full}"
        if final in runs:
            found[arm][1.0] = final
    return found


def seeds_of(runs, run_id):
    return sorted(runs[run_id])


def train_rows(recs, class_index=None):
    sizes = [r.get("split_sizes", {}).get("train") for r in recs]
    if any(s is None for s in sizes):
        return float("nan")
    if class_index is None:
        return float(np.mean([sum(s) for s in sizes]))
    return float(np.mean([s[class_index] for s in sizes]))


def kept_fractions(rec, class_names):
    """Per class: share of the train part kept by train_fraction and train_class_fractions."""
    sp = rec["config"].get("split_params", {})
    out = np.full(len(class_names), float(sp.get("train_fraction", 1.0)))
    for name, frac in (sp.get("train_class_fractions") or {}).items():
        out[class_names.index(name)] *= frac
    return out, int(sp.get("min_keep_per_class", 0))


def full_sizes(runs, found, part):
    """Per class rows of the full train part, from the run that keeps the largest share of it.

    Classes held at the floor give no estimate (nan)."""
    best = None
    for by_level in found.values():
        for level, run_id in by_level.items():
            if level == 1.0:
                continue
            for seed in seeds_of(runs, run_id):
                rec = runs[run_id][seed]
                if "split_sizes" not in rec:
                    continue
                names = rec["metrics"][part]["class_names"]
                frac, floor = kept_fractions(rec, names)
                rows = np.asarray(rec["split_sizes"]["train"], dtype=float)
                est = np.where(rows > floor, rows / frac, np.nan)
                if best is None:
                    best = (frac, est)
                else:
                    take = ((frac > best[0]) & np.isfinite(est)) | ~np.isfinite(best[1])
                    best = (np.where(take, frac, best[0]), np.where(take, est, best[1]))
    return None if best is None else best[1]


def table(runs, found, part, metric, cls):
    """Rows (arm, level, n seeds, train rows, class rows, mean, std)."""
    class_idx = None
    out = []
    full = full_sizes(runs, found, part)
    for arm in ARMS:
        for level in sorted(found[arm]):
            recs = [runs[found[arm][level]][s] for s in seeds_of(runs, found[arm][level])]
            if cls and class_idx is None:
                class_idx = recs[0]["metrics"][part]["class_names"].index(cls)
            scores = [metric_value(r, part, metric) for r in recs]
            if level == 1.0:  # finals: no split_sizes, estimated
                rows = float(full.sum()) if full is not None else float("nan")
                crows = float(full[class_idx]) if full is not None and cls else float("nan")
            else:
                rows = train_rows(recs)
                crows = train_rows(recs, class_idx) if cls else float("nan")
            std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            out.append((arm, level, len(recs), rows, crows, float(np.mean(scores)), std))
    return out


def knowledge_effect(runs, found, part, metric, hi="ltn_full", lo="ltn_sup"):
    """Per level: hi minus lo over the common seeds."""
    out = []
    for level in sorted(set(found[hi]) & set(found[lo])):
        seeds = sorted(set(runs[found[hi][level]]) & set(runs[found[lo][level]]))
        if len(seeds) < 2:
            continue
        a = [metric_value(runs[found[lo][level]][s], part, metric) for s in seeds]
        b = [metric_value(runs[found[hi][level]][s], part, metric) for s in seeds]
        out.append((level, paired(a, b)))
    return out


def tex_scores(rows, metric):
    levels = sorted({r[1] for r in rows})
    lines = [r"\begin{tabular}{@{}l" + "r" * len(levels) + r"@{}}", r"\toprule",
             "Arm & " + " & ".join(f"{lv:g}" for lv in levels) + r" \\", r"\midrule"]
    for arm in ARMS:
        cells = {r[1]: r for r in rows if r[0] == arm}
        if not cells:
            continue
        lines.append(arm.replace("_", "-") + " & " + " & ".join(
            f"{cells[lv][5]:.3f} $\\pm$ {cells[lv][6]:.3f}" if lv in cells else "--"
            for lv in levels) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tex_effect(effect, metric):
    lines = [r"\begin{tabular}{@{}rrrrr@{}}", r"\toprule",
             rf"Level & median diff ({metric.replace('_', ' ')}) & 95\% interval & wins & $p$ \\",
             r"\midrule"]
    for level, r in effect:
        lines.append(f"{level:g} & {r.median:+.4f} & [{r.low:+.4f}, {r.high:+.4f}] "
                     f"& {r.wins}/{r.n} & {r.p:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def plot(rows, path, metric, x_label, by_class=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = 4 if by_class else 3
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in ARMS:
        pts = [r for r in rows if r[0] == arm and np.isfinite(r[x])]
        if pts:
            ax.errorbar([r[x] for r in pts], [r[5] for r in pts], [r[6] for r in pts],
                        marker="o", capsize=2, label=arm)
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.replace("_", " "))
    ax.legend(fontsize=7)
    fig.savefig(path, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--exp", required=True, choices=["e7", "e8"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--part", default="test", choices=["val", "test"])
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--class", dest="cls", help="class whose train rows are listed (e.g. R2L)")
    ap.add_argument("--tex", help="writes <tex>_scores.tex and <tex>_effect.tex")
    ap.add_argument("--plot")
    ap.add_argument("--ltn-full", help="edit tag of the ltn_full runs, e.g. no_A3 (data axioms out)")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    found = collect(runs, args.dataset, args.exp, args.ltn_full)
    if not any(found.values()):
        raise SystemExit(f"no {args.exp} runs for {args.dataset} in {args.results_dir}")
    rows = table(runs, found, args.part, args.metric, args.cls)
    print(f"{args.dataset} {args.exp} {args.metric} ({args.part}); finals are level 1")
    print(f"{'arm':14} {'level':>7} {'n':>3} {'train rows':>11} {'class rows':>11} {'mean':>8} {'std':>7}")
    for arm, level, n, rows_, crows, m, s in rows:
        print(f"{arm:14} {level:7g} {n:3d} {rows_:11.0f} {crows:11.0f} {m:8.4f} {s:7.4f}")
    effect = knowledge_effect(runs, found, args.part, args.metric)
    print("\nknowledge effect (ltn_full minus ltn_sup)")
    for level, r in effect:
        print(f"  level {level:g}  n={r.n}  median {r.median:+.4f} [{r.low:+.4f}, {r.high:+.4f}]"
              f"  wins {r.wins}/{r.n} ties {r.ties}  p {r.p:.4f}")
    if args.tex:
        base = Path(args.tex)
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_name(base.stem + "_scores.tex").write_text(tex_scores(rows, args.metric))
        base.with_name(base.stem + "_effect.tex").write_text(tex_effect(effect, args.metric))
    if args.plot:
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        label = f"{args.cls} train rows" if args.cls else "train rows"
        plot(rows, args.plot, args.metric, label, by_class=bool(args.cls))


if __name__ == "__main__":
    main()
