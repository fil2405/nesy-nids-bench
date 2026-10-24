"""Paired tests, Holm and Friedman/Nemenyi over the seeds of finished runs.

    python analysis/compare_runs.py --part test --metric macro_f1 \
        --pair nslkdd_mlp_noweights nslkdd_mlp_weights --pair ... \
        [--rank run1 run2 run3 ...] [--tex out.tex] [--cd-plot cd.pdf]

metric: macro_f1 | mcc | accuracy | recall:CLASS | f1:CLASS | precision:CLASS | pr_auc:CLASS
       | label_recall:LABEL | label_detected:LABEL   (fine label; detected = predicted not Normal)
Difference is B minus A. Holm over the pairs of one call.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from summarize_runs import load_runs  # noqa: E402

from nesybench.compare import friedman_nemenyi, holm, paired  # noqa: E402


def metric_value(rec, part, metric):
    m = rec["metrics"][part]
    name, _, cls = metric.partition(":")
    if name in ("label_recall", "label_detected"):  # one fine label, e.g. label_detected:DoS Slowloris
        return m["by_label"][cls][name.split("_")[1]]
    if not cls:
        return m[name]
    return m[name][m["class_names"].index(cls)]


def seed_scores(runs, run_id, part, metric, seeds):
    return [metric_value(runs[run_id][s], part, metric) for s in seeds]


def common_seeds(runs, ids):
    missing = [i for i in ids if i not in runs]
    if missing:
        raise SystemExit(f"run_id not found: {missing}")
    seeds = sorted(set.intersection(*(set(runs[i]) for i in ids)))
    if len(seeds) < 2:
        raise SystemExit(f"fewer than two common seeds for {ids}")
    return seeds


def tex_table(rows, metric):
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             rf"A & B & median diff ({metric.replace('_', ' ')}) & 95\% interval & wins & $p$ & $p_{{\mathrm{{Holm}}}}$ \\",
             r"\midrule"]
    for a, b, r, ph in rows:
        lines.append(f"{a.replace('_', '-')} & {b.replace('_', '-')} & {r.median:+.4f} & "
                     f"[{r.low:+.4f}, {r.high:+.4f}] & {r.wins}/{r.n} & {r.p:.4f} & {ph:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def top_limit(k):
    return 0.7 - 0.12 * k - 0.3


def cd_plot(rank, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(rank.mean_ranks)
    fig, ax = plt.subplots(figsize=(6, 1.6 + 0.3 * len(order)))
    k = len(order)
    ax.set_xlim(k + 0.5, 0.5)
    ax.set_ylim(top_limit(len(order)), 1)
    ax.axis("off")
    ax.hlines(0.8, 1, k, color="black", lw=1)
    for r in range(1, k + 1):
        ax.text(r, 0.86, str(r), ha="center", fontsize=8)
    ranks = [rank.mean_ranks[i] for i in order]
    bars, last_end = [], -1
    for i in range(k):
        j = max(m for m in range(i, k) if ranks[m] - ranks[i] < rank.cd)
        if j > i and j > last_end:   # a group not significantly different, not inside the previous
            bars.append((ranks[i], ranks[j]))
            last_end = j
    for b, (lo, hi) in enumerate(bars):
        ax.hlines(0.74 - 0.04 * b, lo, hi, color="black", lw=3)
    top = 0.74 - 0.04 * len(bars)
    for j, i in enumerate(order):
        x = rank.mean_ranks[i]
        y = top - 0.06 - 0.12 * j
        ax.plot([x, x], [0.8, y], color="black", lw=0.7)
        ax.text(x, y - 0.03, f"{rank.names[i]} ({x:.2f})", ha="center", va="top", fontsize=7)
    ax.hlines(0.95, 1, 1 + rank.cd, color="red", lw=2)
    ax.text(1 + rank.cd / 2, 0.98, f"CD {rank.cd:.2f}", ha="center", fontsize=7)
    fig.savefig(path, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--part", default="test", choices=["val", "test"])
    ap.add_argument("--metric", default="macro_f1")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("--rank", nargs="+", metavar="RUN_ID")
    ap.add_argument("--tex")
    ap.add_argument("--cd-plot")
    args = ap.parse_args()
    runs = load_runs(args.results_dir)

    results = []
    for a, b in args.pair:
        seeds = common_seeds(runs, [a, b])
        results.append((a, b, paired(seed_scores(runs, a, args.part, args.metric, seeds),
                                     seed_scores(runs, b, args.part, args.metric, seeds))))
    adjusted = holm([r.p for _, _, r in results]) if results else []
    rows = [(a, b, r, ph) for (a, b, r), ph in zip(results, adjusted)]
    for a, b, r, ph in rows:
        print(f"{a} -> {b}  {args.metric} ({args.part})  n={r.n}  median diff {r.median:+.4f} "
              f"[{r.low:+.4f}, {r.high:+.4f}]  wins {r.wins}/{r.n} ties {r.ties}  "
              f"p {r.p:.4f}  Holm {ph:.4f}")
    if args.tex and rows:
        Path(args.tex).write_text(tex_table(rows, args.metric))

    if args.rank:
        seeds = common_seeds(runs, args.rank)
        rank = friedman_nemenyi({i: seed_scores(runs, i, args.part, args.metric, seeds)
                                 for i in args.rank})
        print(f"Friedman p {rank.friedman_p:.4f}  n={rank.n}  Nemenyi CD {rank.cd:.2f}")
        for name, r in sorted(zip(rank.names, rank.mean_ranks), key=lambda t: t[1]):
            print(f"  {r:.2f}  {name}")
        if args.cd_plot:
            cd_plot(rank, args.cd_plot)


if __name__ == "__main__":
    main()
