"""Tables of the runs on the saved models: post-check (E12), perturbations (E10), explanations (E17), latency (E12),
operating point fixed on validation and cost curves.

    python analysis/saved_tables.py postcheck --dataset nslkdd [--dir results/saved_eval] [--tex out.tex]
    python analysis/saved_tables.py e10 --dataset nslkdd [--tex out.tex]
    python analysis/saved_tables.py e17 --dataset nslkdd [--mode annotate]
    python analysis/saved_tables.py e12 --dataset nslkdd [--dir results/latency]
    python analysis/saved_tables.py latency [--dir results/latency] [--tex thesis/back/appendixC_latency.tex]
    python analysis/saved_tables.py threshold --dataset nslkdd [--dir results/operating]
    python analysis/saved_tables.py cost --dataset nslkdd [--fig thesis/Images/cost_nslkdd.pdf]

Reads the saved_eval reports whose run_id starts with <dataset>_ (one per arm, every seed in it).
Differences are paired by seed, no Holm (sensitivity checks).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nesybench.compare import paired  # noqa: E402

ORDER = ("mlp_noweights", "mlp_weights", "cnn", "ltn_sup", "ltn_full", "rf", "xgb", "rules")


def arm_of(run_id, dataset):
    rest = run_id[len(dataset) + 1:]
    return rest[:-len("_tuned")] if rest.endswith("_tuned") else rest


def _rank(arm):
    return (ORDER.index(arm), arm) if arm in ORDER else (len(ORDER), arm)


def load_reports(folder, dataset):
    """{arm: report} for the saved_eval reports of `dataset`."""
    out = {}
    for path in sorted(Path(folder).glob("*.json")):
        rep = json.loads(path.read_text())
        if "per_seed" in rep and rep.get("run_id", "").startswith(dataset + "_"):
            out[arm_of(rep["run_id"], dataset)] = rep
    return dict(sorted(out.items(), key=lambda kv: _rank(kv[0])))


def seeds(rep):
    return sorted(rep["per_seed"], key=int)


def values(rep, get):
    return [get(rep["per_seed"][s]) for s in seeds(rep)]


def ms(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def postcheck_rows(reports):
    """(arm, n, model, annotate, enforce, diff enforce - model, fixed, broken)."""
    out = []
    for arm, rep in reports.items():
        base = values(rep, lambda r: r["macro_f1"])
        enf = values(rep, lambda r: r["postcheck"]["enforce"]["macro_f1"])
        out.append((arm, len(base), ms(base),
                    ms(values(rep, lambda r: r["postcheck"].get("annotate", {}).get("macro_f1", np.nan))), ms(enf),
                    paired(base, enf) if len(base) > 1 else None,
                    ms(values(rep, lambda r: r["postcheck"]["enforce"].get("fixed", np.nan))),
                    ms(values(rep, lambda r: r["postcheck"]["enforce"].get("broken", np.nan)))))
    return out


def rule_rows(reports):
    """Per axiom: rows fired and precision on the part, from the first report that has them."""
    for rep in reports.values():
        first = rep["per_seed"][seeds(rep)[0]]
        if "rules" in first:
            return [(ident, r["fired"], r["precision"]) for ident, r in first["rules"].items()]
    return []


def perturbations(reports):
    return sorted({n for rep in reports.values() for r in rep["per_seed"].values()
                   for n in r.get("perturbed", {})})


def e10_rows(reports, names):
    """(arm, clean, {name: (mean std, paired diff from clean)})."""
    out = []
    for arm, rep in reports.items():
        clean = values(rep, lambda r: r["macro_f1"])
        cells = {}
        for name in names:
            got = values(rep, lambda r: r.get("perturbed", {}).get(name, {}).get("macro_f1", np.nan))
            ok = [i for i, v in enumerate(got) if np.isfinite(v)]
            diff = paired([clean[i] for i in ok], [got[i] for i in ok]) if len(ok) > 1 else None
            cells[name] = (ms(got), diff)
        out.append((arm, ms(clean), cells))
    return out


def e17_rows(reports, mode):
    keys = ("n_alerts", "coverage", "fidelity", "conciseness")
    return [(arm, *(ms(values(rep, lambda r, k=k: r["postcheck"][mode]["explanations"][k])) for k in keys))
            for arm, rep in reports.items()]


def latency_rows(folder, dataset, q="p50_ms"):
    """(arm, {batch: (model, annotate, enforce)}) from the latency reports."""
    out = []
    for path in sorted(Path(folder).glob(f"{dataset}_*.json")):
        rep = json.loads(path.read_text())
        pipe = rep.get("pipeline")
        src = pipe if pipe else {"model": rep["batches"]}
        batches = sorted(src["model"], key=int)
        out.append((arm_of(path.stem, dataset), {
            b: tuple(src[k][b][q] if k in src else float("nan") for k in ("model", "annotate", "enforce"))
            for b in batches}))
    return sorted(out, key=lambda r: _rank(r[0]))


def threshold_rows(reports):
    """(arm, budget, test TPR, test alerts per 100k benign, validation alerts per 100k, unseen detected)."""
    out = []
    for arm, rep in reports.items():
        for b in rep["budgets"]:
            key = f"{b:g}"
            get = lambda k: values(rep, lambda r: r["threshold"][key].get(k, np.nan))  # noqa: E731
            out.append((arm, b, ms(get("test_tpr")), ms(np.asarray(get("test_fpr")) * 1e5),
                        ms(np.asarray(get("val_fpr")) * 1e5), ms(get("unseen_detected"))))
    return out


def cost_curves(reports):
    """{arm: mean envelope over seeds}, the grid, {arm: area}."""
    grid = np.asarray(next(iter(reports.values()))["grid"])
    curves = {arm: np.mean(values(rep, lambda r: r["cost"]["envelope"]), axis=0) for arm, rep in reports.items()}
    areas = {arm: ms(values(rep, lambda r: r["cost"]["area"])) for arm, rep in reports.items()}
    return grid, curves, areas


def best_ranges(grid, curves):
    """[(pc from, pc to, arm)] of the arm with the lowest mean cost, ties to the first in order."""
    arms = list(curves)
    best = [arms[int(np.argmin([curves[a][i] for a in arms]))] for i in range(len(grid))]
    out = []
    for i, arm in enumerate(best):
        if out and out[-1][2] == arm:
            out[-1][1] = grid[i]
        else:
            out.append([grid[i], grid[i], arm])
    return [tuple(r) for r in out]


LABELS = {"mlp_noweights": "MLP-CE, no weights", "mlp_weights": "MLP-CE, weights", "cnn": "1D-CNN",
          "ltn_sup": "LTN-sup", "ltn_full": "LTN-full", "rf": "Random Forest", "xgb": "XGBoost", "rules": "Rule-only"}


def plot_costs(path, grid, curves, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for arm, c in curves.items():
        ax.plot(grid, c, label=LABELS.get(arm, arm), linewidth=1.2,
                linestyle="--" if arm in ("rf", "xgb", "rules") else "-")
    ax.set_xlabel("Probability cost of an attack, PC(+)")
    ax.set_ylabel("Normalised expected cost")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


NAMES = {"nslkdd": "NSL-KDD", "cicids2017": "CIC-IDS2017", "cicddos2019": "CIC-DDoS2019"}


def latency_detail(folder):
    """(dataset, arm, p50, p95, p99 at batch 1, rows/s at the largest batch, MB of the saved files), model alone."""
    out = []
    for ds in NAMES:
        for path in sorted(Path(folder).glob(f"{ds}_*.json")):
            rep = json.loads(path.read_text())
            b = rep["batches"]
            one, big = b["1"], b[max(b, key=int)]
            out.append((ds, arm_of(path.stem, ds), one["p50_ms"], one["p95_ms"], one["p99_ms"],
                        big["rows_per_second"], sum(rep.get("size_bytes", {}).values()) / 2**20))
    return sorted(out, key=lambda r: (list(NAMES).index(r[0]), _rank(r[1])))


def latency_tex(rows):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize",
             r"\caption{E12, the model alone on the CPU of L1, saved model of seed 0: latency percentiles "
             r"with one row per call, rows per second with 1024 rows per call, and the size of the saved files.}",
             r"\label{tab:app-latency}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrrrrr@{}}",
             r"\toprule", r"Dataset & Arm & p50 ms & p95 ms & p99 ms & rows/s & MB \\", r"\midrule"]
    last = None
    for ds, arm, p50, p95, p99, rps, mb in rows:
        rate = f"{rps:,.0f}".replace(",", "\\,")
        lines.append(f"{NAMES[ds] if ds != last else ''} & {LABELS.get(arm, arm)} & {p50:.2f} & {p95:.2f} & {p99:.2f} "
                     f"& {rate} & {mb:.2f} \\\\")
        last = ds
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def f(m_s, d=4):
    m, s = m_s
    return "--" if not np.isfinite(m) else f"{m:.{d}f} +- {s:.{d}f}"


def tex_f(m_s, d=3):
    m, s = m_s
    return "--" if not np.isfinite(m) else f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def tex_diff(r):
    return "--" if r is None else f"{r.median:+.4f} [{r.low:+.4f}, {r.high:+.4f}]"


def write_tex(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = "l" + "r" * (len(header) - 1)
    lines = [rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("table", choices=["postcheck", "e10", "e17", "e12", "latency", "threshold", "cost"])
    ap.add_argument("--dataset", help="all but latency")
    ap.add_argument("--dir", help="default results/saved_eval, e12: results/latency, "
                                  "threshold and cost: results/operating")
    ap.add_argument("--mode", default="annotate", help="e17: post-check mode")
    ap.add_argument("--tex")
    ap.add_argument("--fig", help="cost: figure file (.pdf)")
    args = ap.parse_args()

    if args.table == "latency":
        rows = latency_detail(args.dir or "results/latency")
        if not rows:
            raise SystemExit("no latency reports")
        for ds, arm, p50, p95, p99, rps, mb in rows:
            print(f"  {ds:12} {arm:14} p50 {p50:7.2f}  p95 {p95:7.2f}  p99 {p99:7.2f}  rows/s {rps:12,.0f}  {mb:8.2f} MB")
        if args.tex:
            Path(args.tex).write_text("% generated by analysis/saved_tables.py latency, do not edit by hand\n\n"
                                      + latency_tex(rows))
            print(args.tex)
        return
    if not args.dataset:
        raise SystemExit(f"{args.table} needs --dataset")

    if args.table in ("threshold", "cost"):
        reports = load_reports(args.dir or "results/operating", args.dataset)
        if not reports:
            raise SystemExit(f"no operating reports for {args.dataset}")
        if args.table == "threshold":
            rows = threshold_rows(reports)
            print(f"{args.dataset}: threshold set on validation at each FPR budget, applied to test; "
                  "alerts per 100k benign")
            for arm, b, tpr, fa, fa_val, unseen in rows:
                tail = "" if not np.isfinite(unseen[0]) else f"  unseen {f(unseen)}"
                print(f"  {arm:14} FPR {b:g}  test TPR {f(tpr)}  test alerts {f(fa, 1)}  val alerts {f(fa_val, 1)}{tail}")
            if args.tex:
                write_tex(args.tex, ["Arm", "Budget", "Test TPR", "Test alerts / 100k", "Unseen detected"],
                          [[a.replace("_", "-"), f"{b * 1e5:.0f}", tex_f(t), tex_f(fa, 0), tex_f(u)]
                           for a, b, t, fa, fv, u in rows])
        else:
            grid, curves, areas = cost_curves(reports)
            marks = (0.1, 0.25, 0.5, 0.75, 0.9)
            idx = [int(np.argmin(np.abs(grid - m))) for m in marks]
            print(f"{args.dataset}: normalised expected cost on test (mean envelope over seeds), area under it")
            print(f"  {'':14} " + "  ".join(f"PC {m:4.2f}" for m in marks) + "  area")
            for arm, c in curves.items():
                print(f"  {arm:14} " + "  ".join(f"{c[i]:7.4f}" for i in idx) + f"  {f(areas[arm])}")
            print("lowest cost:", ", ".join(f"PC {lo:.2f}-{hi:.2f} {arm}" for lo, hi, arm in best_ranges(grid, curves)))
            if args.tex:
                write_tex(args.tex, ["Arm", *[f"PC {m:g}" for m in marks], "Area"],
                          [[a.replace("_", "-"), *[f"{c[i]:.4f}" for i in idx], tex_f(areas[a], 4)]
                           for a, c in curves.items()])
            if args.fig:
                plot_costs(args.fig, grid, curves, args.dataset)
        return

    if args.table == "e12":
        rows = latency_rows(args.dir or "results/latency", args.dataset)
        if not rows:
            raise SystemExit(f"no latency reports for {args.dataset}")
        print(f"{args.dataset} latency p50 ms (model / + annotate / + enforce), one process, CPU, seed 0")
        for arm, by_batch in rows:
            print(f"  {arm:14} " + "  ".join(f"b{b}: " + " / ".join(f"{v:.2f}" for v in t)
                                              for b, t in by_batch.items()))
        if args.tex:
            batches = list(rows[0][1])
            write_tex(args.tex, ["Arm", *[f"batch {b}" for b in batches]],
                      [[arm.replace("_", "-"), *[" / ".join(f"{v:.2f}" for v in t[b]) for b in batches]]
                       for arm, t in rows])
        return

    reports = load_reports(args.dir or "results/saved_eval", args.dataset)
    if not reports:
        raise SystemExit(f"no saved_eval reports for {args.dataset}")

    if args.table == "postcheck":
        rows = postcheck_rows(reports)
        print(f"{args.dataset} macro-F1 (test): model, annotate, enforce; diff = enforce minus model")
        for arm, n, base, ann, enf, d, fixed, broken in rows:
            tail = "" if d is None else (f"  diff {d.median:+.4f} [{d.low:+.4f}, {d.high:+.4f}]"
                                         f" wins {d.wins}/{d.n} p {d.p:.4f}")
            print(f"  {arm:14} n={n:2d}  {f(base)}  ann {f(ann)}  enf {f(enf)}{tail}"
                  f"  fixed {fixed[0]:.0f} broken {broken[0]:.0f}")
        print("rules on the test part (same rows for every arm):")
        for ident, n, prec in rule_rows(reports):
            print(f"  {ident:6} fired {n:7d}  precision {prec:.4f}")
        if args.tex:
            write_tex(args.tex, ["Arm", "Model", "Enforce", "Diff [95\\% interval]", "Fixed", "Broken"],
                      [[a.replace("_", "-"), tex_f(b), tex_f(e), tex_diff(d), f"{fx[0]:.0f}", f"{br[0]:.0f}"]
                       for a, n, b, ann, e, d, fx, br in rows])
    elif args.table == "e10":
        names = perturbations(reports)
        rows = e10_rows(reports, names)
        print(f"{args.dataset} E10 macro-F1 (test), model alone; diff = perturbed minus clean")
        for arm, clean, cells in rows:
            print(f"  {arm:14} clean {f(clean)}")
            for name, (m_s, d) in cells.items():
                tail = "" if d is None else f"  diff {d.median:+.4f} [{d.low:+.4f}, {d.high:+.4f}] p {d.p:.4f}"
                print(f"  {'':14} {name:11} {f(m_s)}{tail}")
        if args.tex:
            write_tex(args.tex, ["Arm", "Clean", *names],
                      [[a.replace("_", "-"), tex_f(c), *[tex_f(cells[n][0]) for n in names]]
                       for a, c, cells in rows])
    else:
        rows = e17_rows(reports, args.mode)
        print(f"{args.dataset} E17 explanations ({args.mode}), over the alerts of each seed")
        for arm, n, cov, fid, con in rows:
            print(f"  {arm:14} alerts {n[0]:9.0f}  coverage {f(cov)}  fidelity {f(fid)}  conciseness {f(con, 3)}")
        if args.tex:
            write_tex(args.tex, ["Arm", "Coverage", "Fidelity", "Conciseness"],
                      [[a.replace("_", "-"), tex_f(c), tex_f(fi), tex_f(co, 2)] for a, n, c, fi, co in rows])


if __name__ == "__main__":
    main()
