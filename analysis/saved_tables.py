"""Tables of the runs on the saved models: post-check (E12), perturbations (E10), explanations (E17), latency (E12).

    python analysis/saved_tables.py postcheck --dataset nslkdd [--dir results/saved_eval] [--tex out.tex]
    python analysis/saved_tables.py e10 --dataset nslkdd [--tex out.tex]
    python analysis/saved_tables.py e17 --dataset nslkdd [--mode annotate]
    python analysis/saved_tables.py e12 --dataset nslkdd [--dir results/latency]

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
    ap.add_argument("table", choices=["postcheck", "e10", "e17", "e12"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dir", help="default results/saved_eval, e12: results/latency")
    ap.add_argument("--mode", default="annotate", help="e17: post-check mode")
    ap.add_argument("--tex")
    args = ap.parse_args()

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
