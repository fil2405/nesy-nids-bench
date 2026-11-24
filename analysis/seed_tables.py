"""Per-seed test macro-F1 and the secondary metrics of the final runs, as LaTeX (Appendix C).

    python analysis/seed_tables.py [--results-dir results] [--out thesis/back/appendixC_seeds.tex]

Per dataset: macro-F1 of every seed and arm; accuracy and MCC; precision, recall and F1 per class;
C5 per axiom (axiom-level satisfaction and domain-rule violation rate).
Means and sample deviations are over the seeds of each arm.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from summarize_runs import load_runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"nslkdd": "NSL-KDD", "cicids2017": "CIC-IDS2017", "cicddos2019": "CIC-DDoS2019"}
ARMS = (("rules", "Rule-only", "Rules"), ("mlp_noweights", "MLP-CE, no weights", "MLP nw"),
        ("mlp_weights", "MLP-CE, weights", "MLP w"), ("cnn", "1D-CNN", "CNN"), ("ltn_sup", "LTN-sup", "LTN-sup"),
        ("ltn_full", "LTN-full", "LTN-full"), ("rf", "Random Forest", "RF"), ("xgb", "XGBoost", "XGB"))


def run_id(ds, arm):
    return f"{ds}_rules" if arm == "rules" else f"{ds}_{arm}_tuned"


def arms_of(runs, ds):
    """[(arm, long name, short name, {seed: test metrics})] for the arms with test metrics."""
    out = []
    for arm, name, short in ARMS:
        by_seed = runs.get(run_id(ds, arm), {})
        tests = {s: r["metrics"]["test"] for s, r in by_seed.items() if "test" in r.get("metrics", {})}
        if tests:
            out.append((arm, name, short, tests))
    return out


def ms(v, d=4):
    v = np.asarray(v, dtype=float)
    if len(v) == 1:
        return f"{v[0]:.{d}f}"
    return f"{v.mean():.{d}f} $\\pm$ {v.std(ddof=1):.{d}f}"


def table(lines_head, rows, caption, label, cols):
    out = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
           " & ".join(lines_head) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def seed_table(ds, arms):
    seeds = sorted({s for *_, t in arms for s in t})
    rows = []
    for s in seeds:
        rows.append([str(s)] + [f"{t[s]['macro_f1']:.4f}" if s in t else "--" for *_, t in arms])
    return table(["Seed"] + [short for _, _, short, _ in arms], rows,
                 f"{DATASETS[ds]}, test macro-F1 of every seed of the final runs. The rule-only classifier has one "
                 f"value, repeated.", f"tab:app-seeds-{ds}", "l" + "r" * len(arms))


def secondary_table(ds, arms):
    rows = [[name, ms([t[s]["accuracy"] for s in t]), ms([t[s]["mcc"] for s in t])] for _, name, _, t in arms]
    return table(["Arm", "Accuracy", "MCC"], rows,
                 f"{DATASETS[ds]}, accuracy and MCC on the test part, mean $\\pm$ sample deviation over the seeds.",
                 f"tab:app-secondary-{ds}", "lcc")


def class_table(ds, arms):
    classes = next(iter(arms[0][3].values()))["class_names"]
    rows = []
    for _, name, _, t in arms:
        for key, tag in (("precision", "P"), ("recall", "R"), ("f1", "F1")):
            vals = [np.mean([t[s][key][i] for s in t]) for i in range(len(classes))]
            rows.append([name if tag == "P" else "", tag] + [f"{v:.3f}" for v in vals])
    return table(["Arm", ""] + list(classes), rows,
                 f"{DATASETS[ds]}, precision (P), recall (R) and F1 per class on the test part, mean over the seeds.",
                 f"tab:app-classes-{ds}", "ll" + "r" * len(classes))


def c5_table(ds, arms):
    """Axiom-level satisfaction and violation rate per axiom on the test part, mean over the seeds."""
    ids = []
    for *_, t in arms:
        for r in t.values():
            ids += [i for i in r.get("c5", {}).get("axioms", {}) if i not in ids]
    if not ids:
        return ""
    rows = []
    for _, name, _, t in arms:
        for key, tag in (("satisfaction", "Sat."), ("violation_rate", "Viol.")):
            cells = []
            for i in ids:
                v = [r["c5"]["axioms"][i][key] for r in t.values() if i in r.get("c5", {}).get("axioms", {})]
                v = [x for x in v if x == x]
                cells.append(f"{np.mean(v):.3f}" if v else "--")
            rows.append([name if tag == "Sat." else "", tag] + cells)
    return table(["Arm", ""] + ids, rows,
                 f"{DATASETS[ds]}, C5 on the test part per axiom of the main base, mean over the seeds: axiom-level "
                 f"satisfaction (Sat.) and domain-rule violation rate (Viol.) on the flows that the axiom covers.",
                 f"tab:app-c5-{ds}", "ll" + "r" * len(ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=str(ROOT / "thesis/back/appendixC_seeds.tex"))
    args = ap.parse_args()
    runs = load_runs(args.results_dir)
    parts = ["% written by analysis/seed_tables.py, do not edit by hand", ""]
    for ds in DATASETS:
        arms = arms_of(runs, ds)
        if not arms:
            print(f"{ds}: no final runs")
            continue
        print(f"{ds}: {', '.join(f'{a} n={len(t)}' for a, _, _, t in arms)}")
        parts += [seed_table(ds, arms), secondary_table(ds, arms), class_table(ds, arms), c5_table(ds, arms)]
    Path(args.out).write_text("\n".join(parts))
    print(args.out)


if __name__ == "__main__":
    main()
