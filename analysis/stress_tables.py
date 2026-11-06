"""E9 (held-out attack labels) and E4 (LTN semantics) tables over the seeds.

    python analysis/stress_tables.py e9 --dataset cicids2017 [--tex out/e9.tex]
    python analysis/stress_tables.py e4 --dataset nslkdd [--metric macro_f1] [--tex out/e4.tex]
    python analysis/stress_tables.py e3|e11|e13 --dataset nslkdd [--tag _weak]
    python analysis/stress_tables.py unseen --dataset cicddos2019 --labels Portmap
    python analysis/stress_tables.py transfer [--tex out/transfer.tex]

E9: per variant and arm, macro-F1 and, pooled over the held-out labels, detection
(predicted not Normal) and class recall; detection on their Attempted flows (class Normal)
is a false alarm. Knowledge effect on detection: ltn_full minus ltn_sup, paired by seed.
The E9 test part differs from the finals: compare only inside a variant.
unseen: on the finals, the same detection for labels that only the test part has.
transfer: E9 from CIC-IDS2017 to CIC-DDoS2019, per arm and KB base: macro-F1, attacks detected,
benign flows with an alert, and the paired differences of TRANSFER_PAIRS.
E4: per setting, the score and its paired difference from the finals of the same arm.
No Holm in either (sensitivity checks, Ch. 10).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from compare_runs import metric_value  # noqa: E402
from stress_curves import ARMS  # noqa: E402
from summarize_runs import load_runs  # noqa: E402

from nesybench.compare import paired  # noqa: E402
from nesybench.data.splits import ATTEMPTED_SUFFIX  # noqa: E402

LTN_ARMS = ("ltn_sup", "ltn_full")


def by_variant(runs, dataset, exp, arms=ARMS, tag=""):
    """{variant: {arm: run_id}} for <dataset>_<arm>_tuned<tag>_<exp>_<variant>."""
    pat = re.compile(rf"^{dataset}_(?P<arm>{'|'.join(arms)})_tuned{tag}_{exp}_(?P<var>.+)$")
    out = {}
    for run_id in runs:
        m = pat.match(run_id)
        if m:
            out.setdefault(m["var"], {})[m["arm"]] = run_id
    return dict(sorted(out.items()))


def pooled(rec, labels, key, part="test"):
    """Row-weighted mean of by_label[key] over the labels present in the part; nan if none."""
    by = rec["metrics"][part].get("by_label", {})
    rows = [(by[lab]["n"], by[lab][key]) for lab in labels if lab in by and by[lab]["n"]]
    if not rows:
        return float("nan")
    n = np.array([r[0] for r in rows], dtype=float)
    return float((n * np.array([r[1] for r in rows])).sum() / n.sum())


def held_labels(rec):
    return list(rec["config"].get("split_params", {}).get("holdout_labels") or [])


def mean_std(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def e9_rows(runs, variants, part="test"):
    """(variant, arm, n, labels, macro-F1, detected, recall, attempted alarms), each a (mean, std)."""
    out = []
    for var, arms in variants.items():
        for arm in ARMS:
            if arm not in arms:
                continue
            recs = [runs[arms[arm]][s] for s in sorted(runs[arms[arm]])]
            labels = held_labels(recs[0])
            attempted = [lab + ATTEMPTED_SUFFIX for lab in labels]
            out.append((var, arm, len(recs), labels,
                        mean_std([r["metrics"][part]["macro_f1"] for r in recs]),
                        mean_std([pooled(r, labels, "detected", part) for r in recs]),
                        mean_std([pooled(r, labels, "recall", part) for r in recs]),
                        mean_std([pooled(r, attempted, "detected", part) for r in recs])))
    return out


def e9_effect(runs, variants, part="test", hi="ltn_full", lo="ltn_sup"):
    """Per variant: detection of the held-out labels, hi minus lo over the common seeds."""
    out = []
    for var, arms in variants.items():
        if hi not in arms or lo not in arms:
            continue
        seeds = sorted(set(runs[arms[hi]]) & set(runs[arms[lo]]))
        if len(seeds) < 2:
            continue
        labels = held_labels(runs[arms[hi]][seeds[0]])
        a = [pooled(runs[arms[lo]][s], labels, "detected", part) for s in seeds]
        b = [pooled(runs[arms[hi]][s], labels, "detected", part) for s in seeds]
        out.append((var, paired(a, b)))
    return out


TRANSFER = re.compile(r"^cic_transfer_(?P<arm>.+?)(?:_tuned)?_e9_(?P<base>.+)$")
TRANSFER_PAIRS = ((("ltn_sup", "xfer"), ("ltn_full", "xfer")),
                  (("ltn_full", "xfer"), ("ltn_full", "xfer_mech")),
                  (("ltn_sup", "xfer"), ("ltn_full", "xfer_mech")),
                  (("mlp_weights", "xfer"), ("ltn_full", "xfer_mech")))


def transfer_runs(runs):
    """{(arm, base): run_id} of the E9 transfer runs."""
    out = {}
    for run_id in runs:
        m = TRANSFER.match(run_id)
        if m:
            out[(m["arm"], m["base"])] = run_id
    return out


def transfer_rows(runs, part="test"):
    """(arm, base, n, macro-F1, detected, false alarms), each (mean, std); by macro-F1, best first."""
    out = []
    for (arm, base), run_id in transfer_runs(runs).items():
        recs = [runs[run_id][s] for s in sorted(runs[run_id])]
        recall = [r["metrics"][part]["recall"] for r in recs]
        out.append((arm, base, len(recs), mean_std([r["metrics"][part]["macro_f1"] for r in recs]),
                    mean_std([r[1] for r in recall]), mean_std([1 - r[0] for r in recall])))
    return sorted(out, key=lambda r: -r[3][0])


def transfer_effects(runs, part="test", pairs=TRANSFER_PAIRS):
    """(a, b, paired b minus a on macro-F1) over the common seeds."""
    ids = transfer_runs(runs)
    out = []
    for a, b in pairs:
        if a not in ids or b not in ids:
            continue
        seeds = sorted(set(runs[ids[a]]) & set(runs[ids[b]]))
        if len(seeds) < 2:
            continue
        out.append((a, b, paired([runs[ids[a]][s]["metrics"][part]["macro_f1"] for s in seeds],
                                 [runs[ids[b]][s]["metrics"][part]["macro_f1"] for s in seeds])))
    return out


def tex_transfer(rows):
    lines = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Arm & Base & Macro-F1 & Detected & False alarms \\", r"\midrule"]
    for arm, base, n, mf1, det, fa in rows:
        tt = base.replace("_", r"\_")
        lines.append(f"{arm.replace('_', '-')} & \\texttt{{{tt}}} & {fmt(mf1, 4)} "
                     f"& {det[0]:.3f} & {fa[0]:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def unseen_rows(runs, dataset, labels, part="test"):
    """Finals only: per arm, detection and class recall pooled over labels absent from train."""
    out = []
    for arm in (*ARMS, "rules"):
        run_id = f"{dataset}_{arm}_tuned" if arm != "rules" else f"{dataset}_rules"
        if run_id not in runs:
            continue
        recs = [runs[run_id][s] for s in sorted(runs[run_id])]
        out.append((arm, len(recs),
                    mean_std([pooled(r, labels, "detected", part) for r in recs]),
                    mean_std([pooled(r, labels, "recall", part) for r in recs])))
    return out


def saved_eval_runs(folder):
    """saved_eval reports as runs: {run_id: {seed: {"metrics": {part: {"by_label": ...}}}}}."""
    import json
    out = {}
    for path in sorted(Path(folder).glob("*.json")):
        rep = json.loads(path.read_text())
        if "per_seed" not in rep:
            continue
        out[rep["run_id"]] = {int(s): {"metrics": {rep["part"]: {"by_label": r.get("by_label", {}),
                                                                 "macro_f1": r["macro_f1"]}}}
                              for s, r in rep["per_seed"].items()}
    return out


def e4_rows(runs, dataset, variants, part, metric, arms=LTN_ARMS, tag=""):
    """(arm, setting, n, mean, std, paired diff from the base run or None)."""
    out = []
    for arm in arms:
        final = f"{dataset}_{arm}_tuned{tag}"
        for var, arms in variants.items():
            if arm not in arms:
                continue
            by_seed = runs[arms[arm]]
            scores = [metric_value(by_seed[s], part, metric) for s in sorted(by_seed)]
            m, s = mean_std(scores)
            diff = None
            if final in runs:
                seeds = sorted(set(by_seed) & set(runs[final]))
                if len(seeds) >= 2:
                    diff = paired([metric_value(runs[final][k], part, metric) for k in seeds],
                                  [metric_value(by_seed[k], part, metric) for k in seeds])
            out.append((arm, var, len(scores), m, s, diff))
    return out


def base_setting(runs, dataset, arm, tag=""):
    """p, crisp, s_scale of the base run of `arm`, as written in its config."""
    final = f"{dataset}_{arm}_tuned{tag}"
    if final not in runs:
        return {}
    training = next(iter(runs[final].values()))["config"].get("training", {})
    return {k: training[k] for k in ("p", "crisp", "s_scale") if k in training}


def fmt(ms, digits=3):
    m, s = ms
    return "--" if not np.isfinite(m) else f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def tex_e9(rows):
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Variant & Arm & macro-F1 & Detected & Class recall & Alarms on Attempted \\",
             r"\midrule"]
    for var, arm, n, labels, mf1, det, rec, att in rows:
        lines.append(f"{var.replace('_', '-')} & {arm.replace('_', '-')} & {fmt(mf1)} & {fmt(det)} "
                     f"& {fmt(rec)} & {fmt(att)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tex_e4(rows, metric):
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             rf"Arm & Setting & {metric.replace('_', ' ')} & median diff from base & 95\% interval & wins \\",
             r"\midrule"]
    for arm, var, n, m, s, d in rows:
        cells = ("--", "--", "--") if d is None else (
            f"{d.median:+.4f}", f"[{d.low:+.4f}, {d.high:+.4f}]", f"{d.wins}/{d.n}")
        lines.append(f"{arm.replace('_', '-')} & {var.replace('_', '-')} & {m:.3f} $\\pm$ {s:.3f} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_tex(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("exp", choices=["e9", "e4", "e3", "e11", "e13", "unseen", "transfer"])
    ap.add_argument("--tag", default="", help="e3/e11/e13 on another base, e.g. _weak")
    ap.add_argument("--labels", nargs="+", help="unseen: fine labels absent from train (e.g. Portmap)")
    ap.add_argument("--saved-eval", help="unseen: folder of saved_eval reports, for finals without by_label")
    ap.add_argument("--dataset", help="all but transfer")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--part", default="test", choices=["val", "test"])
    ap.add_argument("--metric", default="macro_f1", help="E4 only")
    ap.add_argument("--tex")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if args.exp != "transfer" and not args.dataset:
        raise SystemExit(f"{args.exp} needs --dataset")
    if args.exp == "transfer":
        rows = transfer_rows(runs, args.part)
        if not rows:
            raise SystemExit(f"no cic_transfer runs in {args.results_dir}")
        print(f"E9 transfer CIC-IDS2017 -> CIC-DDoS2019 ({args.part}): detected = Attack recall, "
              "false alarms = share of Normal flagged")
        for arm, base, n, mf1, det, fa in rows:
            print(f"  {arm:14} {base:10} n={n:2d}  macro-F1 {mf1[0]:.4f} +- {mf1[1]:.4f}"
                  f"  detected {det[0]:.4f}  false alarms {fa[0]:.4f}")
        print("paired macro-F1, second minus first")
        for a, b, r in transfer_effects(runs, args.part):
            print(f"  {'/'.join(b)} - {'/'.join(a)}: median {r.median:+.4f} [{r.low:+.4f}, {r.high:+.4f}]"
                  f"  wins {r.wins}/{r.n}  p {r.p:.4f}")
        if args.tex:
            write_tex(args.tex, tex_transfer(rows))
    elif args.exp == "unseen":
        if not args.labels:
            raise SystemExit("unseen needs --labels")
        if args.saved_eval:
            runs = saved_eval_runs(args.saved_eval)
        print(f"{args.dataset} finals ({args.part}), labels {args.labels}")
        for arm, n, det, rec in unseen_rows(runs, args.dataset, args.labels, args.part):
            print(f"  {arm:14} n={n:2d}  detected {det[0]:.4f} +- {det[1]:.4f}"
                  f"  recall {rec[0]:.4f} +- {rec[1]:.4f}")
    elif args.exp == "e9":
        variants = by_variant(runs, args.dataset, "e9")
        if not variants:
            raise SystemExit(f"no e9 runs for {args.dataset} in {args.results_dir}")
        rows = e9_rows(runs, variants, args.part)
        print(f"{args.dataset} e9 ({args.part}); E9 test differs from the finals")
        last = None
        for var, arm, n, labels, mf1, det, rec, att in rows:
            if var != last:
                print(f"\n{var}: held out {labels}")
                print(f"  {'arm':14} {'n':>3} {'macro-F1':>16} {'detected':>16} {'recall':>16} {'alarms Att.':>16}")
                last = var
            print(f"  {arm:14} {n:3d} " + " ".join(f"{m:8.4f} +- {s:.4f}" for m, s in (mf1, det, rec, att)))
        print("\nknowledge effect on detection (ltn_full minus ltn_sup)")
        for var, r in e9_effect(runs, variants, args.part):
            print(f"  {var:10} n={r.n}  median {r.median:+.4f} [{r.low:+.4f}, {r.high:+.4f}]"
                  f"  wins {r.wins}/{r.n} ties {r.ties}  p {r.p:.4f}")
        if args.tex:
            write_tex(args.tex, tex_e9(rows))
    else:
        arms = LTN_ARMS if args.exp == "e4" else ("ltn_full",)
        variants = by_variant(runs, args.dataset, args.exp, arms, args.tag)
        if not variants:
            raise SystemExit(f"no {args.exp} runs for {args.dataset}{args.tag} in {args.results_dir}")
        rows = e4_rows(runs, args.dataset, variants, args.part, args.metric, arms, args.tag)
        print(f"{args.dataset}{args.tag} {args.exp} {args.metric} ({args.part}); diff = variant minus base run")
        for arm in arms:
            print(f"  {arm} base: {base_setting(runs, args.dataset, arm, args.tag) or 'base run not found'}")
        for arm, var, n, m, s, d in rows:
            tail = "" if d is None else (f"  diff {d.median:+.4f} [{d.low:+.4f}, {d.high:+.4f}]"
                                         f"  wins {d.wins}/{d.n}  p {d.p:.4f}")
            print(f"  {arm:9} {var:12} n={n:2d}  {m:.4f} +- {s:.4f}{tail}")
        if args.tex:
            write_tex(args.tex, tex_e4(rows, args.metric))


if __name__ == "__main__":
    main()
