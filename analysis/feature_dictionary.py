"""Feature dictionary of each dataset and the column mapping of the transfer, as LaTeX (Appendix D).

    python analysis/feature_dictionary.py [--datasets nslkdd cicids2017 cicddos2019] \
        [--out thesis/back/appendixD_tables.tex]

Per dataset: the model inputs in column order, categorical ones with their categories in the
training data, columns constant in it, and the axioms of the main KB that read them.
With both CIC datasets: the source columns renamed in the common schema and those left out.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nesybench.data import get_spec, load_dataset  # noqa: E402
from nesybench.logic.kb import load_kb  # noqa: E402

NAMES = {"nslkdd": "NSL-KDD", "cicids2017": "CIC-IDS2017", "cicddos2019": "CIC-DDoS2019"}
READ = {"nslkdd": "KDDTrain+", "cicids2017": "the whole dataset", "cicddos2019": "the sampled training day"}
KB = {"nslkdd": "kb/nslkdd_v1.yaml", "cicids2017": "kb/cicids2017_v1.yaml",
      "cicddos2019": "kb/cicddos2019_v1.yaml"}
PER_COLUMN = 2


def tt(name):
    out = name
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        out = out.replace(a, b)
    return r"\texttt{" + out + "}"


def axioms_by_column(kb_path):
    out = {}
    for a in load_kb(ROOT / kb_path).axioms:
        for c in a.conditions:
            out.setdefault(c.col, []).append(a.id)
    return {k: sorted(set(v)) for k, v in out.items()}


def describe(df, spec, kb_path=None):
    """[(column, note)] for the model inputs of `df`."""
    used = axioms_by_column(kb_path) if kb_path else {}
    rows = []
    for col in (c for c in df.columns if c not in spec.non_features):
        notes = []
        if col in spec.categorical:
            notes.append(f"cat., {df[col].nunique()}")
        elif df[col].nunique() <= 1:
            notes.append("const.")
        if col in used:
            notes.append(", ".join(used[col]))
        rows.append((col, "; ".join(notes)))
    return rows


def table(rows, caption, label, per_column=PER_COLUMN):
    n = -(-len(rows) // per_column)
    cols = [rows[i * n:(i + 1) * n] for i in range(per_column)]
    spec = "@{}" + "ll" * per_column + "@{}"
    lines = [r"\begin{table}[p]", r"\centering", r"\scriptsize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & ".join(["Feature", "Note"] * per_column) + r" \\", r"\midrule"]
    for i in range(n):
        cells = []
        for c in cols:
            name, note = c[i] if i < len(c) else ("", "")
            cells += [tt(name) if name else "", note]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def mapping_table(source_cols, target_cols):
    from nesybench.data.cic_transfer import common_columns
    cols = common_columns(source_cols, target_cols)
    renamed = [(s, t) for s, t in cols.items() if s != t]
    lines = [r"\begin{table}[p]", r"\centering", r"\scriptsize",
             r"\caption{Common schema of the transfer: columns of CIC-IDS2017 whose name changes, with the name "
             rf"of CIC-DDoS2019 they take. {len(cols)} columns are common.}}",
             r"\label{tab:app-transfer-schema}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}ll@{}}", r"\toprule", r"CIC-IDS2017 & CIC-DDoS2019 \\", r"\midrule"]
    lines += [f"{tt(s)} & {tt(t)} \\\\" for s, t in renamed]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    only_source = [c for c in source_cols if c not in cols]
    only_target = [c for c in target_cols if c not in cols.values()]
    text = (f"% only CIC-IDS2017: {', '.join(only_source) or 'none'}\n"
            f"% only CIC-DDoS2019: {', '.join(only_target) or 'none'}\n")
    return "\n".join(lines), cols, only_source, only_target, text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(NAMES), choices=list(NAMES))
    ap.add_argument("--out", default=str(ROOT / "thesis/back/appendixD_tables.tex"))
    args = ap.parse_args()

    parts = ["% written by analysis/feature_dictionary.py, do not edit by hand", ""]
    inputs = {}
    for ds in args.datasets:
        spec = get_spec(ds)
        df = load_dataset(ds, "train")
        rows = describe(df, spec, KB[ds])
        inputs[ds] = [r[0] for r in rows]
        print(f"{ds}: {len(rows)} input columns, {sum('cat.' in r[1] for r in rows)} categorical, "
              f"{sum('const.' in r[1] for r in rows)} constant")
        parts.append(table(rows, f"{NAMES[ds]}: input columns in the order of the data. cat., $k$: categorical "
                                 f"with $k$ values in {READ[ds]}; const.: one value in all of it; "
                                 f"axiom identifiers: axioms of the main knowledge base that read the column "
                                 f"(\\cref{{app:axioms}}).", f"tab:app-features-{ds}"))
    if {"cicids2017", "cicddos2019"} <= set(inputs):
        tex, cols, only_s, only_t, text = mapping_table(inputs["cicids2017"], inputs["cicddos2019"])
        print(f"transfer: {len(cols)} common, {len(only_s)} only in CIC-IDS2017, {len(only_t)} only in CIC-DDoS2019")
        parts += [tex, text]
    Path(args.out).write_text("\n".join(parts))
    print(args.out)


if __name__ == "__main__":
    main()
