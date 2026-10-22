"""Candidate axiom conditions on the CIC-DDoS2019 training day (the test file is never read).

    python analysis/axiom_candidates_cicddos2019.py               # named candidates
    python analysis/axiom_candidates_cicddos2019.py --cond "<expr>" [--target Normal]
    python analysis/axiom_candidates_cicddos2019.py --rules       # single-feature scan
    python analysis/axiom_candidates_cicddos2019.py --tree        # depth-3 tree, fit/check halves
    ... --exclude "<expr>"   # work only on the rows the expression does NOT cover (what is left)

Columns with spaces go in backticks in --cond. Inbound, IPs and ports are not model inputs;
they are read here only to see what they would add (the PORTS candidate).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from nesybench.data import get_spec, load_dataset
from nesybench.data.base import CLASS_COL, LABEL_COL

ATTACK, NORMAL = "DDoS", "Normal"
AMPLIFIERS = (53, 69, 111, 123, 137, 161, 389, 1434, 1900)  # DNS, TFTP, Portmap, NTP, NetBIOS, SNMP, LDAP, MSSQL, SSDP

# name: (expression, class it should point to, origin)
CANDIDATES = {
    "SYN_NO_REPLY": ("Protocol == 'tcp' and `SYN Flag Count` >= 1 and `Total Backward Packets` == 0",
                     ATTACK, "mechanism: SYN flood leaves the handshake open"),
    "UDP_NO_REPLY": ("Protocol == 'udp' and `Total Backward Packets` == 0",
                     ATTACK, "mechanism: UDP flood and reflection are one-way"),
    "UDP_BIG_NO_REPLY": ("Protocol == 'udp' and `Total Backward Packets` == 0 and `Fwd Packet Length Mean` > 500",
                         ATTACK, "mechanism: amplified replies are large (500 is a placeholder)"),
    "NO_REPLY": ("`Total Backward Packets` == 0", ATTACK, "mechanism: no answer from the victim"),
    "TCP_TWO_WAY": ("Protocol == 'tcp' and `Total Backward Packets` > 0 and `ACK Flag Count` >= 1",
                    NORMAL, "mechanism: completed exchange"),
    "PORTS": ("Protocol == 'udp' and `Source Port` in %s" % (AMPLIFIERS,),
              ATTACK, "meta column (service port of the reflector): information only, not for the KB"),
}


def feature_table(df, spec):
    cols = [c for c in df.columns if c not in spec.non_features and c not in spec.categorical]
    X = df[cols].astype(float)
    for v in sorted(df["Protocol"].unique()):
        X[f"Protocol={v}"] = (df["Protocol"] == v).astype(float)
    return X


def cond_expr(column, op, cut):
    if column.startswith("Protocol="):
        value = column.split("=")[1]
        return f"Protocol {'==' if op == '>' else '!='} '{value}'"
    return f"`{column}` {op} {cut:g}"


def stats(df, expr, target):
    hit = df.eval(expr, engine="python").astype(bool).to_numpy()
    is_t = (df[CLASS_COL] == target).to_numpy()
    other = (df[CLASS_COL] != target).to_numpy()
    n_hit = int(hit.sum())
    return hit, {"hits": n_hit, "support": (hit & is_t).sum() / max(is_t.sum(), 1),
                 "confidence": (hit & is_t).sum() / n_hit if n_hit else float("nan"),
                 "wrong_class_hits": int((hit & other).sum())}


def by_label(df, hit):
    tab = pd.DataFrame({"label": df[LABEL_COL], "hit": hit}).groupby("label").agg(
        rows=("hit", "size"), holds=("hit", "sum"))
    tab["share"] = tab["holds"] / tab["rows"]
    return tab[tab["holds"] > 0].sort_values("holds", ascending=False)


def named(df):
    rows = {}
    for name, (expr, target, origin) in CANDIDATES.items():
        hit, s = stats(df, expr, target)
        rows[name] = {"target": target, **s, "origin": origin}
        print(f"\n{name}: {expr}\n  rows by fine label where it holds:")
        print(by_label(df, hit).head(12).to_string())
    print("\n== summary (support = share of the target class covered; wrong_class_hits = rows of the other class)")
    print(pd.DataFrame(rows).T.to_string())


def rule_scan(df, X, min_support, top):
    is_t = (df[CLASS_COL] == ATTACK).to_numpy()
    rows = {}
    for c in X.columns:
        v = X[c].to_numpy()
        for cut in np.unique(np.quantile(v[is_t], np.linspace(0, 1, 11), method="nearest")):
            for op, hit in ((">", v > cut), ("<=", v <= cut)):
                n_hit = int(hit.sum())
                h_t = int((hit & is_t).sum())
                if n_hit and h_t / is_t.sum() >= min_support:
                    rows[cond_expr(c, op, cut)] = (c, h_t / is_t.sum(), h_t / n_hit, n_hit,
                                                   int((hit & ~is_t).sum()))
    out = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=["feature", "support", "confidence", "hits", "normal_hits"])
    out = out.sort_values(["confidence", "support"], ascending=False).groupby("feature", sort=False).head(1)
    print(out.head(top).reset_index(names="condition").drop(columns="feature").to_string(index=False))


def leaf_rules(tree, names):
    t, out = tree.tree_, {}

    def walk(node, conds):
        if t.children_left[node] == -1:
            out[node] = conds
            return
        c, cut = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [cond_expr(c, "<=", cut)])
        walk(t.children_right[node], conds + [cond_expr(c, ">", cut)])

    walk(0, [])
    return out


def tree_rules(df, X, depth, min_leaf, min_hits, seed):
    is_t = (df[CLASS_COL] == ATTACK).to_numpy()
    a, b = train_test_split(np.arange(len(df)), test_size=0.5, random_state=seed,
                            stratify=df[CLASS_COL])
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf,
                                  class_weight="balanced", random_state=seed)
    tree.fit(X.iloc[a].to_numpy(), is_t[a])  # balanced only steers the splits; counts below are plain
    leaf = tree.apply(X.to_numpy())
    rows = []
    for node, conds in leaf_rules(tree, list(X.columns)).items():
        res = []
        for part in (a, b):
            m = leaf[part] == node
            n, h = int(m.sum()), int((m & is_t[part]).sum())
            res += [h, n - h, h / n if n else float("nan")]
        if res[0] >= min_hits:
            rows.append((*res, " and ".join(conds)))
    out = pd.DataFrame(rows, columns=["hits_fit", "other_fit", "conf_fit", "hits_chk", "other_chk",
                                      "conf_chk", "rule"])
    pd.options.display.max_colwidth = None
    print(out.sort_values("conf_fit", ascending=False).to_string(index=False)
          if len(out) else "no leaf with enough target rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cond", help="pandas expression, columns with spaces in backticks")
    ap.add_argument("--target", default=ATTACK, choices=[ATTACK, NORMAL])
    ap.add_argument("--exclude", help="drop the rows this expression covers before anything else")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-support", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--min-hits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, spec = load_dataset("cicddos2019", "train"), get_spec("cicddos2019")
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 220
    print(f"CIC-DDoS2019 training day  n={len(df)}  {df[CLASS_COL].value_counts().to_dict()}")
    if args.exclude:
        df = df[~df.eval(args.exclude, engine="python").astype(bool).to_numpy()].reset_index(drop=True)
        print(f"after excluding {args.exclude}: n={len(df)}  {df[CLASS_COL].value_counts().to_dict()}")
    if args.cond:
        hit, s = stats(df, args.cond, args.target)
        print(f"{args.cond}\n{s}")
        print(by_label(df, hit).head(20).to_string())
    elif args.rules:
        rule_scan(df, feature_table(df, spec), args.min_support, args.top)
    elif args.tree:
        tree_rules(df, feature_table(df, spec), args.depth, args.min_leaf, args.min_hits, args.seed)
    else:
        named(df)


if __name__ == "__main__":
    main()
