"""Controlli sui dati dei CIC: conflitti di etichetta per classe e quanto e' facile il problema.

    python analysis/cic_sanity.py [cicddos2019 cicids2017]

Legge solo la parte train (il test non si apre). Il taglio 80/20 e' casuale come
il validation dei tuning: i numeri non sono risultati, misurano la difficolta' dei dati.
"""
import sys

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from nesybench.data import get_spec, load_dataset
from nesybench.data.base import CLASS_COL, LABEL_COL


def feature_frame(df, spec):
    cols = [c for c in df.columns if c not in spec.non_features]
    return df[cols], cols


def conflicts(df, spec):
    """Righe con lo stesso vettore di feature (quelle viste dai modelli) e etichette diverse."""
    X, cols = feature_frame(df, spec)
    key = pd.util.hash_pandas_object(X, index=False)
    by_key = df.groupby(key)
    n_label = by_key[LABEL_COL].transform("nunique") > 1
    n_class = by_key[CLASS_COL].transform("nunique") > 1
    return {
        "rows": len(df),
        "distinct_vectors": int(key.nunique()),
        "rows_label_conflict": int(n_label.sum()),
        "rows_class_conflict": int(n_class.sum()),
        "class_conflict_by_class": df.loc[n_class, CLASS_COL].value_counts().to_dict(),
    }


def trees(df, spec, depths=(1, 2, 3), seed=0):
    X, cols = feature_frame(df, spec)
    X = pd.get_dummies(X, columns=list(spec.categorical), dtype=float)
    y = df[CLASS_COL]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    out = []
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(X_tr, y_tr)
        score = f1_score(y_va, tree.predict(X_va), average="macro")
        out.append((depth, score, X.columns[tree.tree_.feature[0]]))
    return out


def report(name):
    spec = get_spec(name)
    df = load_dataset(name, "train")
    print(f"== {name}")
    print("classi:", df[CLASS_COL].value_counts().to_dict())
    c = conflicts(df, spec)
    print(f"righe {c['rows']}, vettori distinti {c['distinct_vectors']}")
    print(f"righe con stesso vettore e etichetta fine diversa: {c['rows_label_conflict']}")
    print(f"righe con stesso vettore e classe diversa: {c['rows_class_conflict']} "
          f"{c['class_conflict_by_class']}")
    for depth, score, root in trees(df, spec):
        print(f"albero di profondita' {depth}: macro-F1 {score:.4f} (radice: {root})")
    print()


if __name__ == "__main__":
    for dataset in sys.argv[1:] or ["cicddos2019", "cicids2017"]:
        report(dataset)
