"""Support, confidence and benign hits of each axiom on the train part (crisp).

    python analysis/kb_train_stats.py kb/nslkdd_v1_weak.yaml [dataset]

Prints the `train:` line of each axiom, to paste in the file. Test never read.
"""
import sys
from pathlib import Path

import yaml

from nesybench.data import load_dataset
from nesybench.data.base import CLASS_COL
from nesybench.logic.kb import antecedents, load_kb
from nesybench.logic.c5 import THRESHOLD


def stats(kb, df):
    a = antecedents(kb, df, crisp=True) >= THRESHOLD
    out = {}
    for j, axiom in enumerate(kb.axioms):
        hit = a[:, j]
        target = (df[CLASS_COL] == axiom.target).to_numpy()
        n = int(hit.sum())
        out[axiom.id] = {
            "support": round(float((hit & target).sum() / max(target.sum(), 1)), 3),
            "confidence": round(float((hit & target).sum() / n), 3) if n else float("nan"),
            "benign_hit": int((hit & (df[CLASS_COL] == "Normal").to_numpy()).sum()),
        }
    return out


def main():
    path = Path(sys.argv[1])
    dataset = sys.argv[2] if len(sys.argv) > 2 else yaml.safe_load(path.read_text())["version"].split("_")[0]
    kb = load_kb(path)
    df = load_dataset(dataset, "train")
    print(f"# {kb.version} on {dataset} train, {len(df)} rows")
    for ident, s in stats(kb, df).items():
        print(f"{ident}: train: {{support: {s['support']}, confidence: {s['confidence']}, "
              f"benign_hit: {s['benign_hit']}}}")


if __name__ == "__main__":
    main()
