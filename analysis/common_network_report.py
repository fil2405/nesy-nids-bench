"""Mean validation macro-F1 per network setting across the neural tuning arms."""
import argparse
import json
from pathlib import Path
from statistics import mean

HIDDEN, DROPOUT, BATCH = "network.hidden_units", "network.dropout", "training.batch_size"


def load_arms(directory):
    arms = {}
    for path in sorted(Path(directory).glob("tune_*.json")):
        if "_seed" in path.stem:
            continue
        data = json.loads(path.read_text())
        trials = data.get("trials") or []
        ok = [t for t in trials if t["status"] == "ok"]
        if ok and HIDDEN in ok[0]["params"] and DROPOUT in ok[0]["params"]:
            arms[data["tuning_id"].removeprefix("tune_")] = (ok, len(trials) - len(ok))
    return arms


def _fmt(p, key):
    return "x".join(map(str, p[key])) if key == HIDDEN else str(p[key])


def table(arms, keys, title):
    cells = {}
    for arm, (trials, _) in arms.items():
        for t in trials:
            label = ", ".join(_fmt(t["params"], k) for k in keys)
            cells.setdefault(label, {}).setdefault(arm, []).append(t["mean"])
    names = list(arms)
    print(f"\n{title}")
    print("setting | " + " | ".join(names) + " | avg of arms | arms")
    for label in sorted(cells):
        row, means = [], []
        for arm in names:
            scores = cells[label].get(arm)
            row.append(f"{mean(scores):.4f} (n={len(scores)})" if scores else "-")
            if scores:
                means.append(mean(scores))
        print(f"{label} | " + " | ".join(row) + f" | {mean(means):.4f} | {len(means)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/tuning")
    args = parser.parse_args()
    arms = load_arms(args.dir)
    if not arms:
        raise SystemExit(f"no neural tuning summaries in {args.dir}")
    for arm, (trials, failed) in arms.items():
        best = max(trials, key=lambda t: t["mean"])
        p = best["params"]
        print(f"{arm}: {len(trials)} ok, {failed} failed; best trial {best['trial']} "
              f"{best['mean']:.4f} net {_fmt(p, HIDDEN)} dropout {p[DROPOUT]} "
              f"batch {p.get(BATCH, '-')}")
    table(arms, (HIDDEN, DROPOUT), "by architecture and dropout")
    table(arms, (HIDDEN,), "by architecture")
    table(arms, (DROPOUT,), "by dropout")
    table(arms, (BATCH,), "by batch size")


if __name__ == "__main__":
    main()
