"""Piano e avanzamento delle run.

    python server/plan.py table              conteggi e stato per fase
    python server/plan.py list DS PHASE      bracci pronti per tune|final (blocchi su stderr)
    python server/plan.py progress           run finite / attese
    python server/plan.py audit              run finali con git_dirty non falso
    python server/plan.py kind ARM | seeds   per gli script bash
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PLAN = yaml.safe_load((ROOT / "server" / "plan.yaml").read_text())


def tuning_path(ds, arm):
    return ROOT / "configs" / "tuning" / f"{ds}_{arm}.yaml"


def load_tuning(ds, arm):
    path = tuning_path(ds, arm)
    return yaml.safe_load(path.read_text()) if path.exists() else None


def tuned(arm):
    return PLAN["arms"][arm].get("tuned", True)


def blocked(ds, arm, phase):
    """Motivo per cui il braccio non puo' girare ora, o None."""
    spec = PLAN["arms"][arm]
    if spec.get("needs_kb") and not PLAN["datasets"][ds]["kb"]:
        return "manca la KB"
    if tuned(arm):
        tuning = load_tuning(ds, arm)
        if tuning is None:
            return f"manca configs/tuning/{ds}_{arm}.yaml"
        space = tuning.get("space", {})
        if phase == "tune" and spec["kind"] == "neural" and "network.hidden_units" in space:
            return None  # gira, ma D6 non applicata (vedi table)
    elif not (ROOT / "configs" / f"{ds}_rules.yaml").exists():
        return f"manca configs/{ds}_rules.yaml"
    return None


def d6_pending(ds, arm):
    tuning = load_tuning(ds, arm)
    return bool(tuning and PLAN["arms"][arm]["kind"] == "neural"
                and "network.hidden_units" in tuning.get("space", {}))


def counts(ds):
    t, s = PLAN["tuning"], PLAN["final"]["seeds"]
    tune_all = sum(t["trials"] * len(t["seeds"]) for a in PLAN["arms"] if tuned(a))
    tune_ready = sum(t["trials"] * len(t["seeds"]) for a in PLAN["arms"]
                     if tuned(a) and not blocked(ds, a, "tune"))
    fin_all = len(PLAN["arms"]) * len(s)
    fin_ready = sum(len(s) for a in PLAN["arms"] if not blocked(ds, a, "final"))
    return tune_all, tune_ready, fin_all, fin_ready


def cmd_table(_):
    print(f"{'dataset':12} {'tuning':>14} {'finali':>12}   note")
    for ds in PLAN["datasets"]:
        ta, tr, fa, fr = counts(ds)
        notes = [f"{a}: {blocked(ds, a, 'tune')}" for a in PLAN["arms"] if blocked(ds, a, "tune")]
        notes += [f"{a}: spazio con rete/batch per braccio (D6 da applicare)"
                  for a in PLAN["arms"] if d6_pending(ds, a)]
        print(f"{ds:12} {tr:>6}/{ta:<7} {fr:>5}/{fa:<6}   {'; '.join(notes)}")
    print("\nstress                      run (bracci x livelli x seed)   da fare prima")
    n_seeds = len(PLAN["final"]["seeds"])
    for name, e in PLAN["stress"].items():
        lv = e["levels"]
        runs = "n.d." if lv is None or not e["arms"] else str(
            len(e["datasets"]) * len(e["arms"]) * lv * n_seeds)
        print(f"{name:5} {e['what'][:38]:38} {runs:>6}   {'; '.join(e['needs'])}")
    print("\nconteggi = pronte/previste. Nessun tempo qui: si misura con 01_scaling.sh.")


def cmd_list(args):
    for arm in PLAN["arms"]:
        if args.phase == "tune" and not tuned(arm):
            continue
        reason = blocked(args.dataset, arm, args.phase)
        if reason:
            print(f"salto {args.dataset}/{arm}: {reason}", file=sys.stderr)
        else:
            print(arm)


def count_files(pattern):
    return len(glob.glob(str(ROOT / pattern)))


def cmd_kind(args):
    print(PLAN["arms"][args.arm]["kind"])


def cmd_seeds(_):
    print(" ".join(str(x) for x in PLAN["final"]["seeds"]))


def cmd_progress(_):
    t, seeds = PLAN["tuning"], PLAN["final"]["seeds"]
    print(f"{'dataset':12} {'braccio':14} {'tuning':>10} {'finali':>8}")
    for ds in PLAN["datasets"]:
        for arm in PLAN["arms"]:
            if blocked(ds, arm, "final"):
                continue
            cell_t = "-"
            if tuned(arm):
                tun = load_tuning(ds, arm)
                done = count_files(f"results/tuning/{tun['tuning_id']}_t??_seed*.json")
                cell_t = f"{done}/{t['trials'] * len(t['seeds'])}"
                run_id = tun.get("final_run_id")
            else:
                run_id = yaml.safe_load((ROOT / "configs" / f"{ds}_rules.yaml").read_text())["run_id"]
            done_f = count_files(f"results/{run_id}_seed*.json") if run_id else 0
            print(f"{ds:12} {arm:14} {cell_t:>10} {done_f:>5}/{len(seeds)}")


def cmd_audit(_):
    bad = []
    for path in glob.glob(str(ROOT / "results" / "*_seed*.json")):
        record = json.loads(Path(path).read_text())
        if record.get("git_dirty") is not False:
            bad.append((Path(path).name, record.get("git_commit"), record.get("git_dirty")))
    for name, commit, dirty in sorted(bad):
        print(f"non citabile: {name} commit={commit} git_dirty={dirty}")
    print(f"{len(bad)} run non citabili")
    return 1 if bad else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table").set_defaults(fn=cmd_table)
    p = sub.add_parser("list")
    p.add_argument("dataset", choices=list(PLAN["datasets"]))
    p.add_argument("phase", choices=["tune", "final"])
    p.set_defaults(fn=cmd_list)
    sub.add_parser("progress").set_defaults(fn=cmd_progress)
    p = sub.add_parser("kind")
    p.add_argument("arm", choices=list(PLAN["arms"]))
    p.set_defaults(fn=cmd_kind)
    sub.add_parser("seeds").set_defaults(fn=cmd_seeds)
    sub.add_parser("audit").set_defaults(fn=cmd_audit)
    args = parser.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
