"""Configs of the stress runs (E3, E11, E13) from a tuned LTN-full config.

    python -m nesybench.stress --base results/tuning/tune_nslkdd_ltn_full_best.yaml \
        --experiment base|e3|e11|e13 [--kb FILE --tag NAME] [--out configs/stress]

The hyperparameters stay those of the base config, nothing is tuned again.
"""
import argparse
import copy
from pathlib import Path

import yaml

from nesybench.logic.edit import kb_file
from nesybench.logic.kb import load_kb
from nesybench.run import load_config

LEVELS = (-0.10, -0.05, 0.05, 0.10, 0.20)   # tau shift, percentile points
COPIES = (2, 4, 6, 8)                        # extra copies of one axiom


def _derive(base: dict, suffix: str, edit: dict) -> dict:
    config = copy.deepcopy(base)
    config["run_id"] = f"{base['run_id']}_{suffix}"
    config["kb_edit"] = edit
    config["evaluate_test"] = False
    config.pop("trial_set", None)
    return config


def _tag(level: float) -> str:
    return f"{'m' if level < 0 else 'p'}{abs(level) * 100:.0f}"


def e3(base: dict, kb) -> list[dict]:
    """One axiom out, then one group out; both only if something stays in the KB."""
    out = []
    if len(kb.axioms) > 1:
        out = [_derive(base, f"e3_no_{a.id}", {"exclude": [a.id]}) for a in kb.axioms]
    groups = sorted({a.group for a in kb.axioms})
    if len(groups) > 1:
        out += [_derive(base, f"e3_no_g{g}", {"exclude": [f"group:{g}"]}) for g in groups]
    return out


def e11(base: dict, kb, levels=LEVELS, swap=()) -> list[dict]:
    """Every threshold shifted by a level; plus one target swap per `ID:class` in swap."""
    out = [_derive(base, f"e11_tau_{_tag(v)}", {"tau_shift": {"by": v}}) for v in levels]
    ids = set(kb.ids)
    for item in swap:
        ident, target = item.split(":", 1)
        if ident not in ids:
            raise ValueError(f"swap: {ident!r} not in {kb.version}")
        out.append(_derive(base, f"e11_swap_{ident}", {"swap_target": {ident: target}}))
    return out


def e13(base: dict, kb, axiom=None, counts=COPIES) -> list[dict]:
    axiom = axiom or kb.axioms[0].id
    if axiom not in kb.ids:
        raise ValueError(f"copies: {axiom!r} not in {kb.version}")
    return [_derive(base, f"e13_k{len(kb.axioms) + n}",
                    {"copies": {"axiom": axiom, "n": n, "spread": 0.1}}) for n in counts]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--experiment", required=True, choices=["base", "e3", "e11", "e13"])
    parser.add_argument("--out", default="configs/stress")
    parser.add_argument("--kb", help="another KB file, e.g. the weak-knowledge variant")
    parser.add_argument("--tag", default="kb", help="run_id suffix with --kb")
    parser.add_argument("--swap", nargs="*", default=[], help="E11: ID:class")
    parser.add_argument("--axiom", help="E13: axiom to copy (default: the first)")
    args = parser.parse_args(argv)

    base = load_config(args.base)
    if base.get("model") != "ltn_full" or not base.get("kb"):
        raise SystemExit("the base config must be an ltn_full config with a kb")
    if args.kb:
        base["kb"] = base["c5_kb"] = args.kb
        base["run_id"] = f"{base['run_id']}_{args.tag}"
    kb = load_kb(kb_file(base))
    if args.experiment == "base":
        configs = [{**copy.deepcopy(base), "evaluate_test": False}]
    elif args.experiment == "e3":
        configs = e3(base, kb)
    elif args.experiment == "e11":
        configs = e11(base, kb, swap=args.swap)
    else:
        configs = e13(base, kb, args.axiom)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for config in configs:
        (out / f"{config['run_id']}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(out / f"{config['run_id']}.yaml")


if __name__ == "__main__":
    main()
