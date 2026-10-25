"""Configs of the stress runs (E3, E7, E9, E11, E13) from a tuned config.

    python -m nesybench.stress --base results/tuning/tune_nslkdd_ltn_full_best.yaml \
        --experiment base|e3|e7|e9|e11|e13 [--kb FILE --tag NAME] [--holdout LABEL ...] [--out configs/stress]
E7 takes any arm: --levels F ... (train fractions), --floor N, --exclude ID ... (ltn_full only).

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
E7_LEVELS = {                                # train fractions; 1.0 is the final run
    "nslkdd": (0.5, 0.2, 0.1, 0.05),
    "cicids2017": (0.25, 0.1, 0.03, 0.01),
    "cicddos2019": (0.1, 0.01, 0.001),
}
E7_FLOOR = 5                                 # rows kept per class at any fraction


def _derive(base: dict, suffix: str, edit: dict) -> dict:
    config = copy.deepcopy(base)
    config["run_id"] = f"{base['run_id']}_{suffix}"
    if edit is not None:
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


def e9(base: dict, labels, tag: str) -> dict:
    """Any arm: the fine labels never seen in train or validation, all in the test part."""
    if not labels or tag == "kb":
        raise SystemExit("e9 needs --holdout LABEL... and --tag NAME")
    if base.get("split") != "stratified_random":
        raise SystemExit("e9 holds labels out of a stratified_random split")
    config = _derive(base, f"e9_{tag}", None)
    config["split_params"] = {**base.get("split_params", {}), "holdout_labels": list(labels)}
    return config


def _pct(fraction: float) -> str:
    return f"{fraction * 100:.6g}".replace(".", "p")


def e7(base: dict, levels=None, floor=E7_FLOOR, exclude=()) -> list[dict]:
    """Any arm: the train part cut to a fraction (validation and test whole), floor per class."""
    levels = tuple(levels or E7_LEVELS.get(base.get("dataset"), ()))
    if not levels:
        raise SystemExit(f"e7: no default levels for {base.get('dataset')!r}, use --levels")
    if any(not 0 < f < 1 for f in levels):
        raise SystemExit("e7: levels must be in (0, 1); 1.0 is the final run")
    if exclude and base.get("model") != "ltn_full":
        raise SystemExit("e7: --exclude needs an ltn_full base config")
    suffix = "_no_" + "_".join(exclude) if exclude else ""
    out = []
    for f in levels:
        config = _derive(base, f"e7{suffix}_f{_pct(f)}", {"exclude": list(exclude)} if exclude else None)
        config["split_params"] = {**base.get("split_params", {}), "train_fraction": f,
                                  "min_keep_per_class": floor}
        out.append(config)
    return out


def _write(configs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for config in configs:
        (out / f"{config['run_id']}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(out / f"{config['run_id']}.yaml")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--experiment", required=True, choices=["base", "e3", "e7", "e9", "e11", "e13"])
    parser.add_argument("--out", default="configs/stress")
    parser.add_argument("--kb", help="another KB file, e.g. the weak-knowledge variant")
    parser.add_argument("--tag", default="kb", help="run_id suffix with --kb")
    parser.add_argument("--holdout", nargs="+", help="E9: fine labels kept out of train")
    parser.add_argument("--swap", nargs="*", default=[], help="E11: ID:class")
    parser.add_argument("--axiom", help="E13: axiom to copy (default: the first)")
    parser.add_argument("--levels", nargs="+", type=float, help="E7: train fractions")
    parser.add_argument("--floor", type=int, default=E7_FLOOR, help="E7: rows kept per class")
    parser.add_argument("--exclude", nargs="+", default=[], help="E7: axiom ids out of the KB")
    args = parser.parse_args(argv)

    base = load_config(args.base)
    if args.experiment == "e9":
        configs = [e9(base, args.holdout, args.tag)]
        _write(configs, args.out)
        return
    if args.experiment == "e7":
        _write(e7(base, args.levels, args.floor, args.exclude), args.out)
        return
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

    _write(configs, args.out)


if __name__ == "__main__":
    main()
