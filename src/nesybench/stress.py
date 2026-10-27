"""Configs of the stress runs (E3, E4, E7, E8, E9, E11, E13) from a tuned config.

    python -m nesybench.stress --base results/tuning/tune_nslkdd_ltn_full_best.yaml \
        --experiment base|e3|e4|e7|e8|e9|e11|e13 [--kb FILE --tag NAME] [--holdout LABEL ...] [--out configs/stress]
E7 takes any arm: --levels F ... (train fractions), --floor N, --exclude ID ... (ltn_full only).
E8 takes any arm: --classes C ... thinned to --levels F ... of their train rows (--floor N).
E7 and E8 cut the validation part the same way (shrink_val): the labelled budget is train plus validation.
E4 takes an LTN arm: p, crisp thresholds and s_scale one at a time (crisp and s_scale: ltn_full).

E3, E11 and the weak KB also take a rule_only base (configs/<dataset>_rules.yaml); E13 does not.
The hyperparameters stay those of the base config, nothing is tuned again.
"""
import argparse
import copy
from pathlib import Path

import yaml

from nesybench.logic.edit import kb_file
from nesybench.logic.kb import load_kb
from nesybench.run import load_config

LEVELS = (-0.10, -0.05, 0.05, 0.10, 0.20)   # tau shift, fraction of the train column (0.05 = 5 percentile points)
COPIES = (2, 4, 6, 8)                        # extra copies of one axiom
E7_LEVELS = {                                # train fractions; 1.0 is the final run
    "nslkdd": (0.5, 0.2, 0.1, 0.05),
    "cicids2017": (0.25, 0.1, 0.03, 0.01),
    "cicddos2019": (0.1, 0.01, 0.001),
}
E7_FLOOR = 5                                 # rows kept per class at any fraction
E8_CLASSES = {                               # classes made rarer in train
    "nslkdd": ("R2L", "U2R"),
    "cicids2017": ("R2L", "U2R", "Botnet"),
    "cicddos2019": ("Normal",),
}
E8_LEVELS = (0.5, 0.25, 0.1)
E4_P = (1, 2, 4, 6)                          # same set as the tuning
E4_S_SCALE = (0.5, 1.0, 2.0)


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


def e9(base: dict, labels, tag: str, kb=None) -> dict:
    """Any arm: the fine labels never seen in train or validation, all in the test part.

    kb: a KB derived without those labels; C5 uses it in every arm, training only in ltn_full.
    """
    if not labels or tag == "kb":
        raise SystemExit("e9 needs --holdout LABEL... and --tag NAME")
    if base.get("split") != "stratified_random":
        raise SystemExit("e9 holds labels out of a stratified_random split")
    config = _derive(base, f"e9_{tag}", None)
    config["split_params"] = {**base.get("split_params", {}), "holdout_labels": list(labels),
                              "holdout_attempted": True}
    if kb:
        config["c5_kb"] = kb
        if base.get("kb"):
            config["kb"] = kb
    return config


def _pct(fraction: float) -> str:
    return f"{fraction * 100:.6g}".replace(".", "p")


def e7(base: dict, levels=None, floor=E7_FLOOR, exclude=()) -> list[dict]:
    """Any arm: train and validation cut to a fraction (test whole), floor per class."""
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
                                  "min_keep_per_class": floor, "shrink_val": True}
        out.append(config)
    return out


def e8(base: dict, classes=None, levels=None, floor=E7_FLOOR) -> list[dict]:
    """Any arm: the listed classes cut to a fraction of their train and validation rows."""
    classes = tuple(classes or E8_CLASSES.get(base.get("dataset"), ()))
    if not classes:
        raise SystemExit(f"e8: no default classes for {base.get('dataset')!r}, use --classes")
    levels = tuple(levels or E8_LEVELS)
    if any(not 0 < f < 1 for f in levels):
        raise SystemExit("e8: levels must be in (0, 1)")
    out = []
    for f in levels:
        config = _derive(base, f"e8_f{_pct(f)}", None)
        config["split_params"] = {**base.get("split_params", {}),
                                  "train_class_fractions": {c: f for c in classes},
                                  "min_keep_per_class": floor, "shrink_val": True}
        out.append(config)
    return out


def e4(base: dict, p_levels=E4_P, s_levels=E4_S_SCALE) -> list[dict]:
    """One LTN setting at a time against the base: p, crisp thresholds, width scale."""
    model = base.get("model")
    if model not in ("ltn_sup", "ltn_full"):
        raise SystemExit("e4 needs an ltn_sup or ltn_full base config")
    training = base.get("training", {})
    out = []
    for p in p_levels:
        if p != training.get("p", 2):
            config = _derive(base, f"e4_p{p:g}", None)
            config["training"] = {**training, "p": p}
            out.append(config)
    if model == "ltn_full":
        if not training.get("crisp", False):
            config = _derive(base, "e4_crisp", None)
            config["training"] = {**training, "crisp": True}
            out.append(config)
            for s in s_levels:
                if s != training.get("s_scale", 1.0):
                    config = _derive(base, f"e4_s{s:g}".replace(".", "p"), None)
                    config["training"] = {**training, "s_scale": s}
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
    parser.add_argument("--experiment", required=True, choices=["base", "e3", "e4", "e7", "e8", "e9", "e11", "e13"])
    parser.add_argument("--out", default="configs/stress")
    parser.add_argument("--kb", help="another KB file, e.g. the weak-knowledge variant (E9: the one derived without the held-out labels)")
    parser.add_argument("--tag", default="kb", help="run_id suffix with --kb")
    parser.add_argument("--holdout", nargs="+", help="E9: fine labels kept out of train")
    parser.add_argument("--swap", nargs="*", default=[], help="E11: ID:class")
    parser.add_argument("--axiom", help="E13: axiom to copy (default: the first)")
    parser.add_argument("--levels", nargs="+", type=float, help="E7, E8: train fractions")
    parser.add_argument("--classes", nargs="+", help="E8: classes made rarer")
    parser.add_argument("--floor", type=int, default=E7_FLOOR, help="E7, E8: rows kept per class")
    parser.add_argument("--exclude", nargs="+", default=[], help="E7: axiom ids out of the KB")
    args = parser.parse_args(argv)

    base = load_config(args.base)
    if args.experiment == "e9":
        configs = [e9(base, args.holdout, args.tag, args.kb)]
        _write(configs, args.out)
        return
    if args.experiment == "e7":
        _write(e7(base, args.levels, args.floor, args.exclude), args.out)
        return
    if args.experiment == "e8":
        _write(e8(base, args.classes, args.levels, args.floor), args.out)
        return
    if args.experiment == "e4":
        _write(e4(base), args.out)
        return
    if base.get("model") not in ("ltn_full", "rule_only") or not base.get("kb"):
        raise SystemExit("the base config must be an ltn_full or rule_only config with a kb")
    if base["model"] == "rule_only" and args.experiment == "e13":
        raise SystemExit("e13 measures training cost: it needs an ltn_full base config")
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
