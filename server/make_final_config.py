"""Config delle run finali: python server/make_final_config.py DATASET ARM

Parte dal _best.yaml del tuning (o dal config del rule-only), accende evaluate_test e mette
i seed finali. Scrive in results/final_configs/ (ignorata da git) e stampa "percorso<TAB>run_id".
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan import PLAN, ROOT, load_tuning, tuned  # noqa: E402


def main():
    ds, arm = sys.argv[1:3]
    if tuned(arm):
        tuning = load_tuning(ds, arm)
        if tuning is None:
            sys.exit(f"manca configs/tuning/{ds}_{arm}.yaml")
        src = ROOT / "results" / "tuning" / f"{tuning['tuning_id']}_best.yaml"
        if not src.exists():
            sys.exit(f"manca {src.relative_to(ROOT)}: prima server/02_tune.sh {ds} {arm}")
    else:
        src = ROOT / "configs" / f"{ds}_rules.yaml"
    config = yaml.safe_load(src.read_text())
    config["evaluate_test"] = True
    config["seeds"] = list(PLAN["final"]["seeds"])
    out = ROOT / "results" / "final_configs" / f"{ds}_{arm}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"{out}\t{config['run_id']}")


if __name__ == "__main__":
    main()
