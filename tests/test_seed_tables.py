import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "analysis"))
import seed_tables as st  # noqa: E402


def rec(seed, macro):
    return {"seed": seed, "metrics": {"test": {"macro_f1": macro, "accuracy": macro + 0.1, "mcc": macro - 0.1,
                                               "precision": [0.9, 0.5], "recall": [0.8, 0.4],
                                               "class_names": ["Normal", "DDoS"]}}}


def test_tables_per_seed_and_class():
    runs = {"cicddos2019_ltn_full_tuned": {s: rec(s, 0.99 - 0.001 * s) for s in range(3)},
            "cicddos2019_rules": {0: rec(0, 0.92)}}
    arms = st.arms_of(runs, "cicddos2019")
    assert [a[0] for a in arms] == ["rules", "ltn_full"]
    seeds = st.seed_table("cicddos2019", arms)
    assert "1 & -- & 0.9890" in seeds
    assert "Rule-only & 1.0200 & 0.8200" in st.secondary_table("cicddos2019", arms)
    classes = st.class_table("cicddos2019", arms)
    assert "Rule-only & P & 0.900 & 0.500" in classes and " & R & 0.800 & 0.400" in classes
