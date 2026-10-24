import yaml

from nesybench import stress

KB_YAML = """
version: toy_v1
axioms:
  - {id: A1, group: ddos, origin: mechanism, target: DoS,
     all: [{col: count, op: gt, tau: 30, s: 5}]}
  - {id: A2, group: dataset, origin: data, target: Probe,
     all: [{col: rate, op: gt, tau: 0.9, s: 0.1}]}
"""


def _base(tmp_path):
    kb = tmp_path / "kb.yaml"
    kb.write_text(KB_YAML)
    cfg = {"run_id": "toy_ltn_full", "dataset": "nslkdd", "split": "official_test_with_stratified_val",
           "model": "ltn_full", "kb": str(kb), "seeds": [0, 1], "training": {"p": 2},
           "evaluate_test": False, "trial_set": {"x": 1}}
    path = tmp_path / "best.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_e3_one_per_axiom_and_group(tmp_path):
    out = tmp_path / "o"
    stress.main(["--base", str(_base(tmp_path)), "--experiment", "e3", "--out", str(out)])
    names = sorted(p.stem for p in out.glob("*.yaml"))
    assert names == ["toy_ltn_full_e3_no_A1", "toy_ltn_full_e3_no_A2",
                     "toy_ltn_full_e3_no_gdataset", "toy_ltn_full_e3_no_gddos"]
    cfg = yaml.safe_load((out / "toy_ltn_full_e3_no_A1.yaml").read_text())
    assert cfg["kb_edit"] == {"exclude": ["A1"]}
    assert cfg["seeds"] == [0, 1] and "trial_set" not in cfg and cfg["evaluate_test"] is False


def test_e11_levels_and_swap(tmp_path):
    out = tmp_path / "o"
    stress.main(["--base", str(_base(tmp_path)), "--experiment", "e11", "--out", str(out),
                 "--swap", "A1:Probe"])
    names = sorted(p.stem for p in out.glob("*.yaml"))
    assert "toy_ltn_full_e11_tau_m10" in names and "toy_ltn_full_e11_tau_p20" in names
    assert "toy_ltn_full_e11_swap_A1" in names and len(names) == 6


def test_e13_counts(tmp_path):
    out = tmp_path / "o"
    stress.main(["--base", str(_base(tmp_path)), "--experiment", "e13", "--out", str(out)])
    names = sorted(p.stem for p in out.glob("*.yaml"))
    assert set(names) == {f"toy_ltn_full_e13_k{k}" for k in (4, 6, 8, 10)}
