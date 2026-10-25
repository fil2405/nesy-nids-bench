import yaml

from nesybench import stress

KB_YAML = """
version: toy_v1
axioms:
  - {id: A1, group: dos, origin: mechanism, target: DoS,
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
                     "toy_ltn_full_e3_no_gdataset", "toy_ltn_full_e3_no_gdos"]
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


def test_other_kb_and_base(tmp_path):
    out = tmp_path / "o"
    other = tmp_path / "weak.yaml"
    other.write_text(KB_YAML.replace("toy_v1", "toy_v1_weak"))
    stress.main(["--base", str(_base(tmp_path)), "--experiment", "base", "--out", str(out),
                 "--kb", str(other), "--tag", "weak"])
    (path,) = out.glob("*.yaml")
    cfg = yaml.safe_load(path.read_text())
    assert path.stem == "toy_ltn_full_weak" and cfg["run_id"] == "toy_ltn_full_weak"
    assert cfg["kb"] == cfg["c5_kb"] == str(other) and "kb_edit" not in cfg


def test_e9_holds_labels_out_of_any_arm(tmp_path):
    out = tmp_path / "o"
    cfg = {"run_id": "toy_mlp", "dataset": "cicids2017", "split": "stratified_random",
           "split_params": {"test_seed": 0}, "model": "mlp_ce", "kb": None, "seeds": [0],
           "training": {}, "evaluate_test": False}
    base = tmp_path / "mlp.yaml"
    base.write_text(yaml.safe_dump(cfg))
    stress.main(["--base", str(base), "--experiment", "e9", "--out", str(out),
                 "--holdout", "DoS Slowloris", "DoS Slowhttptest", "--tag", "slow"])
    (path,) = out.glob("*.yaml")
    got = yaml.safe_load(path.read_text())
    assert path.stem == "toy_mlp_e9_slow"
    assert got["split_params"] == {"test_seed": 0,
                                   "holdout_labels": ["DoS Slowloris", "DoS Slowhttptest"]}
    assert "kb_edit" not in got


def _arm(tmp_path, dataset="cicids2017", model="mlp_ce", **extra):
    cfg = {"run_id": f"toy_{model}", "dataset": dataset, "split": "stratified_random",
           "split_params": {"test_seed": 0}, "model": model, "kb": None, "seeds": [0, 1],
           "training": {}, "evaluate_test": False, "trial_set": {"x": 1}, **extra}
    path = tmp_path / f"{model}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_e7_levels_per_dataset_on_any_arm(tmp_path):
    out = tmp_path / "o"
    stress.main(["--base", str(_arm(tmp_path)), "--experiment", "e7", "--out", str(out)])
    names = sorted(p.stem for p in out.glob("*.yaml"))
    assert names == ["toy_mlp_ce_e7_f1", "toy_mlp_ce_e7_f10", "toy_mlp_ce_e7_f25",
                     "toy_mlp_ce_e7_f3"]
    got = yaml.safe_load((out / "toy_mlp_ce_e7_f3.yaml").read_text())
    assert got["split_params"] == {"test_seed": 0, "train_fraction": 0.03, "min_keep_per_class": 5}
    assert got["seeds"] == [0, 1] and got["evaluate_test"] is False
    assert "trial_set" not in got and "kb_edit" not in got


def test_e7_levels_floor_and_small_fraction_name(tmp_path):
    out = tmp_path / "o"
    stress.main(["--base", str(_arm(tmp_path, "cicddos2019")), "--experiment", "e7",
                 "--out", str(out), "--floor", "3"])
    names = sorted(p.stem for p in out.glob("*.yaml"))
    assert names == ["toy_mlp_ce_e7_f0p1", "toy_mlp_ce_e7_f1", "toy_mlp_ce_e7_f10"]
    got = yaml.safe_load((out / "toy_mlp_ce_e7_f0p1.yaml").read_text())
    assert got["split_params"]["min_keep_per_class"] == 3


def test_e7_explicit_levels_and_axiom_out(tmp_path):
    out = tmp_path / "o"
    kb = tmp_path / "kb.yaml"
    kb.write_text(KB_YAML)
    base = _arm(tmp_path, "nslkdd", "ltn_full", kb=str(kb))
    stress.main(["--base", str(base), "--experiment", "e7", "--out", str(out),
                 "--levels", "0.5", "0.1", "--exclude", "A2"])
    names = sorted(p.stem for p in out.glob("*.yaml"))
    assert names == ["toy_ltn_full_e7_no_A2_f10", "toy_ltn_full_e7_no_A2_f50"]
    got = yaml.safe_load((out / "toy_ltn_full_e7_no_A2_f10.yaml").read_text())
    assert got["kb_edit"] == {"exclude": ["A2"]}


def test_e7_refuses_bad_input(tmp_path):
    import pytest
    out = tmp_path / "o"
    with pytest.raises(SystemExit):
        stress.main(["--base", str(_arm(tmp_path)), "--experiment", "e7", "--out", str(out),
                     "--levels", "1.0"])
    with pytest.raises(SystemExit):
        stress.main(["--base", str(_arm(tmp_path)), "--experiment", "e7", "--out", str(out),
                     "--exclude", "A1"])


def test_e8_thins_the_rare_classes_of_any_arm(tmp_path):
    out = tmp_path / "o"
    stress.main(["--base", str(_arm(tmp_path, "nslkdd")), "--experiment", "e8", "--out", str(out)])
    names = sorted(p.stem for p in out.glob("*.yaml"))
    assert names == ["toy_mlp_ce_e8_f10", "toy_mlp_ce_e8_f25", "toy_mlp_ce_e8_f50"]
    got = yaml.safe_load((out / "toy_mlp_ce_e8_f25.yaml").read_text())
    assert got["split_params"] == {"test_seed": 0, "min_keep_per_class": 5,
                                   "train_class_fractions": {"R2L": 0.25, "U2R": 0.25}}
    assert "kb_edit" not in got and "trial_set" not in got


def test_e8_explicit_classes_and_levels(tmp_path):
    out = tmp_path / "o"
    stress.main(["--base", str(_arm(tmp_path, "cicddos2019")), "--experiment", "e8",
                 "--out", str(out), "--levels", "0.2"])
    (path,) = out.glob("*.yaml")
    assert yaml.safe_load(path.read_text())["split_params"]["train_class_fractions"] == {"Normal": 0.2}
    stress.main(["--base", str(_arm(tmp_path)), "--experiment", "e8", "--out", str(out),
                 "--classes", "Probe", "--levels", "0.5"])
    got = yaml.safe_load((out / "toy_mlp_ce_e8_f50.yaml").read_text())
    assert got["split_params"]["train_class_fractions"] == {"Probe": 0.5}


def test_e4_one_setting_at_a_time(tmp_path):
    out = tmp_path / "o"
    kb = tmp_path / "kb.yaml"
    kb.write_text(KB_YAML)
    base = _arm(tmp_path, "nslkdd", "ltn_full", kb=str(kb), training={"p": 4})
    stress.main(["--base", str(base), "--experiment", "e4", "--out", str(out)])
    names = sorted(p.stem for p in out.glob("*.yaml"))
    assert names == ["toy_ltn_full_e4_crisp", "toy_ltn_full_e4_p1", "toy_ltn_full_e4_p2",
                     "toy_ltn_full_e4_p6", "toy_ltn_full_e4_s0p5", "toy_ltn_full_e4_s2"]
    got = yaml.safe_load((out / "toy_ltn_full_e4_crisp.yaml").read_text())
    assert got["training"] == {"p": 4, "crisp": True}
    got = yaml.safe_load((out / "toy_ltn_full_e4_s2.yaml").read_text())
    assert got["training"] == {"p": 4, "s_scale": 2.0}


def test_e4_sup_only_varies_p_and_refuses_other_arms(tmp_path):
    import pytest
    out = tmp_path / "o"
    stress.main(["--base", str(_arm(tmp_path, "nslkdd", "ltn_sup")), "--experiment", "e4",
                 "--out", str(out)])
    assert sorted(p.stem for p in out.glob("*.yaml")) == [
        "toy_ltn_sup_e4_p1", "toy_ltn_sup_e4_p4", "toy_ltn_sup_e4_p6"]
    with pytest.raises(SystemExit):
        stress.main(["--base", str(_arm(tmp_path)), "--experiment", "e4", "--out", str(out)])


def test_e9_other_kb_reaches_training_only_where_the_arm_trains_on_it(tmp_path):
    out = tmp_path / "o"
    kb = tmp_path / "kb.yaml"
    kb.write_text(KB_YAML)
    ltn = _arm(tmp_path, "cicids2017", "ltn_full", kb=str(kb), c5_kb=None)
    mlp = _arm(tmp_path, "cicids2017", "mlp_ce", c5_kb=str(kb))
    for base in (ltn, mlp):
        stress.main(["--base", str(base), "--experiment", "e9", "--out", str(out),
                     "--holdout", "SSH-Patator", "--tag", "ssh", "--kb", "kb/other.yaml"])
    got_ltn = yaml.safe_load((out / "toy_ltn_full_e9_ssh.yaml").read_text())
    got_mlp = yaml.safe_load((out / "toy_mlp_ce_e9_ssh.yaml").read_text())
    assert got_ltn["kb"] == got_ltn["c5_kb"] == "kb/other.yaml"
    assert got_mlp["c5_kb"] == "kb/other.yaml" and got_mlp["kb"] is None
