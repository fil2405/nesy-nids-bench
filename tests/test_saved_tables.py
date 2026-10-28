import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "analysis"))
import saved_tables as st  # noqa: E402


def seed_rec(base, enf, noise, fixed=3, broken=5):
    pc = {"macro_f1": enf, "n_changed": 8, "fixed": fixed, "broken": broken,
          "explanations": {"n_alerts": 100, "coverage": 0.5, "fidelity": 0.9, "conciseness": 1.0}}
    ann = {**pc, "macro_f1": base}
    return {"macro_f1": base, "postcheck": {"annotate": ann, "enforce": pc},
            "rules": {"A1": {"fired": 40, "precision": 0.75}},
            "perturbed": {"noise:0.1": {"macro_f1": noise, "postcheck": {"annotate": ann, "enforce": pc}}}}


@pytest.fixture
def folder(tmp_path):
    for run_id, b in (("nslkdd_ltn_full_tuned", 0.6), ("nslkdd_rules", 0.3), ("cicids2017_rf_tuned", 0.9)):
        rep = {"run_id": run_id, "part": "test",
               "per_seed": {str(s): seed_rec(b + 0.01 * s, b + 0.01 * s - 0.02, b - 0.05) for s in range(3)}}
        (tmp_path / f"{run_id}.json").write_text(json.dumps(rep))
    return tmp_path


def test_reports_by_arm_in_order(folder):
    assert list(st.load_reports(folder, "nslkdd")) == ["ltn_full", "rules"]
    assert st.arm_of("cic_transfer_rf_tuned_e9_xfer", "cic_transfer") == "rf_tuned_e9_xfer"


def test_postcheck_diff_and_counts(folder):
    rows = st.postcheck_rows(st.load_reports(folder, "nslkdd"))
    arm, n, base, ann, enf, d, fixed, broken = rows[0]
    assert (arm, n) == ("ltn_full", 3) and d.median == pytest.approx(-0.02)
    assert fixed[0] == 3 and broken[0] == 5 and ann[0] == pytest.approx(base[0])
    assert st.rule_rows(st.load_reports(folder, "nslkdd")) == [("A1", 40, 0.75)]


def test_e10_diff_from_clean(folder):
    reports = st.load_reports(folder, "nslkdd")
    (arm, clean, cells), _ = st.e10_rows(reports, st.perturbations(reports))
    assert cells["noise:0.1"][1].median == pytest.approx(-0.06)


def test_e17_and_latency(folder, tmp_path):
    (arm, n, cov, fid, con), _ = st.e17_rows(st.load_reports(folder, "nslkdd"), "annotate")
    assert cov[0] == 0.5 and fid[0] == 0.9
    lat = {"batches": {}, "pipeline": {k: {"1": {"p50_ms": v}, "64": {"p50_ms": v * 2}}
                                       for k, v in (("model", 1.0), ("annotate", 1.5), ("enforce", 1.6))}}
    (tmp_path / "lat").mkdir()
    (tmp_path / "lat" / "nslkdd_xgb_tuned.json").write_text(json.dumps(lat))
    assert st.latency_rows(tmp_path / "lat", "nslkdd") == [("xgb", {"1": (1.0, 1.5, 1.6), "64": (2.0, 3.0, 3.2)})]


def test_postcheck_without_annotate(tmp_path):
    rec = seed_rec(0.6, 0.62, 0.5)
    del rec["postcheck"]["annotate"]
    rep = {"run_id": "nslkdd_xgb_tuned", "part": "test", "per_seed": {"0": rec, "1": rec}}
    (tmp_path / "x.json").write_text(json.dumps(rep))
    (row,) = st.postcheck_rows(st.load_reports(tmp_path, "nslkdd"))
    assert row[4][0] == pytest.approx(0.62)
