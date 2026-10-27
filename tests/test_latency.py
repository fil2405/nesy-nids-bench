import json

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from nesybench import latency


def test_measure_shape_and_order():
    calls = []
    x = np.zeros((10, 3), dtype=np.float32)
    out = latency.measure(lambda b: calls.append(len(b)), x, batch_sizes=(1, 4), repeats=5, warmup=2)
    assert set(out) == {"1", "4"}
    assert calls == [1] * 7 + [4] * 7
    for row in out.values():
        assert row["p50_ms"] <= row["p95_ms"] <= row["p99_ms"]
        assert row["rows_per_second"] > 0


def test_batch_larger_than_the_data_cycles_rows():
    seen = []
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    latency.measure(lambda b: seen.append(b.copy()), x, batch_sizes=(5,), repeats=1, warmup=0)
    assert seen[0][:, 0].tolist() == [0.0, 2.0, 4.0, 0.0, 2.0]


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"warmup": -1}, {"batch_sizes": (0,)}])
def test_bad_arguments_are_refused(kwargs):
    with pytest.raises(ValueError):
        latency.measure(lambda b: None, np.zeros((2, 2)), **{"repeats": 1, "warmup": 0, **kwargs})


def test_cli_on_a_saved_forest(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 4))
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(x, rng.integers(0, 2, 60))
    folder = tmp_path / "toy_seed0"
    folder.mkdir()
    joblib.dump(forest, folder / "model.joblib")
    (folder / "meta.json").write_text(json.dumps({"feature_names": list("abcd"),
                                                  "class_names": ["Normal", "DDoS"]}))
    out = tmp_path / "lat.json"
    latency.main(["--model-dir", str(folder), "--batch-sizes", "1", "8", "--repeats", "3",
                  "--warmup", "1", "--out", str(out)])
    got = json.loads(out.read_text())
    assert got["kind"] == "random_forest" and got["input"] == "synthetic" and got["n_features"] == 4
    assert set(got["batches"]) == {"1", "8"} and "model.joblib" in got["size_bytes"]


def test_missing_model_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        latency.load_predictor(tmp_path)


def test_cli_on_a_saved_network(tmp_path):
    from nesybench.models.mlp import build_mlp, save_mlp

    folder = tmp_path / "toy_net_seed0"
    folder.mkdir()
    save_mlp(build_mlp(6, 3, [8], 0.1, "relu", seed=0), folder)
    (folder / "meta.json").write_text(json.dumps({"feature_names": list("abcdef"),
                                                  "class_names": ["a", "b", "c"]}))
    predict, kind = latency.load_predictor(folder)
    assert kind == "keras"
    proba = predict(np.zeros((5, 6), dtype=np.float32))
    assert proba.shape == (5, 3) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    out = tmp_path / "lat.json"
    latency.main(["--model-dir", str(folder), "--batch-sizes", "1", "--repeats", "3",
                  "--warmup", "1", "--device", "cpu", "--out", str(out)])
    assert json.loads(out.read_text())["batches"]["1"]["p50_ms"] > 0


def _kb(tmp_path):
    from nesybench.logic.kb import load_kb
    path = tmp_path / "kb.yaml"
    path.write_text("version: toy\naxioms:\n  - id: A1\n    group: dos\n    origin: mechanism\n"
                    "    target: DDoS\n    all:\n      - {col: rate, op: gt, tau: 0.5, s: 0.1}\n")
    return load_kb(path)


def test_pipeline_times_model_and_rules_on_the_same_rows(tmp_path):
    import pandas as pd
    x = np.zeros((4, 2), dtype=np.float32)
    raw = pd.DataFrame({"rate": [0.9, 0.1, 0.9, 0.1], "other": list("abcd")})
    seen = []

    def predict(batch):
        seen.append(len(batch))
        return np.tile([0.9, 0.1], (len(batch), 1))

    out = latency.measure_pipeline(predict, x, raw, _kb(tmp_path), ["Normal", "DDoS"],
                                   batch_sizes=(1, 3), repeats=2, warmup=1)
    assert set(out) == {"model", "annotate", "enforce"}
    assert all(set(v) == {"1", "3"} for v in out.values())
    assert seen == ([1] * 3 + [3] * 3) * 3


def test_pipeline_needs_aligned_rows(tmp_path):
    import pandas as pd
    with pytest.raises(ValueError, match="rows"):
        latency.measure_pipeline(lambda b: b, np.zeros((3, 1)), pd.DataFrame({"rate": [0.0]}),
                                 _kb(tmp_path), ["Normal", "DDoS"])


def test_postcheck_needs_a_config(tmp_path):
    with pytest.raises(SystemExit):
        latency.main(["--model-dir", str(tmp_path), "--postcheck"])
