import json

import numpy as np
import pytest

from nesybench.utils.results import save_run


def test_numpy_values_are_saved(tmp_path):
    metrics = {
        "accuracy": np.float64(0.9),
        "f1": np.array([0.5, 1.0]),
        "confusion_matrix": np.array([[1, 0], [0, 1]]),
    }
    path = save_run(str(tmp_path), "t", {"model": "mlp_ce"}, 0, metrics, {"train_s": 1.5})
    record = json.loads(path.read_text())
    assert record["metrics"]["accuracy"] == 0.9
    assert record["metrics"]["f1"] == [0.5, 1.0]
    assert record["metrics"]["confusion_matrix"] == [[1, 0], [0, 1]]
    assert "git_dirty" in record


def test_existing_run_is_not_overwritten(tmp_path):
    save_run(str(tmp_path), "t", {}, 0, {}, {})
    with pytest.raises(FileExistsError):
        save_run(str(tmp_path), "t", {}, 0, {}, {})
    save_run(str(tmp_path), "t", {}, 0, {}, {}, overwrite=True)
