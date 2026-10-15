import json

import numpy as np
import pytest

from nesybench.utils.results import _package_version, run_path, save_run


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


def test_training_info_is_saved(tmp_path):
    path = save_run(str(tmp_path), "t", {}, 0, {}, {}, training={"best_epoch": 7})
    assert json.loads(path.read_text())["training"] == {"best_epoch": 7}


def test_run_path_is_where_the_result_is_written(tmp_path):
    path = save_run(str(tmp_path), "t", {}, 3, {}, {})
    assert path == run_path(str(tmp_path), "t", 3)


def test_version_falls_back_to_install_metadata(tmp_path, monkeypatch):
    package = tmp_path / "fakepkg_nover"
    package.mkdir()
    (package / "__init__.py").write_text("")
    info = tmp_path / "fakepkg_nover-1.2.3.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakepkg_nover\nVersion: 1.2.3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _package_version("fakepkg_nover") == "1.2.3"


def test_missing_package_has_no_version():
    assert _package_version("surely_not_installed_pkg") is None
