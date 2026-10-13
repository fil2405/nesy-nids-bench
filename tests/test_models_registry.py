import subprocess
import sys

import pytest

from nesybench import models


def test_unknown_model_is_an_error():
    with pytest.raises(ValueError, match="mlp_ce"):  # the message lists the known models
        models.get_model("svm")


def test_registered_class_is_returned(monkeypatch):
    class Dummy:
        pass

    monkeypatch.setitem(models._MODELS, "dummy", Dummy)
    assert models.get_model("dummy") is Dummy
    assert "dummy" in models.known_models()


def test_dotted_path_is_imported_on_first_use(monkeypatch):
    from collections import OrderedDict

    monkeypatch.setitem(models._MODELS, "ordered", "collections:OrderedDict")
    assert models.get_model("ordered") is OrderedDict


def test_importing_the_registry_does_not_load_tensorflow():
    code = "import sys, nesybench.models; print('tensorflow' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
