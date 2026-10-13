"""Model registry: the only place that knows which models exist.

A model is a class with the interface shared by every arm of the benchmark:

    fit(splits, config) -> dict
        Train. `config` is the run configuration plus the key `seed`. The
        returned dict is saved in the result file under "training".
    predict_proba(splits, part) -> array of shape (n_examples, n_classes)
        `part` is "train", "val" or "test".

Input size and number of classes come from `splits`, never from numbers
written in a model. A model is added by writing its module and one line in
_MODELS. A string value ("module:Class") is imported on first use, so that
importing this package does not load TensorFlow.
"""
import importlib

_MODELS: dict[str, str | type] = {
    "mlp_ce": "nesybench.models.mlp_ce:MLPCE",
}


def known_models() -> list[str]:
    return sorted(_MODELS)


def get_model(name: str) -> type:
    if name not in _MODELS:
        raise ValueError(f"unknown model {name!r}, known: {known_models()}")
    entry = _MODELS[name]
    if isinstance(entry, str):
        module, _, class_name = entry.partition(":")
        entry = getattr(importlib.import_module(module), class_name)
    return entry
