"""Model registry.

Interface: fit(splits, config) -> dict saved under "training";
predict_proba(splits, part) -> (n_examples, n_classes).
Entries are "module:Class" strings, imported lazily to keep TensorFlow out
of light imports.
"""
import importlib

_MODELS: dict[str, str | type] = {
    "mlp_ce": "nesybench.models.mlp_ce:MLPCE",
    "rule_only": "nesybench.models.rules:RuleOnly",
    "rf": "nesybench.models.trees:RandomForest",
    "xgb": "nesybench.models.trees:XGBoost",
    "ltn_sup": "nesybench.models.ltn_sup:LTNSup",
    "ltn_full": "nesybench.models.ltn_full:LTNFull",
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
