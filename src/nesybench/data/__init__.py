"""Dataset registry."""
from . import cicids2017, nslkdd
from .base import DatasetSpec

_DATASETS = {
    "nslkdd": (nslkdd.SPEC, nslkdd.load_nslkdd),
    "cicids2017": (cicids2017.SPEC, cicids2017.load_cicids2017),
}


def _entry(name: str):
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}, known: {sorted(_DATASETS)}")
    return _DATASETS[name]


def get_spec(name: str) -> DatasetSpec:
    return _entry(name)[0]


def load_dataset(name: str, split: str = "train"):
    return _entry(name)[1](split)
