"""What models may rely on in Splits. Synthetic data: runs without the dataset."""
import numpy as np
import pandas as pd

from nesybench.data.preprocess import CLASSES, preprocess


def _toy(n_per_class=20):
    n = n_per_class * len(CLASSES)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "duration": rng.random(n),
            "src_bytes": rng.integers(0, 1000, n),
            "protocol_type": np.resize(["tcp", "udp", "icmp"], n),
            "service": np.resize(["http", "ftp"], n),
            "flag": np.resize(["SF", "S0"], n),
            "label": "x",
            "class": np.repeat(CLASSES, n_per_class),
            "difficulty": 21,
        }
    )


def test_feature_names_match_columns():
    s = preprocess(_toy(), _toy(), seed=0)
    assert len(s.feature_names) == s.X_train.shape[1]
    assert "duration" in s.feature_names
    assert "protocol_type_tcp" in s.feature_names


def test_class_names_match_labels():
    s = preprocess(_toy(), _toy(), seed=0)
    assert s.class_names == CLASSES
    assert set(np.unique(s.y_train)) == set(range(len(s.class_names)))
