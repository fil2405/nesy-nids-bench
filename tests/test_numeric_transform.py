import io

import joblib
import numpy as np
import pandas as pd
import pytest

from nesybench.data.base import CLASS_COL, LABEL_COL, DatasetSpec
from nesybench.data.preprocess import preprocess, signed_log1p

SPEC = DatasetSpec(name="toy", classes=("Normal", "DoS"), categorical=("proto",),
                   non_features=(LABEL_COL, CLASS_COL), fixed_test=True)


def _frame(values):
    n = len(values)
    return pd.DataFrame({"rate": values, "flag": [0, 1] * (n // 2),
                         "proto": ["tcp", "udp"] * (n // 2),
                         LABEL_COL: ["x"] * n, CLASS_COL: ["Normal", "DoS"] * (n // 2)})


def test_signed_log1p():
    x = np.array([-1e6, -1.0, 0.0, 1.0, 1e6])
    out = signed_log1p(x)
    assert out[2] == 0 and np.allclose(out, -out[::-1])
    assert np.isclose(out[3], np.log(2))


def test_default_unchanged():
    train = _frame([0.0, 1.0, 10.0, 1e6])
    a = preprocess(train, train, train, SPEC)
    b = preprocess(train, train, train, SPEC, None)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.isclose(a.X_train[:, 0].std(), 1.0)


def test_log_then_scale_on_train_only():
    train, test = _frame([0.0, 1.0, 10.0, 1e6]), _frame([1e9, 0.0])
    s = preprocess(train, train, test, SPEC, "signed_log1p")
    assert np.allclose(s.X_train[:, :2].mean(axis=0), 0, atol=1e-9)
    expected = (np.log1p(1e9) - np.log1p(train["rate"]).mean()) / np.log1p(train["rate"]).std(ddof=0)
    assert np.isclose(s.X_test[0, 0], expected)
    assert s.feature_names[:2] == ["rate", "flag"]
    assert s.X_train.shape[1] == 4


def test_saved_transformer_reloads():
    train = _frame([0.0, 1.0, 10.0, 1e6])
    s = preprocess(train, train, train, SPEC, "signed_log1p")
    buf = io.BytesIO()
    joblib.dump(s.transformer, buf)
    buf.seek(0)
    again = joblib.load(buf)
    X = train.drop(columns=[LABEL_COL, CLASS_COL])
    assert np.allclose(again.transform(X), s.X_train)


def test_unknown_transform():
    train = _frame([0.0, 1.0])
    with pytest.raises(ValueError, match="numeric_transform"):
        preprocess(train, train, train, SPEC, "log")
