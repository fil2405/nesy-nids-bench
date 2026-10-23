"""Random Forest and XGBoost behind the common model interface."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from nesybench.metrics import compute_metrics
from nesybench.models.selection import EarlyStopper
from nesybench.models.weights import class_weights as class_weights_for
from nesybench.timing import EpochClock, Stopwatch


def _sample_weights(training, y_train, n_classes):
    """(sample_weight, class_weights, power); all None without class weights."""
    imbalance = training.get("imbalance", "none")
    if imbalance not in ("none", "class_weight"):
        raise ValueError(f"unsupported value in imbalance {imbalance!r}")
    if imbalance == "none":
        return None, None, None
    power = training.get("class_weight_power", 1.0)
    counts = np.bincount(y_train, minlength=n_classes)
    per_class = class_weights_for(counts, power)
    class_weights = {i: float(w) for i, w in enumerate(per_class)}
    return per_class[y_train], class_weights, power


class RandomForest:

    def fit(self, splits, config):
        params = config.get("params", {})
        self.n_classes = len(splits.class_names)
        sample_weight, class_weights, power = _sample_weights(
            config["training"], splits.y_train, self.n_classes)

        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth"),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=params.get("n_jobs", -1),
            random_state=config["seed"],
        )
        with Stopwatch() as watch:
            self.model.fit(splits.X_train, splits.y_train, sample_weight=sample_weight)

        return {
            "n_trees": len(self.model.estimators_),
            "n_nodes": int(sum(t.tree_.node_count for t in self.model.estimators_)),
            "fit_seconds": watch.seconds,
            "class_weights": class_weights,
            "class_weight_power": power,
        }

    def predict_proba(self, splits, part):
        X, _ = splits.part(part)
        out = np.zeros((len(X), self.n_classes))
        out[:, self.model.classes_] = self.model.predict_proba(X)  # absent class stays 0
        return out

    def save(self, directory):
        joblib.dump(self.model, Path(directory) / "model.joblib", compress=3)


class XGBoost:

    def fit(self, splits, config):
        import xgboost as xgb  # lazy: rf must not need it

        params = config.get("params", {})
        training = config["training"]
        self.n_classes = len(splits.class_names)
        sample_weight, class_weights, power = _sample_weights(
            training, splits.y_train, self.n_classes)

        dtrain = xgb.DMatrix(splits.X_train, label=splits.y_train, weight=sample_weight)
        dval = xgb.DMatrix(splits.X_val, label=splits.y_val)  # unweighted, as in MLP-CE

        window = training.get("selection_window", 1)
        stopper = EarlyStopper(training["early_stopping_patience"], window)
        score_seconds = []

        def val_macro_f1(predt, _dmatrix):
            with Stopwatch() as watch:
                proba = np.asarray(predt).reshape(len(splits.y_val), -1)
                score = compute_metrics(
                    splits.y_val, proba.argmax(axis=1), splits.class_names)["macro_f1"]
            score_seconds.append(watch.seconds)
            return "val_macro_f1", score

        class Selection(xgb.callback.TrainingCallback):

            def __init__(self):
                super().__init__()
                self.clock = EpochClock()

            def before_iteration(self, model, epoch, evals_log):
                self.clock.begin()
                return False

            def after_iteration(self, model, epoch, evals_log):
                self.clock.end()  # includes the scoring, taken out below
                return stopper.update(evals_log["val"]["val_macro_f1"][-1], None)

        selection = Selection()
        with Stopwatch() as watch:
            self.booster = xgb.train(
                {
                    "objective": "multi:softprob",
                    "num_class": self.n_classes,
                    "tree_method": "hist",
                    "eta": params.get("learning_rate", 0.1),
                    "max_depth": params.get("max_depth", 6),
                    "min_child_weight": params.get("min_child_weight", 1),
                    "subsample": params.get("subsample", 1.0),
                    "colsample_bytree": params.get("colsample_bytree", 1.0),
                    "seed": config["seed"],
                    "disable_default_eval_metric": 1,
                },
                dtrain,
                num_boost_round=training["max_rounds"],
                evals=[(dval, "val")],
                custom_metric=val_macro_f1,
                callbacks=[selection],
                verbose_eval=False,
            )
        stopper.close(None)
        self.best_round = stopper.best_epoch  # rounds kept at predict time

        return {
            "n_trees": self.best_round * self.n_classes,
            "rounds_run": len(stopper.scores),
            "best_round": self.best_round,
            "best_val_macro_f1": stopper.best_raw_score,  # raw score of the chosen round
            "selection_score": stopper.best_score,        # the criterion: trailing mean
            "selection_window": window,
            "val_macro_f1_history": stopper.scores,
            "round_seconds": [t - s for t, s in zip(selection.clock.seconds, score_seconds)],
            "val_score_seconds": score_seconds,
            "fit_seconds": watch.seconds,
            "class_weights": class_weights,
            "class_weight_power": power,
        }

    def predict_proba(self, splits, part):
        import xgboost as xgb

        X, _ = splits.part(part)
        return self.booster.predict(xgb.DMatrix(X), iteration_range=(0, self.best_round))

    def save(self, directory):
        self.booster.set_attr(best_round=str(self.best_round))  # rounds kept at predict time
        self.booster.save_model(str(Path(directory) / "model.json"))
