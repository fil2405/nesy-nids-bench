import tf_keras
import numpy as np

from nesybench.models.mlp import build_mlp, save_mlp
from nesybench.models.selection import EarlyStopper, predict_batches, validation_macro_f1
from nesybench.models.weights import class_weights as class_weights_for
from nesybench.timing import EpochClock, Stopwatch


class _EarlyStopping(tf_keras.callbacks.Callback):

    def __init__(self, splits, stopper):
        super().__init__()
        self.splits = splits
        self.stopper = stopper
        self.clock = EpochClock()
        self.val_scores = []
        self.val_score_seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self.clock.begin()

    def on_epoch_end(self, epoch, logs=None):
        self.clock.end()  # epoch time = training only, scoring is timed apart
        with Stopwatch() as watch:
            score = validation_macro_f1(self.model, self.splits)
        self.val_score_seconds.append(watch.seconds)
        self.val_scores.append(score)
        assert self.model is not None
        if self.stopper.update(score, self.model.get_weights()):
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        assert self.model is not None
        self.stopper.close(self.model.get_weights())
        self.model.set_weights(self.stopper.best_weights)


class MLPCE:

    def build(self, input_dim, n_classes, network, seed):
        return build_mlp(input_dim=input_dim, n_classes=n_classes,
                         hidden_units=network["hidden_units"], dropout=network["dropout"],
                         activation=network["activation"], seed=seed)

    def fit(self, splits, config):
        network = config["network"]
        training = config["training"]

        imbalance = training.get("imbalance", "none")
        if imbalance not in ("none", "class_weight"):
            raise ValueError(f"unsupported value in imbalance {imbalance!r}")

        if training["optimizer"] != "adam":
            raise ValueError(f"unsupported optimizer {training['optimizer']!r}")

        tf_keras.utils.set_random_seed(config["seed"])
        self.batch_size = training["batch_size"]
        self.model = self.build(splits.X_train.shape[1], len(splits.class_names), network,
                                config["seed"])
        self.model.compile(
            optimizer=tf_keras.optimizers.Adam(learning_rate=training["learning_rate"]),
            loss="sparse_categorical_crossentropy",
        )

        class_weights = None

        power = None

        if imbalance == 'class_weight':
            power = training.get("class_weight_power", 1.0)
            counts = np.bincount(splits.y_train, minlength=len(splits.class_names))
            weight = class_weights_for(counts, power)
            class_weights = {i : float(weights) for i, weights in enumerate(weight)}

        window = training.get("selection_window", 1)
        stopper = EarlyStopper(training["early_stopping_patience"], window)
        selection = _EarlyStopping(splits, stopper)
        self.model.fit(
            splits.X_train, splits.y_train,
            batch_size=self.batch_size,
            epochs=training["max_epochs"],
            shuffle=True,
            verbose=0, # type: ignore
            callbacks=[selection],
            class_weight=class_weights if imbalance == "class_weight" else None
        )

        return {
            "n_parameters": self.model.count_params(),
            "epochs_run": len(selection.val_scores),
            "best_epoch": stopper.best_epoch,
            "best_val_macro_f1": stopper.best_raw_score,  # raw score of the chosen epoch
            "selection_score": stopper.best_score,        # the criterion: trailing mean
            "selection_window": window,
            "val_macro_f1_history": selection.val_scores,
            "epoch_seconds": selection.clock.seconds,
            "val_score_seconds": selection.val_score_seconds,
            "class_weights": class_weights,
            "class_weight_power": power,
        }

    def predict_proba(self, splits, part):
        X, _ = splits.part(part)
        return predict_batches(self.model, X, self.batch_size)

    def save(self, directory):
        save_mlp(self.model, directory)