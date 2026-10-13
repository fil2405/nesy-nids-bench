import tf_keras

from nesybench.models.mlp import build_mlp
from nesybench.models.selection import EarlyStopper, validation_macro_f1
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
        self.model.set_weights(self.stopper.best_weights)


class MLPCE:

    def fit(self, splits, config):
        network = config["network"]
        training = config["training"]
        if training["optimizer"] != "adam":
            raise ValueError(f"unsupported optimizer {training['optimizer']!r}")

        tf_keras.utils.set_random_seed(config["seed"])
        self.batch_size = training["batch_size"]
        self.model = build_mlp(
            input_dim=splits.X_train.shape[1],
            n_classes=len(splits.class_names),
            hidden_units=network["hidden_units"],
            dropout=network["dropout"],
            activation=network["activation"],
            seed=config["seed"],
        )
        self.model.compile(
            optimizer=tf_keras.optimizers.Adam(learning_rate=training["learning_rate"]),
            loss="sparse_categorical_crossentropy",
        )

        stopper = EarlyStopper(training["early_stopping_patience"])
        selection = _EarlyStopping(splits, stopper)
        self.model.fit(
            splits.X_train, splits.y_train,
            batch_size=self.batch_size,
            epochs=training["max_epochs"],
            shuffle=True,
            verbose=0, # type: ignore
            callbacks=[selection],
        )

        return {
            "n_parameters": self.model.count_params(),
            "epochs_run": len(selection.val_scores),
            "best_epoch": stopper.best_epoch,
            "best_val_macro_f1": stopper.best_score,
            "val_macro_f1_history": selection.val_scores,
            "epoch_seconds": selection.clock.seconds,
            "val_score_seconds": selection.val_score_seconds,
        }

    def predict_proba(self, splits, part):
        X, _ = splits.part(part)
        return self.model.predict(X, batch_size=self.batch_size, verbose=0) # type: ignore