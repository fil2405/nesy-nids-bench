from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras

from nesybench.data.base import REPO_ROOT
from nesybench.logic.kb import antecedents, load_kb, target_index
from nesybench.logic.semantics import axiom_degrees, check_p, domain_degrees, sat_agg
from nesybench.models.mlp import build_mlp
from nesybench.models.sampling import sample_batches
from nesybench.models.selection import EarlyStopper, validation_macro_f1
from nesybench.timing import EpochClock, Stopwatch


class LTNFull:
    """LTN-sup plus the domain axioms of a KB; same loop, network and sampler."""

    def fit(self, splits, config):
        network = config["network"]
        training = config["training"]

        if training.get("imbalance", "none") != "none":
            raise ValueError("ltn_full takes no class weights")
        if training["optimizer"] != "adam":
            raise ValueError(f"unsupported optimizer {training['optimizer']!r}")
        if not config.get("kb"):
            raise ValueError("ltn_full needs a kb path in the config")

        seed = config["seed"]
        tf_keras.utils.set_random_seed(seed)
        n_classes = len(splits.class_names)
        self.batch_size = training["batch_size"]
        power = training.get("sampling_power", 0.0)
        min_per_class = training.get("min_per_class", 2)
        weight = float(training.get("domain_weight", 0.25))
        crisp = bool(training.get("crisp", False))
        s_scale = float(training.get("s_scale", 1.0))
        p = check_p(training.get("p", 2))
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"domain_weight must be in [0, 1], got {weight}")

        kb_path = Path(config["kb"])
        kb = load_kb(kb_path if kb_path.is_absolute() else REPO_ROOT / kb_path)
        targets = target_index(kb, splits.class_names)
        A_train = antecedents(kb, splits.raw("train"), crisp=crisp, s_scale=s_scale).astype("float32")

        self.model = build_mlp(
            input_dim=splits.X_train.shape[1],
            n_classes=n_classes,
            hidden_units=network["hidden_units"],
            dropout=network["dropout"],
            activation=network["activation"],
            seed=seed,
        )
        optimizer = tf_keras.optimizers.Adam(learning_rate=training["learning_rate"])

        @tf.function(reduce_retracing=True)
        def step(x, y, a):
            with tf.GradientTape() as tape:
                probs = self.model(x, training=True)  # one forward pass, all axioms read it
                sat_sup = sat_agg(axiom_degrees(probs, y, n_classes, p), p).tensor
                sat_dom = sat_agg(domain_degrees(probs, a, targets, p), p).tensor
                loss = 1.0 - ((1.0 - weight) * sat_sup + weight * sat_dom)
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        X = tf.constant(splits.X_train, dtype=tf.float32)
        y = tf.constant(splits.y_train, dtype=tf.int32)
        A = tf.constant(A_train, dtype=tf.float32)
        stopper = EarlyStopper(training["early_stopping_patience"],
                               training.get("selection_window", 1))
        clock = EpochClock()
        val_scores, val_score_seconds, train_loss = [], [], []

        for epoch in range(training["max_epochs"]):
            clock.begin()
            batches = sample_batches(splits.y_train, n_classes, self.batch_size,
                                     power, min_per_class, seed=seed * 10_000 + epoch)
            losses = [step(tf.gather(X, idx), tf.gather(y, idx), tf.gather(A, idx))
                      for idx in batches]
            clock.end()  # epoch time = training only, scoring is timed apart
            loss = float(np.mean([v.numpy() for v in losses]))
            if not np.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            train_loss.append(loss)

            with Stopwatch() as watch:
                score = validation_macro_f1(self.model, splits)
            val_score_seconds.append(watch.seconds)
            val_scores.append(score)
            if stopper.update(score, self.model.get_weights()):
                break

        stopper.close(self.model.get_weights())
        self.model.set_weights(stopper.best_weights)

        return {
            "n_parameters": self.model.count_params(),
            "epochs_run": len(val_scores),
            "best_epoch": stopper.best_epoch,
            "best_val_macro_f1": stopper.best_raw_score,
            "selection_score": stopper.best_score,
            "selection_window": training.get("selection_window", 1),
            "val_macro_f1_history": val_scores,
            "train_loss_history": train_loss,
            "epoch_seconds": clock.seconds,
            "val_score_seconds": val_score_seconds,
            "sampling_power": power,
            "min_per_class": min_per_class,
            "p": p,
            "class_weights": None,
            "class_weight_power": None,
            "kb_version": kb.version,
            "kb_ids": kb.ids,
            "domain_weight": weight,
            "crisp": crisp,
            "s_scale": s_scale,
            "antecedent_mean_train": A_train.mean(axis=0).tolist(),
        }

    def predict_proba(self, splits, part):
        X, _ = splits.part(part)
        return self.model.predict(X, batch_size=self.batch_size, verbose=0)  # type: ignore
