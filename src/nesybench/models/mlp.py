from pathlib import Path

import tf_keras


def save_mlp(model, directory):
    model.save(Path(directory) / "model.keras")


def _initializer(seed, layer_index):
    if seed is None:
        return tf_keras.initializers.GlorotUniform()
    return tf_keras.initializers.GlorotUniform(seed=seed + layer_index)


def build_mlp(input_dim, n_classes, hidden_units, dropout, activation, seed=None):
    layers = [tf_keras.layers.Input(shape=(input_dim,))]

    for i, units in enumerate(hidden_units):
        layers.append(tf_keras.layers.Dense(
            units, activation=activation, kernel_initializer=_initializer(seed, i))) # type: ignore
        layers.append(tf_keras.layers.Dropout(dropout))

    layers.append(tf_keras.layers.Dense(
        n_classes, activation="softmax",
        kernel_initializer=_initializer(seed, len(hidden_units)))) # type: ignore

    return tf_keras.Sequential(layers)