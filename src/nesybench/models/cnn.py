"""1D-CNN baseline: the MLP-CE training loop with a convolutional network over the feature vector."""
import tf_keras

from nesybench.models.mlp import _initializer
from nesybench.models.mlp_ce import MLPCE


def build_cnn(input_dim, n_classes, conv_filters, kernel_size, hidden_units, dropout, activation,
              seed=None):
    layers = [tf_keras.layers.Input(shape=(input_dim,)),
              tf_keras.layers.Reshape((input_dim, 1))]
    i = 0
    for filters in conv_filters:
        layers.append(tf_keras.layers.Conv1D(filters, kernel_size, padding="same", activation=activation,
                                             kernel_initializer=_initializer(seed, i)))
        i += 1
    layers += [tf_keras.layers.MaxPooling1D(2), tf_keras.layers.Flatten(),
               tf_keras.layers.Dropout(dropout)]
    for units in hidden_units:
        layers.append(tf_keras.layers.Dense(units, activation=activation,
                                            kernel_initializer=_initializer(seed, i)))
        layers.append(tf_keras.layers.Dropout(dropout))
        i += 1
    layers.append(tf_keras.layers.Dense(n_classes, activation="softmax",
                                        kernel_initializer=_initializer(seed, i)))
    return tf_keras.Sequential(layers)


class CNN1D(MLPCE):

    def build(self, input_dim, n_classes, network, seed):
        return build_cnn(input_dim, n_classes, network["conv_filters"], network["kernel_size"],
                         network["hidden_units"], network["dropout"], network["activation"], seed)
