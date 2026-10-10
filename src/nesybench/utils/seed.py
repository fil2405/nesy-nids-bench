"""Global seeding, so that a run is fully determined by its configuration and seed."""
import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and TensorFlow.

    deterministic=True also asks TensorFlow for deterministic kernels:
    slower on GPU, but two runs with the same seed give the same numbers.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()
