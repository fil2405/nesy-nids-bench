"""Print versions, GPU and RAM, and run a tiny ltn formula."""
import os
import platform
import sys

# ltn 2.1 needs Keras 2; must be set before importing tensorflow.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def main() -> int:
    print(f"Python      {platform.python_version()}  ({platform.platform()})")

    try:
        import psutil

        print(f"RAM         {psutil.virtual_memory().total / 2**30:.1f} GB")
    except ImportError:
        print("RAM         psutil not installed")

    import tensorflow as tf

    print(f"TensorFlow  {tf.__version__}")
    print(f"tf.keras    {tf.keras.__name__}  (must contain tf_keras for ltn)")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU         {gpus if gpus else 'none (CPU only)'}")

    try:
        import numpy as np
        import ltn

        x = ltn.Variable("x", np.random.rand(8, 3).astype("float32"))
        P = ltn.Predicate.MLP([3], hidden_layer_sizes=(4,))
        Forall = ltn.Wrapper_Quantifier(
            ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall"
        )
        sat = Forall(x, P(x)).tensor.numpy()
        print(f"ltn         OK, test formula satisfaction = {sat:.3f}")
    except Exception as exc:
        print(f"ltn         FAILED: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
