"""nesybench: benchmark of LTN against matched ML/DL baselines for NIDS."""
import os

# Must happen before any tensorflow import: ltn 2.1 needs Keras 2 (tf_keras).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
