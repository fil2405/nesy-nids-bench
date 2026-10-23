"""Rule-only arm: the KB axioms alone, nothing is learned."""
import json
import shutil
from pathlib import Path

import numpy as np

from nesybench.logic.c5 import THRESHOLD
from nesybench.logic.kb import antecedents, load_kb, target_index
from nesybench.data.base import REPO_ROOT

DEFAULT_CLASS = 0  # benign class, when no rule fires


class RuleOnly:

    def fit(self, splits, config):
        if not config.get("kb"):
            raise ValueError("rule_only needs a kb path in the config")
        path = Path(config["kb"])
        self.kb_path = path if path.is_absolute() else REPO_ROOT / path
        self.kb = load_kb(self.kb_path)
        self.n_classes = len(splits.class_names)
        self.targets = target_index(self.kb, splits.class_names)
        fired = self._fired(splits.raw("train"))
        return {
            "kb_version": self.kb.version,
            "kb_ids": self.kb.ids,
            "train_covered": int(fired.any(axis=1).sum()),
            "train_conflicts": int(self._conflicts(fired).sum()),
        }

    def predict_proba(self, splits, part):
        fired = self._fired(splits.raw(part))
        pred = np.full(len(fired), DEFAULT_CLASS)
        # first axiom in KB order wins a conflict
        for j in reversed(range(len(self.targets))):
            pred[fired[:, j]] = self.targets[j]
        out = np.zeros((len(pred), self.n_classes))
        out[np.arange(len(pred)), pred] = 1.0
        return out

    def save(self, directory):
        directory = Path(directory)
        shutil.copy(self.kb_path, directory / "kb.yaml")
        rules = {
            "kb_version": self.kb.version,
            "default_class": DEFAULT_CLASS,
            "targets": [int(t) for t in self.targets],
        }
        (directory / "rules.json").write_text(json.dumps(rules, indent=2))

    def _fired(self, raw):
        return antecedents(self.kb, raw, crisp=True) >= THRESHOLD

    def _conflicts(self, fired):
        t = np.where(fired, self.targets, -1)
        first = np.where(fired.any(axis=1), t[np.arange(len(t)), fired.argmax(axis=1)], -1)
        return ((t != -1) & (t != first[:, None])).any(axis=1)
