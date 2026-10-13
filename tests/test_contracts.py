"""No dataset-specific numbers or class names in models, metrics and runner."""
import re
from pathlib import Path

import nesybench

PACKAGE = Path(nesybench.__file__).parent
FORBIDDEN = re.compile(r"\b(122|Normal|DoS|Probe|R2L|U2R)\b")


def _generic_files():
    return [*sorted((PACKAGE / "models").glob("*.py")), PACKAGE / "metrics.py", PACKAGE / "run.py"]


def test_no_dataset_literals_in_generic_code():
    offenders = [
        f"{path.name}:{number}: {line.strip()}"
        for path in _generic_files()
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if FORBIDDEN.search(line)
    ]
    assert not offenders, offenders
