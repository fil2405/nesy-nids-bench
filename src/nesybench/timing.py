"""Wall-clock measurements shared by the runner and by every training loop."""
from time import perf_counter


class Stopwatch:
    """Time a block: `with Stopwatch() as sw: ...` then `sw.seconds`.

    With a dict and a key, the time is also stored there:
    `with Stopwatch(timings, "fit_s"): ...`. It is recorded even if the block fails.
    """

    def __init__(self, sink: dict | None = None, key: str | None = None):
        self.seconds = 0.0
        self._sink = sink
        self._key = key

    def __enter__(self):
        self._start = perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.seconds = perf_counter() - self._start
        if self._sink is not None:
            self._sink[self._key] = self.seconds
        return False


class EpochClock:
    """Seconds per epoch, for a Keras callback or a custom loop alike.

    Call begin() when an epoch starts and end() when it ends; `seconds` has
    one value per epoch.
    """

    def __init__(self):
        self.seconds: list[float] = []
        self._start: float | None = None

    def begin(self) -> None:
        self._start = perf_counter()

    def end(self) -> float:
        if self._start is None:
            raise RuntimeError("end() called before begin()")
        elapsed = perf_counter() - self._start
        self.seconds.append(elapsed)
        self._start = None
        return elapsed
