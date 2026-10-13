"""Wall-clock timers."""
from time import perf_counter


class Stopwatch:
    """Context manager; optionally writes the elapsed time to sink[key], also on error."""

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
    """Per-epoch durations, usable from a Keras callback or a custom loop."""

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
