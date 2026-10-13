import time

import pytest

from nesybench.timing import EpochClock, Stopwatch


def test_stopwatch_measures_elapsed_time():
    with Stopwatch() as sw:
        time.sleep(0.02)
    assert sw.seconds >= 0.015


def test_stopwatch_writes_into_a_dict():
    timings = {}
    with Stopwatch(timings, "fit_s"):
        pass
    assert timings["fit_s"] >= 0


def test_stopwatch_records_time_even_if_the_block_fails():
    timings = {}
    with pytest.raises(ZeroDivisionError):
        with Stopwatch(timings, "fit_s"):
            1 / 0
    assert "fit_s" in timings


def test_epoch_clock_has_one_value_per_epoch():
    clock = EpochClock()
    for _ in range(3):
        clock.begin()
        clock.end()
    assert len(clock.seconds) == 3
    assert all(s >= 0 for s in clock.seconds)


def test_epoch_clock_end_before_begin_is_an_error():
    with pytest.raises(RuntimeError):
        EpochClock().end()
