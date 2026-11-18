"""Scale and resource tests for numeric distribution profiles.

These tests assert the properties that matter for production use rather than
wall-clock times: the batch source is traversed exactly once, the retained
state stays small, the compact JSON stays under the size budget, peak memory
does not grow with the number of rows, and the reference data can be released
before the profile is used for comparison.
"""

from __future__ import annotations

import gc
import json
import tracemalloc
import weakref

import numpy as np
import pandas as pd
import pytest

from pandera.numeric_profile import NumericProfile

BATCH_SIZE = 4096
EPSILON = 0.01


def _generate(kind, n):
    rng = np.random.default_rng(7)
    if kind == "uniform":
        return rng.random(n).astype("float64")
    if kind == "duplicates":
        return rng.integers(0, 1000, n).astype("int64")
    if kind == "skewed":
        return np.concatenate(
            [np.zeros(n - 10, dtype="float64"), np.arange(10, dtype="float64")]
        )
    raise KeyError(kind)


class _CountingBatches:
    """A single-pass batch source that records how often it is iterated."""

    def __init__(self, values, batch_size):
        self._values = values
        self._batch_size = batch_size
        self.iterations = 0
        self.batches = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise RuntimeError("batch source iterated more than once")
        for start in range(0, len(self._values), self._batch_size):
            self.batches += 1
            yield pd.Series(self._values[start : start + self._batch_size])


@pytest.mark.parametrize("kind", ["uniform", "duplicates", "skewed"])
@pytest.mark.parametrize("n", [100_000, 1_000_000])
def test_scale_single_traversal_and_compact_state(kind, n):
    values = _generate(kind, n)
    source = _CountingBatches(values, BATCH_SIZE)

    tracemalloc.start()
    try:
        profile = NumericProfile.from_batches(source, epsilon=EPSILON)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert source.iterations == 1
    assert source.batches == (n + BATCH_SIZE - 1) // BATCH_SIZE
    assert profile.n == n

    # Retained state is a few dozen to a few hundred tuples, independent of n.
    state = profile.to_dict()
    assert len(state["values"]) < 1000

    # Compact JSON stays well under the 256 KiB budget at epsilon=0.01.
    encoded = profile.to_json().encode("utf-8")
    assert len(encoded) < 256 * 1024

    # Peak memory is dominated by one batch plus the small state; it must not
    # scale with the total number of rows (compare 100k versus 1M).
    assert peak < 16 * 1024 * 1024

    # The bounds remain valid on a coarse grid.
    ordered = np.sort(values)
    for t in np.linspace(values.min(), values.max(), 25):
        lower, upper = profile.cdf_bounds(float(t))
        cdf = np.searchsorted(ordered, t, side="right") / n
        assert 0.0 <= lower <= cdf + 1e-9
        assert cdf - 1e-9 <= upper <= 1.0
        assert upper - lower <= 2 * EPSILON + 1e-12


@pytest.mark.parametrize("n", [100_000, 1_000_000])
def test_state_size_is_independent_of_row_count(n):
    small = NumericProfile.from_batches(
        [pd.Series(_generate("uniform", 100_000))], epsilon=EPSILON
    )
    large = NumericProfile.from_batches(
        [pd.Series(_generate("uniform", n))], epsilon=EPSILON
    )
    small_bytes = len(small.to_json().encode("utf-8"))
    large_bytes = len(large.to_json().encode("utf-8"))
    # Both fit in the same order of magnitude; the large one must not grow
    # linearly with rows.
    assert large_bytes < 4 * small_bytes + 4096


def test_reference_data_can_be_released_before_comparison():
    reference = pd.Series(np.arange(50_000, dtype="float64"))
    tracker = weakref.ref(reference)
    profile = NumericProfile.from_batches([reference], epsilon=EPSILON)
    del reference
    gc.collect()
    assert tracker() is None  # observations were not retained
    follow_up = pd.Series(np.arange(50_000, dtype="float64"))
    result = profile.compare(follow_up, tolerance=0.02)
    assert result["status"] == "pass"


def test_compare_does_not_retain_the_batch():
    profile = NumericProfile.from_batches(
        [pd.Series(np.arange(10_000, dtype="float64"))], epsilon=EPSILON
    )
    batch = pd.Series(np.arange(10_000, dtype="float64"))
    tracker = weakref.ref(batch)
    profile.compare(batch, tolerance=0.02)
    del batch
    gc.collect()
    assert tracker() is None
    # The profile's own state is unchanged by the comparison.
    assert profile.n == 10_000
    json.dumps(profile.to_dict())
