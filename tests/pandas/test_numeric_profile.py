"""Tests for deterministic numeric distribution profiles."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pandas as pd
import pytest

from pandera.errors import (
    DistributionCheckError,
    DistributionUncertainError,
)
from pandera.numeric_profile import NumericProfile


def _brute_cdf(sorted_values, t):
    return sum(1 for x in sorted_values if x <= t) / len(sorted_values)


def _probe_thresholds(values):
    """Every jump point of the empirical CDF plus points just around them."""
    distinct = sorted(set(values))
    probes = list(distinct)
    probes.append(distinct[0] - 1)
    probes.append(distinct[-1] + 1)
    for low, high in zip(distinct, distinct[1:]):
        if isinstance(low, int) and isinstance(high, int):
            if high - low > 1:
                probes.append(low + 1)
                probes.append(high - 1)
        else:
            probes.append((low + high) / 2.0)
    return probes


def _assert_bounds_hold(profile, values, epsilon):
    ordered = sorted(values)
    n = len(values)
    assert profile.n == n
    for t in _probe_thresholds(values):
        lower, upper = profile.cdf_bounds(t)
        cdf = _brute_cdf(ordered, t)
        assert 0.0 <= lower <= cdf + 1e-12
        assert cdf - 1e-12 <= upper <= 1.0
        assert upper - lower <= 2 * epsilon + 1e-12


def _profile_from(values, *, dtype="float64", batches=None, epsilon=0.01):
    if batches is None:
        batches = [values]
    return NumericProfile.from_batches(
        [pd.Series(batch, dtype=dtype) for batch in batches],
        epsilon=epsilon,
    )


SCENARIOS = {
    "uniform": ([random.Random(1).random() for _ in range(3000)], "float64"),
    "duplicates": (
        [random.Random(2).randrange(20) for _ in range(3000)],
        "int64",
    ),
    "same_min_max": ([0.0] * 1500 + [1.0] * 1500, "float64"),
    "all_same": ([7.0] * 500, "float64"),
    "skewed": ([0.0] * 2900 + [float(i) for i in range(100)], "float64"),
    "negative": (
        [-random.Random(3).random() * 1e6 for _ in range(2000)],
        "float64",
    ),
    "adjacent_big_ints": ([2**53 + i for i in range(1500)], "int64"),
    "repeated_big_ints": ([2**53 + i % 3 for i in range(1500)], "int64"),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.1])
def test_cdf_bounds_hold_for_all_jump_points(name, epsilon):
    """The deterministic bound holds at every jump point and its interior."""
    values, dtype = SCENARIOS[name]
    profile = _profile_from(values, dtype=dtype, epsilon=epsilon)
    _assert_bounds_hold(profile, values, epsilon)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_bounds_are_layout_and_order_invariant(name):
    """Batch layout and row order may change the summary, not the guarantee."""
    values, dtype = SCENARIOS[name]
    layouts = {
        "single": [values],
        "sevens": [values[i : i + 7] for i in range(0, len(values), 7)],
        "one": [[value] for value in values],
    }
    reversed_values = values[::-1]
    layouts["reversed"] = [
        reversed_values[i : i + 113] for i in range(0, len(values), 113)
    ]
    for layout in layouts.values():
        profile = _profile_from(values, dtype=dtype, batches=layout)
        _assert_bounds_hold(profile, values, 0.01)


def test_empty_batches_are_skipped():
    profile = NumericProfile.from_batches(
        [
            pd.Series([], dtype="float64"),
            pd.Series([1.0, 2.0]),
            pd.Series([], dtype="Int64"),
            pd.Series([3.0]),
        ]
    )
    assert profile.n == 3
    assert profile.cdf_bounds(2.0) == (2 / 3, 2 / 3)


def test_no_valid_observations_raises_value_error():
    with pytest.raises(ValueError, match="no valid"):
        NumericProfile.from_batches(
            [pd.Series([], dtype="float64"), pd.Series([np.nan, np.nan])]
        )


def test_missing_values_are_ignored_and_multiplicity_counted():
    profile = NumericProfile.from_batches(
        [pd.Series([1.0, 1.0, 2.0, np.nan, 3.0])]
    )
    assert profile.n == 4
    lower, upper = profile.cdf_bounds(1.0)
    assert (lower, upper) == (0.5, 0.5)


def test_adjacent_integers_near_2_53_stay_distinct():
    values = [2**53, 2**53 + 1, 2**53 + 2]
    profile = NumericProfile.from_batches([pd.Series(values, dtype="int64")])
    assert profile.n == 3
    assert profile.cdf_bounds(2**53) == (1 / 3, 1 / 3)
    assert profile.cdf_bounds(2**53 + 1) == (2 / 3, 2 / 3)
    assert profile.cdf_bounds(2**53 + 2) == (1.0, 1.0)


@pytest.mark.parametrize(
    "values, dtype",
    [
        ([1, 2, 3], "int32"),
        ([1.0, 2.0], "float32"),
        ([True, False], "bool"),
        (["a", "b"], "object"),
        ([1.0, 2.0], "object"),
    ],
)
def test_unsupported_dtypes_are_rejected(values, dtype):
    with pytest.raises(TypeError, match="only support"):
        NumericProfile.from_batches([pd.Series(values, dtype=dtype)])


def test_non_series_batch_is_rejected():
    with pytest.raises(TypeError, match="pandas.Series"):
        NumericProfile.from_batches([[1, 2, 3]])
    with pytest.raises(TypeError, match="single"):
        NumericProfile.from_batches(pd.Series([1, 2, 3]))


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_non_finite_observations_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        NumericProfile.from_batches([pd.Series([1.0, bad])])
    with pytest.raises(ValueError, match="finite"):
        NumericProfile.from_batches([pd.Series([1.0, bad], dtype="Float64")])


@pytest.mark.parametrize(
    "epsilon",
    [0, -0.01, 0.11, True, np.bool_(True), "0.01", math.nan, math.inf],
)
def test_invalid_epsilon_is_rejected(epsilon):
    with pytest.raises((TypeError, ValueError)):
        NumericProfile.from_batches([pd.Series([1.0])], epsilon=epsilon)


@pytest.mark.parametrize("threshold", [math.inf, -math.inf, math.nan])
def test_non_finite_threshold_is_rejected(threshold):
    profile = _profile_from([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        profile.cdf_bounds(threshold)


@pytest.mark.parametrize("threshold", [True, "1", None])
def test_non_numeric_threshold_is_rejected(threshold):
    profile = _profile_from([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="real number"):
        profile.cdf_bounds(threshold)


def test_extreme_finite_floats():
    values = [-1e308, -1e-300, 0.0, 1e-300, 1e308]
    profile = _profile_from(values)
    _assert_bounds_hold(profile, values, 0.01)
    assert profile.cdf_bounds(-1e308) == (0.2, 0.2)


@pytest.mark.parametrize("dtype", ["Int64", "Float64"])
def test_nullable_dtypes(dtype):
    series = pd.Series([1, None, 3, 3], dtype=dtype)
    profile = NumericProfile.from_batches([series])
    assert profile.n == 3
    # valid observations are [1, 3, 3]: F(1) = 1/3 and F(3) = 1.
    assert profile.cdf_bounds(1) == (pytest.approx(1 / 3),) * 2
    assert profile.cdf_bounds(3) == (1.0, 1.0)
    assert profile.cdf_bounds(2) == (pytest.approx(1 / 3),) * 2


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _true_distance(reference, follow_up):
    ordered_ref = sorted(reference)
    ordered_new = sorted(follow_up)
    best = 0.0
    for t in sorted(set(reference) | set(follow_up)):
        distance = abs(_brute_cdf(ordered_ref, t) - _brute_cdf(ordered_new, t))
        best = max(best, distance)
    return best


COMPARE_SCENARIOS = [
    ("identical", list(range(400)), list(range(400))),
    ("frequency_only", [0] * 200 + [1] * 200, [0] * 80 + [1] * 320),
    (
        "same_min_max_middle_mass",
        [0.0] * 200 + [0.3] * 300 + [1.0] * 200,
        [0.0] * 200 + [0.7] * 300 + [1.0] * 200,
    ),
    ("shifted", list(range(400)), list(range(200, 600))),
    ("disjoint", list(range(100)), list(range(1000, 1100))),
    ("all_identical", [5] * 300, [5] * 300),
    ("skew_vs_uniform", [0] * 390 + list(range(10)), list(range(400))),
]


@pytest.mark.parametrize("name, reference, follow_up", COMPARE_SCENARIOS)
def test_compare_distance_bounds(name, reference, follow_up):
    profile = _profile_from(reference, dtype="float64")
    result = profile.compare(
        pd.Series(follow_up, dtype="float64"), tolerance=0.5
    )
    assert set(result) == {"distance_lower", "distance_upper", "status"}
    json.dumps(result)
    distance = _true_distance(reference, follow_up)
    lower = result["distance_lower"]
    upper = result["distance_upper"]
    assert 0.0 <= lower <= distance + 1e-12
    assert distance - 1e-12 <= upper <= 1.0
    assert upper - lower <= 4 * 0.01 + 1e-12


def test_compare_status_thresholds():
    reference = list(range(1000))
    follow_up = list(range(1000))
    for index in range(100):
        follow_up[index] += 10000  # true distance == 0.1
    profile = _profile_from(reference, dtype="float64")
    results = {
        tolerance: profile.compare(
            pd.Series(follow_up, dtype="float64"), tolerance=tolerance
        )
        for tolerance in (0.05, 0.095, 0.105, 0.12, 0.5)
    }
    assert results[0.05]["status"] == "fail"
    assert results[0.095]["status"] == "fail"
    assert results[0.105]["status"] == "uncertain"
    assert results[0.12]["status"] == "pass"
    assert results[0.5]["status"] == "pass"


def test_compare_identical_is_confirmed_pass():
    profile = _profile_from(list(range(500)), dtype="float64")
    # An identical batch has true distance 0. With a tolerance at or above
    # the 2*epsilon summary slack the comparison is a confirmed pass; a
    # tolerance of 0 can only ever be "uncertain" because the reference CDF
    # itself is only known to within epsilon.
    definite = profile.compare(
        pd.Series(list(range(500)), dtype="float64"), tolerance=0.02
    )
    assert definite["status"] == "pass"
    assert definite["distance_lower"] == 0.0
    tight = profile.compare(
        pd.Series(list(range(500)), dtype="float64"), tolerance=0.0
    )
    assert tight["status"] == "uncertain"


def test_compare_rejects_invalid_input():
    profile = _profile_from([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        profile.compare(pd.Series([True]), tolerance=0.1)
    with pytest.raises(ValueError, match="empty"):
        profile.compare(pd.Series([], dtype="float64"), tolerance=0.1)
    for tolerance in (-0.1, 1.1, True, "0.1", math.nan):
        with pytest.raises((TypeError, ValueError)):
            profile.compare(pd.Series([1.0]), tolerance=tolerance)


def test_compare_does_not_mutate_or_retain_input():
    reference = [1.0, 2.0, 3.0, 4.0]
    profile = _profile_from(reference)
    before = profile.to_dict()
    follow_up_array = np.array([1.0, 2.0, 3.0, 4.0])
    original = follow_up_array.copy()
    result = profile.compare(pd.Series(follow_up_array), tolerance=0.1)
    assert np.array_equal(follow_up_array, original)
    assert profile.to_dict() == before
    assert result["status"] == "pass"


def test_profile_repeated_queries_are_stable():
    profile = _profile_from(list(range(1000)), dtype="float64")
    first = [profile.cdf_bounds(t) for t in (0, 250, 500, 999)]
    second = [profile.cdf_bounds(t) for t in (0, 250, 500, 999)]
    assert first == second


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


def test_round_trip_preserves_counts_bounds_and_decisions():
    values = list(range(700)) + [random.Random(4).random() for _ in range(30)]
    profile = _profile_from(values)
    restored = NumericProfile.from_dict(json.loads(profile.to_json()))
    assert restored.n == profile.n
    for t in (0, 100, 350, 699, 700):
        assert restored.cdf_bounds(t) == profile.cdf_bounds(t)
    follow_up = pd.Series(values[:500], dtype="float64")
    assert restored.compare(follow_up, tolerance=0.1) == profile.compare(
        follow_up, tolerance=0.1
    )
    assert restored == profile


def _corrupt(profile_dict, mutate):
    data = json.loads(json.dumps(profile_dict))
    mutate(data)
    return data


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda d: d.__setitem__("version", 999), ValueError),
        (lambda d: d.__setitem__("version", "1"), TypeError),
        (lambda d: d.__setitem__("format", "other"), ValueError),
        (lambda d: d.__setitem__("epsilon", 0.5), ValueError),
        (lambda d: d.__setitem__("epsilon", True), TypeError),
        (lambda d: d.__setitem__("n", 10**9), ValueError),
        (lambda d: d.__setitem__("n", True), TypeError),
        (lambda d: d.pop("delta"), ValueError),
        (lambda d: d.__setitem__("values", []), ValueError),
        (lambda d: d.__setitem__("g", d["g"][:-1]), ValueError),
        (
            lambda d: d.__setitem__(
                "values", [float("nan")] * len(d["values"])
            ),
            ValueError,
        ),
        (
            lambda d: d.__setitem__("values", ["x"] * len(d["values"])),
            TypeError,
        ),
        (lambda d: d["g"].__setitem__(0, 0), ValueError),
        (lambda d: d["g"].__setitem__(0, True), TypeError),
        (
            lambda d: d.__setitem__("values", list(reversed(d["values"]))),
            ValueError,
        ),
    ],
)
def test_corrupted_serialisation_is_rejected(mutate, error):
    profile = _profile_from(list(range(40)), dtype="int64")
    with pytest.raises(error):
        NumericProfile.from_dict(_corrupt(profile.to_dict(), mutate))


def test_from_dict_does_not_accept_a_non_mapping():
    with pytest.raises(TypeError, match="mapping"):
        NumericProfile.from_dict([1, 2, 3])


def test_profile_json_size_is_compact():
    profile = _profile_from(
        [random.Random(5).random() for _ in range(100_000)]
    )
    assert len(profile.to_json().encode("utf-8")) < 256 * 1024


def test_randomized_bounds_property():
    """Fuzz the bounds against brute-force empirical CDFs and distances."""
    rng = random.Random(1234)
    for _ in range(400):
        dtype = rng.choice(["float64", "int64"])
        epsilon = rng.choice([0.01, 0.02, 0.05, 0.1])
        if dtype == "float64":
            reference = [
                round(rng.random(), 3) for _ in range(rng.randint(1, 80))
            ]
            follow_up = [
                round(rng.random(), 3) for _ in range(rng.randint(1, 80))
            ]
        else:
            reference = [
                rng.randint(-40, 40) for _ in range(rng.randint(1, 80))
            ]
            follow_up = [
                rng.randint(-40, 40) for _ in range(rng.randint(1, 120))
            ]
        # split the reference into arbitrary batches
        cuts = (
            sorted(
                rng.sample(
                    range(1, len(reference)), k=min(3, len(reference) - 1)
                )
            )
            if len(reference) > 1
            else []
        )
        batches = []
        last = 0
        for cut in cuts + [len(reference)]:
            batches.append(reference[last:cut])
            last = cut
        profile = _profile_from(
            reference, dtype=dtype, batches=batches, epsilon=epsilon
        )
        _assert_bounds_hold(profile, reference, epsilon)

        result = profile.compare(
            pd.Series(follow_up, dtype=dtype), tolerance=0.5
        )
        distance = _true_distance(reference, follow_up)
        lower, upper = result["distance_lower"], result["distance_upper"]
        assert 0.0 <= lower <= distance + 1e-12
        assert distance - 1e-12 <= upper <= 1.0
        assert upper - lower <= 4 * epsilon + 1e-9
