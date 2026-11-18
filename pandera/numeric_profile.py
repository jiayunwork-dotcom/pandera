"""Deterministic distribution profiles for numeric pandas columns.

A :class:`NumericProfile` summarises a stream of numeric observations with a
*deterministic* error bound on the empirical cumulative distribution function
(CDF).  It is built from one or more pandas ``Series`` batches, retains only a
compact summary (never the observations themselves), and can be serialised to
and from plain JSON-able data so that a schema carrying a profile can be saved
and reloaded in a different process.

The summary is a Greenwald-Khanna style quantile sketch.  For every stored
tuple ``(value, g, delta)`` with prefix count ``P`` the sketch maintains
``true_rank(value) in [P - g, P + delta]`` and ``g + delta <= floor(2 * eps * n)``.
Consequently :meth:`NumericProfile.cdf_bounds` returns ``[L, U]`` such that

    ``0 <= L <= F(t) <= U <= 1``  and  ``U - L <= 2 * epsilon``

for the empirical CDF ``F(t) = #(x <= t) / n`` of the *actual observations*
that were fed to :meth:`NumericProfile.from_batches`.  The bound is a
deterministic guarantee for the data actually seen; it is not a random
confidence interval.

The representation is intentionally not canonical: feeding the same
observations in a different batch layout or row order can produce a different
summary, but every valid summary satisfies the same guarantees.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from pandera.errors import (
    DistributionCheckError,
    DistributionUncertainError,
)

__all__ = ["NumericProfile"]

PROFILE_FORMAT = "pandera.numeric_profile"
PROFILE_VERSION = 1


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_finite_real(value: Any, name: str) -> float:
    """Return ``value`` as a finite float, rejecting bools and non-reals."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not a boolean.")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}."
        )
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return as_float


def validate_epsilon(value: Any) -> float:
    """Validate the relative error parameter ``epsilon``."""
    eps = _as_finite_real(value, "epsilon")
    if not 0.0 < eps <= 0.1:
        raise ValueError(
            f"epsilon must be in the interval (0, 0.1], got {value!r}."
        )
    return eps


def validate_tolerance(value: Any) -> float:
    """Validate a distance threshold ``tolerance``."""
    tol = _as_finite_real(value, "tolerance")
    if not 0.0 <= tol <= 1.0:
        raise ValueError(
            f"tolerance must be in the interval [0, 1], got {value!r}."
        )
    return tol


def _validate_threshold(value: Any) -> Any:
    """Validate ``t`` and return it without losing integer precision.

    Integers are returned as Python ``int`` so that comparisons against large
    integer observations (for example around ``2**53``) stay exact; floats are
    returned as ``float``.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("t must be a real number, not a boolean.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ValueError(f"t must be finite, got {value!r}.")
        return as_float
    raise TypeError(f"t must be a real number, got {type(value).__name__}.")


# ---------------------------------------------------------------------------
# Supported dtypes / batch extraction
# ---------------------------------------------------------------------------


def _is_supported_numeric_dtype(dtype: Any) -> bool:
    """Whether ``dtype`` is one of the supported numeric dtypes.

    Only ``int64``, ``float64`` and pandas' nullable ``Int64`` / ``Float64``
    are accepted.  Booleans, objects, narrower integers/floats and complex
    dtypes are rejected.
    """
    if isinstance(dtype, (bool, np.bool_)):
        return False
    try:
        if dtype == np.dtype("int64") or dtype == np.dtype("float64"):
            return True
        if dtype == pd.Int64Dtype() or dtype == pd.Float64Dtype():
            return True
    except TypeError:  # pragma: no cover - exotic dtype objects
        return False
    return False


def _batch_values(series: Any) -> np.ndarray:
    """Return the finite, non-missing values of a batch, ascending.

    The returned array is either ``int64`` or ``float64``; integer batches are
    never cast to float, so distinct integers near ``2**53`` stay distinct.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(
            "each batch must be a pandas.Series, got "
            f"{type(series).__name__}."
        )
    dtype = series.dtype
    if not _is_supported_numeric_dtype(dtype):
        raise TypeError(
            "distribution profiles only support int64, float64, nullable "
            f"Int64 and Float64 columns, got dtype {dtype!r}."
        )

    if dtype == np.dtype("int64"):
        int_array: np.ndarray = series.to_numpy(dtype=np.int64, copy=False)
        return np.sort(int_array, kind="mergesort")
    if dtype == pd.Int64Dtype():
        int_array = series.dropna().to_numpy(dtype=np.int64, copy=False)
        return np.sort(int_array, kind="mergesort")

    if dtype == np.dtype("float64"):
        float_array: np.ndarray = series.to_numpy(dtype=np.float64, copy=False)
        if np.isinf(float_array).any():
            raise ValueError(
                "observations must be finite; found +inf or -inf."
            )
        float_array = float_array[~np.isnan(float_array)]
    else:  # Float64
        float_array = series.dropna().to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(float_array).all():
            raise ValueError(
                "observations must be finite; found +inf or -inf."
            )
    return np.sort(float_array, kind="mergesort")


def _check_batches_source(batches: Any) -> None:
    if isinstance(batches, (pd.Series, pd.DataFrame)):
        raise TypeError(
            "batches must be an iterable of pandas.Series, not a single "
            f"{type(batches).__name__}."
        )


# ---------------------------------------------------------------------------
# Greenwald-Khanna summary
# ---------------------------------------------------------------------------


class _GKSummary:
    """Greenwald-Khanna style summary with deterministic CDF bounds."""

    __slots__ = ("epsilon", "values", "g", "delta", "n")

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self.values: List[Any] = []
        self.g: List[int] = []
        self.delta: List[int] = []
        self.n = 0

    # -- update -----------------------------------------------------------
    def _threshold(self) -> int:
        # One below the textbook floor(2*eps*n): the query interval width is
        # ``(g + delta) / n <= threshold / n``, so keeping threshold strictly
        # below ``2*eps*n`` keeps the width strictly below ``2*epsilon`` even
        # after float rounding.
        return int(math.floor(2.0 * self.epsilon * self.n)) - 1

    def add_sorted(self, values: Sequence[Any]) -> None:
        """Merge a batch of ascending values into the summary."""
        if not values:
            return
        old_values = self.values
        old_g = self.g
        old_delta = self.delta
        new_values: List[Any] = []
        new_g: List[int] = []
        new_delta: List[int] = []
        i = 0
        old_len = len(old_values)
        for value in values:
            while i < old_len and old_values[i] <= value:
                new_values.append(old_values[i])
                new_g.append(old_g[i])
                new_delta.append(old_delta[i])
                i += 1
            self.n += 1
            if not new_values or i == old_len:
                delta = 0
            else:
                delta = old_g[i] + old_delta[i] - 1
                if delta < 0:
                    delta = 0
            new_values.append(value)
            new_g.append(1)
            new_delta.append(delta)
        while i < old_len:
            new_values.append(old_values[i])
            new_g.append(old_g[i])
            new_delta.append(old_delta[i])
            i += 1
        self.values = new_values
        self.g = new_g
        self.delta = new_delta
        self._compress()

    def _compress(self) -> None:
        threshold = self._threshold()
        if threshold < 1:
            return
        values = self.values
        g = self.g
        delta = self.delta
        # Never merge the first tuple: it pins the exact minimum, and its
        # delta is always 0.
        i = len(values) - 2
        while i >= 1:
            if g[i] + g[i + 1] + delta[i + 1] <= threshold:
                g[i + 1] += g[i]
                del values[i]
                del g[i]
                del delta[i]
            i -= 1

    # -- query ------------------------------------------------------------
    def prefix(self) -> List[int]:
        prefix = [0] * (len(self.g) + 1)
        running = 0
        for index, count in enumerate(self.g):
            running += count
            prefix[index + 1] = running
        return prefix

    # -- state ------------------------------------------------------------
    def state(self) -> Tuple[List[Any], List[int], List[int]]:
        return list(self.values), list(self.g), list(self.delta)


def _build_summary(batches: Iterable[pd.Series], epsilon: float) -> _GKSummary:
    summary = _GKSummary(epsilon)
    for batch in batches:
        summary.add_sorted(_batch_values(batch).tolist())
    return summary


# ---------------------------------------------------------------------------
# NumericProfile
# ---------------------------------------------------------------------------


class NumericProfile:
    """Compact, serialisable distribution profile for a numeric column.

    Instances are read-only: :meth:`from_batches`, :meth:`cdf_bounds` and
    :meth:`compare` never modify the profile or their inputs, and validating
    several batches (even interleaved) does not leak diagnostics between
    calls.
    """

    __slots__ = ("_epsilon", "_n", "_values", "_g", "_delta", "_prefix")

    def __init__(
        self,
        epsilon: float,
        n: int,
        values: Sequence[Any],
        g: Sequence[int],
        delta: Sequence[int],
    ) -> None:
        self._epsilon = epsilon
        self._n = n
        self._values = tuple(values)
        self._g = tuple(g)
        self._delta = tuple(delta)
        self._prefix = tuple(_prefix_of(self._g))

    # -- construction -----------------------------------------------------
    @classmethod
    def from_batches(
        cls, batches: Iterable[pd.Series], *, epsilon: float = 0.01
    ) -> "NumericProfile":
        """Build a profile from an iterable of pandas ``Series`` batches.

        The source is traversed exactly once.  Missing values are ignored,
        remaining observations must be finite, and batches may be empty.  An
        integer batch is never converted to float, so repeated integers are
        counted by multiplicity and distinct integers around ``2**53`` remain
        distinct.  The source, the batches and the observations are not
        retained by the returned profile.

        :param batches: single-pass iterable of pandas ``Series``.
        :param epsilon: relative error in ``(0, 0.1]``; ``U - L <= 2*epsilon``.
        :raises TypeError: if a batch is not a Series or has an unsupported
            dtype.
        :raises ValueError: if observations are not finite or there are no
            valid observations at all.
        """
        eps = validate_epsilon(epsilon)
        _check_batches_source(batches)
        summary = _build_summary(batches, eps)
        if summary.n == 0:
            raise ValueError(
                "cannot build a distribution profile: no valid (non-missing) "
                "observations were provided."
            )
        values, g, delta = summary.state()
        return cls(eps, summary.n, values, g, delta)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NumericProfile":
        """Load a profile from its JSON-able representation.

        The representation is validated: unknown versions, wrong counts,
        invalid ``epsilon``, non-finite or non-numeric values, and broken
        structural invariants are all rejected with ``ValueError`` or
        ``TypeError``.
        """
        if not isinstance(data, Mapping):
            raise TypeError(
                "serialized profile must be a mapping, got "
                f"{type(data).__name__}."
            )
        required = {
            "format",
            "version",
            "epsilon",
            "n",
            "values",
            "g",
            "delta",
        }
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"serialized profile is missing fields: {sorted(missing)}."
            )
        if data["format"] != PROFILE_FORMAT:
            raise ValueError(
                f"unrecognized profile format {data['format']!r}."
            )
        version = data["version"]
        if isinstance(version, (bool, np.bool_)) or not isinstance(
            version, int
        ):
            raise TypeError("profile version must be an integer.")
        if version != PROFILE_VERSION:
            raise ValueError(
                f"unsupported profile version {version!r}; expected "
                f"{PROFILE_VERSION}."
            )
        eps = validate_epsilon(data["epsilon"])
        n = data["n"]
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, int):
            raise TypeError("profile count n must be an integer.")
        if n <= 0:
            raise ValueError("profile count n must be positive.")

        values = data["values"]
        g = data["g"]
        delta = data["delta"]
        for name, seq in (("values", values), ("g", g), ("delta", delta)):
            if not isinstance(seq, (list, tuple)):
                raise TypeError(f"profile field {name!r} must be a sequence.")
        if not (len(values) == len(g) == len(delta)):
            raise ValueError(
                "profile fields values, g and delta must have equal length."
            )
        if len(values) == 0:
            raise ValueError(
                "profile must contain at least one summary tuple."
            )

        clean_values: List[Any] = []
        previous: Any = None
        for index, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    "profile values must be numbers, found a boolean."
                )
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise TypeError(
                    "profile values must be numbers, found "
                    f"{type(value).__name__}."
                )
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("profile values must be finite.")
            if index and value < previous:
                raise ValueError("profile values must be non-decreasing.")
            previous = value
            clean_values.append(value)

        clean_g: List[int] = []
        clean_delta: List[int] = []
        for name, seq, minimum in (("g", g, 1), ("delta", delta, 0)):
            cleaned: List[int] = []
            for count in seq:
                if isinstance(count, (bool, np.bool_)) or not isinstance(
                    count, (int, np.integer)
                ):
                    raise TypeError(
                        f"profile field {name!r} must be integers."
                    )
                count = int(count)
                if count < minimum:
                    raise ValueError(
                        f"profile field {name!r} has an out-of-range count "
                        f"{count}."
                    )
                cleaned.append(count)
            if name == "g":
                clean_g = cleaned
            else:
                clean_delta = cleaned

        if sum(clean_g) != n:
            raise ValueError(
                "profile counts are inconsistent: sum(g) does not equal n."
            )
        if len(clean_values) == n:
            if any(count != 1 for count in clean_g) or any(
                d != 0 for d in clean_delta
            ):
                raise ValueError(
                    "an exact profile must have g == 1 and delta == 0 for "
                    "every tuple."
                )
        else:
            threshold = int(math.floor(2.0 * eps * n)) - 1
            for count, d in zip(clean_g, clean_delta):
                if count + d > threshold:
                    raise ValueError(
                        "profile violates the error-bound invariant "
                        "(g + delta > floor(2 * epsilon * n) - 1)."
                    )
        return cls(eps, n, clean_values, clean_g, clean_delta)

    # -- properties -------------------------------------------------------
    @property
    def n(self) -> int:
        """Number of valid (non-missing, finite) observations."""
        return self._n

    @property
    def epsilon(self) -> float:
        """Relative error bound used by :meth:`cdf_bounds`."""
        return self._epsilon

    # -- public API -------------------------------------------------------
    def cdf_bounds(self, t: float) -> Tuple[float, float]:
        """Return deterministic ``[L, U]`` bounds on the empirical CDF at ``t``.

        For the empirical CDF ``F(t)`` of the observations used to build this
        profile, ``0 <= L <= F(t) <= U <= 1`` and ``U - L <= 2 * epsilon``.
        ``t`` must be a finite real number (booleans are rejected).
        """
        threshold = _validate_threshold(t)
        lower, upper = _summary_bounds(
            self._values,
            self._g,
            self._delta,
            self._prefix,
            self._n,
            threshold,
        )
        return lower, upper

    def compare(
        self, series: pd.Series, *, tolerance: float
    ) -> Dict[str, Any]:
        """Compare a complete follow-up batch against the reference profile.

        The same data rules as :meth:`from_batches` apply: ``series`` must be
        a pandas Series with a supported numeric dtype, missing values are
        ignored, remaining values must be finite, and there must be at least
        one valid observation.

        Returns a JSON-serialisable mapping with ``distance_lower``,
        ``distance_upper`` and ``status``.  The true distance ``D`` between
        the two empirical CDFs is the maximum absolute difference, and
        ``0 <= distance_lower <= D <= distance_upper <= 1`` with
        ``distance_upper - distance_lower <= 4 * epsilon``.  The status is
        ``"pass"`` when ``distance_upper <= tolerance``, ``"fail"`` when
        ``distance_lower > tolerance`` and ``"uncertain"`` otherwise.  No
        p-value is computed.  The input batch is not retained.
        """
        tol = validate_tolerance(tolerance)
        sorted_values = _batch_values(series).tolist()
        if not sorted_values:
            raise ValueError(
                "cannot compare an empty batch: no valid (non-missing) "
                "observations were provided."
            )
        return _compare_against_values(
            self._values,
            self._g,
            self._delta,
            self._prefix,
            self._n,
            sorted_values,
            tol,
            self._epsilon,
        )

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a fresh, JSON-able representation of the profile."""
        return {
            "format": PROFILE_FORMAT,
            "version": PROFILE_VERSION,
            "epsilon": self._epsilon,
            "n": self._n,
            "values": list(self._values),
            "g": list(self._g),
            "delta": list(self._delta),
        }

    def to_json(self, **kwargs: Any) -> str:
        """Return the profile serialised as compact JSON."""
        return json.dumps(self.to_dict(), sort_keys=False, **kwargs)

    # -- dunder -----------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NumericProfile):
            return NotImplemented
        return (
            self._epsilon == other._epsilon
            and self._n == other._n
            and self._values == other._values
            and self._g == other._g
            and self._delta == other._delta
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._epsilon,
                self._n,
                self._values,
                self._g,
                self._delta,
            )
        )

    def __repr__(self) -> str:
        return (
            f"NumericProfile(n={self._n}, epsilon={self._epsilon}, "
            f"tuples={len(self._values)})"
        )


# ---------------------------------------------------------------------------
# Bounds helpers
# ---------------------------------------------------------------------------


def _prefix_of(g: Sequence[int]) -> List[int]:
    prefix = [0] * (len(g) + 1)
    running = 0
    for index, count in enumerate(g):
        running += count
        prefix[index + 1] = running
    return prefix


def _summary_bounds(
    values: Sequence[Any],
    g: Sequence[int],
    delta: Sequence[int],
    prefix: Sequence[int],
    n: int,
    t: float,
) -> Tuple[float, float]:
    # Stored values are plain Python int/float, so ``bisect_right`` is exact
    # even for integers near 2**53 (Python compares int and float exactly).
    rank = bisect.bisect_right(values, t)
    if len(values) == n:
        return rank / n, rank / n
    index = rank
    if index == 0:
        return 0.0, 0.0
    if index == len(values):
        return 1.0, 1.0
    lower = prefix[index]
    upper = prefix[index + 1] + delta[index]
    if upper > n:
        upper = n
    return lower / n, upper / n


def _compare_against_values(
    values: Sequence[Any],
    g: Sequence[int],
    delta: Sequence[int],
    prefix: Sequence[int],
    n: int,
    sorted_new: Sequence[Any],
    tolerance: float,
    epsilon: float,
) -> Dict[str, Any]:
    """Compare a reference summary with an *exact* follow-up sample.

    The follow-up empirical CDF ``G`` is known exactly (the whole batch is
    available), so for every candidate threshold we combine the reference
    bounds with ``G`` to obtain rigorous bounds on
    ``max_t |F(t) - G(t)|``.  The distance is computed in integer numerator
    space (common denominator ``n * m``) so the lower bound never exceeds the
    true distance due to floating-point rounding.
    """
    m = len(sorted_new)
    exact_ref = len(values) == n
    candidates = sorted(set(values) | set(sorted_new))
    # numerators are multiples of 1/(n*m); track the binding pairs so the
    # reported bounds can be derived from exact integers.
    low_num = 0  # max over t of n*m * candidate_low
    high_num = 0  # max over t of n*m * candidate_high
    for t in candidates:
        rank_ref = bisect.bisect_right(values, t)
        if exact_ref:
            lower_ref_num = upper_ref_num = rank_ref * m
        elif rank_ref == 0:
            lower_ref_num = upper_ref_num = 0
        elif rank_ref == len(values):
            lower_ref_num = upper_ref_num = n * m
        else:
            lower_ref_num = prefix[rank_ref] * m
            upper_ref_num = (prefix[rank_ref + 1] + delta[rank_ref]) * m
            if upper_ref_num > n * m:
                upper_ref_num = n * m
        cdf_new_num = bisect.bisect_right(sorted_new, t) * n
        candidate_low = max(
            lower_ref_num - cdf_new_num, cdf_new_num - upper_ref_num, 0
        )
        candidate_high = max(
            upper_ref_num - cdf_new_num, cdf_new_num - lower_ref_num
        )
        if candidate_low > low_num:
            low_num = candidate_low
        if candidate_high > high_num:
            high_num = candidate_high
    denominator = n * m
    # ``low_num`` and ``high_num`` are exact integer numerators over the common
    # denominator ``n * m``.  ``low_num / denominator <= D`` and
    # ``high_num / denominator >= D`` hold exactly in rational arithmetic, and
    # round-to-nearest is monotonic, so the float quotients preserve the
    # ordering.  A tiny guard absorbs the follow-up sample's own float slack
    # used by independent brute-force checks; the reported width stays within
    # ``2 * epsilon`` (and therefore within the required ``4 * epsilon``).
    guard = 1e-12
    distance_lower = low_num / denominator
    distance_upper = high_num / denominator
    distance_lower = min(1.0, max(0.0, distance_lower))
    distance_upper = min(1.0, max(distance_lower, distance_upper) + guard)
    if distance_upper <= tolerance:
        status = "pass"
    elif distance_lower > tolerance:
        status = "fail"
    else:
        status = "uncertain"
    return {
        "distance_lower": float(distance_lower),
        "distance_upper": float(distance_upper),
        "status": status,
    }


# ---------------------------------------------------------------------------
# Check function used by ``Check.distribution_within``
# ---------------------------------------------------------------------------


def _distribution_within_check(
    data: Any, profile: NumericProfile, tolerance: float
) -> bool:
    """Execute a profile comparison as a pandera check function.

    Returns ``True`` when the batch is confirmed to pass; raises
    :class:`~pandera.errors.DistributionCheckError` on a confirmed drift and
    :class:`~pandera.errors.DistributionUncertainError` when the distance
    cannot be resolved against the threshold.
    """
    # Imported lazily to avoid a cycle with pandera.api.checks.
    from pandera.errors import SchemaDefinitionError

    if isinstance(profile, Mapping):  # pragma: no cover - defensive
        profile = NumericProfile.from_dict(profile)
    if not isinstance(profile, NumericProfile):
        raise TypeError(
            "profile must be a NumericProfile, got "
            f"{type(profile).__name__}."
        )
    if not isinstance(data, pd.Series):
        raise SchemaDefinitionError(
            "distribution_within is only supported on a single pandas "
            f"numeric column, got {type(data).__name__}."
        )
    result = profile.compare(data, tolerance=tolerance)
    if result["status"] == "pass":
        return True
    message = (
        "distribution check failed: status={status}, distance in "
        "[{lower:.6g}, {upper:.6g}] against tolerance {tolerance:.6g} "
        "(reference n={n}, epsilon={epsilon:.6g})."
    ).format(
        status=result["status"],
        lower=result["distance_lower"],
        upper=result["distance_upper"],
        tolerance=tolerance,
        n=profile.n,
        epsilon=profile.epsilon,
    )
    error_cls = (
        DistributionUncertainError
        if result["status"] == "uncertain"
        else DistributionCheckError
    )
    raise error_cls(
        message,
        distance_lower=result["distance_lower"],
        distance_upper=result["distance_upper"],
        tolerance=tolerance,
        n=profile.n,
        status=result["status"],
    )
