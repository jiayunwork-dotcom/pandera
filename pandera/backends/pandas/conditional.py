"""Per-validation context for conditional (``where``) checks.

This module is intentionally backend specific: the conditional validation
feature only supports pandas :class:`~pandas.DataFrame` objects. A
:class:`ConditionalValidationContext` is created exactly once per
``DataFrameSchema.validate`` call and threaded through the existing validation
call stack. It owns three pieces of state:

1. the *full* preprocessed table (after schema-level parsers, default filling
   and dtype coercion), which defines physical row positions,
2. a cache of evaluated ``where`` masks, guaranteeing that every conditional
   callable is evaluated at most once for a given validation,
3. the accumulated set of physical row positions that must be dropped when
   ``drop_invalid_rows=True``.

Nothing in this object is ever attached to a schema or :class:`Check`
instance, so repeated, interleaved or failed validations can never leak state
into one another.
"""

from typing import Any, Callable, Dict, FrozenSet, Optional

import numpy as np
import pandas as pd

from pandera.errors import SchemaDefinitionError, SchemaError, SchemaErrorReason


def coerce_boolean_mask(mask: Any, full_table: pd.DataFrame) -> pd.Series:
    """Coerce a user-provided ``where`` result into a boolean Series.

    ``NA`` values are treated as "not selected" (mapped to ``False``). A
    nullable boolean dtype is preserved up to the final fill so that ordinary
    integer/object columns are rejected early instead of being silently
    interpreted.
    """
    if isinstance(mask, pd.DataFrame):
        raise SchemaError(
            schema=None,
            data=full_table,
            message=(
                "`where` callable must return a boolean Series or a scalar "
                f"bool, got DataFrame with shape {mask.shape}."
            ),
            failure_cases=str(type(mask)),
            check="where",
            reason_code=SchemaErrorReason.CHECK_ERROR,
        )

    if isinstance(mask, bool) or mask is None or np.isscalar(mask):
        raise SchemaError(
            schema=None,
            data=full_table,
            message=(
                "`where` callable must return a boolean Series of the same "
                f"length and index as the validated dataframe, got scalar "
                f"value {mask!r}."
            ),
            failure_cases=repr(mask),
            check="where",
            reason_code=SchemaErrorReason.CHECK_ERROR,
        )

    if not isinstance(mask, pd.Series):
        raise SchemaError(
            schema=None,
            data=full_table,
            message=(
                "`where` callable must return a pandas boolean Series, got "
                f"{type(mask)}."
            ),
            failure_cases=str(type(mask)),
            check="where",
            reason_code=SchemaErrorReason.CHECK_ERROR,
        )

    if len(mask) != len(full_table):
        raise SchemaError(
            schema=None,
            data=full_table,
            message=(
                "`where` callable returned a Series of length "
                f"{len(mask)}, expected {len(full_table)} (the length of "
                "the validated dataframe)."
            ),
            failure_cases=len(mask),
            check="where",
            reason_code=SchemaErrorReason.CHECK_ERROR,
        )

    if not mask.index.equals(full_table.index):
        raise SchemaError(
            schema=None,
            data=full_table,
            message=(
                "`where` callable returned a Series whose index does not "
                "match the validated dataframe's index."
            ),
            failure_cases=mask.index,
            check="where",
            reason_code=SchemaErrorReason.CHECK_ERROR,
        )

    if not isinstance(mask.dtype, pd.BooleanDtype):
        # only accept plain bool / nullable bool; astype(bool) on an object
        # Series would silently turn arbitrary objects into booleans.
        if mask.dtype != np.dtype("bool"):
            raise SchemaError(
                schema=None,
                data=full_table,
                message=(
                    "`where` callable must return a boolean Series (dtype "
                    f"bool or 'boolean'), got dtype {mask.dtype!r}."
                ),
                failure_cases=str(mask.dtype),
                check="where",
                reason_code=SchemaErrorReason.CHECK_ERROR,
            )
        return mask.astype(bool)

    # nullable boolean: NA explicitly means "not selected".
    return mask.fillna(False).astype(bool)


class ConditionalValidationContext:
    """Bookkeeping shared by every check of one validation run."""

    def __init__(self, full_table: pd.DataFrame):
        self._full_table = full_table
        self._mask_cache: Dict[int, pd.Series] = {}
        self._where_call_counts: Dict[int, int] = {}
        # physical positions (0-based, relative to the preprocessed table)
        # that must be removed when drop_invalid_rows=True
        self._dropped_positions: set = set()
        # any collected error that cannot be attributed to concrete physical
        # rows makes dropping impossible; the final drop step then raises.
        self._has_unlocatable_error = False

    @property
    def full_table(self) -> pd.DataFrame:
        return self._full_table

    @property
    def n_rows(self) -> int:
        return len(self._full_table)

    @property
    def dropped_positions(self) -> FrozenSet[int]:
        return frozenset(self._dropped_positions)

    @property
    def has_unlocatable_error(self) -> bool:
        return self._has_unlocatable_error

    def mark_unlocatable_error(self) -> None:
        self._has_unlocatable_error = True

    def add_failure_positions(self, positions) -> None:
        """Record physical row positions that failed a locatable check."""
        self._dropped_positions.update(int(p) for p in positions)

    def evaluate_where(self, check) -> pd.Series:
        """Evaluate (and cache) the selection mask of a conditional check.

        The mask is interpreted positionally: its order is the row order of
        the preprocessed table, regardless of the index labels.
        """
        key = id(check)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        where_fn: Callable = check.where
        self._where_call_counts[key] = (
            self._where_call_counts.get(key, 0) + 1
        )
        # hand the user a read-only view: the mask contract is positional,
        # so callbacks must not be able to mutate the validated table.
        read_only = self._full_table.copy()
        read_only.attrs = dict(self._full_table.attrs)
        try:
            raw_mask = where_fn(read_only)
        except SchemaError:
            raise
        except Exception as exc:
            raise SchemaError(
                schema=None,
                data=self._full_table,
                message=(
                    f"Error while executing `where` callable "
                    f"{getattr(where_fn, '__name__', type(where_fn))}: "
                    f"{exc.__class__.__name__}({exc})"
                ),
                failure_cases=exc.__class__.__name__,
                check="where",
                reason_code=SchemaErrorReason.CHECK_ERROR,
                original_exc=exc,
            ) from exc

        mask = coerce_boolean_mask(raw_mask, self._full_table)
        self._mask_cache[key] = mask
        return mask

    def where_call_count(self, check) -> int:
        return self._where_call_counts.get(id(check), 0)

    def sample_positions(
        self,
        head: Optional[int],
        tail: Optional[int],
        sample: Optional[int],
        random_state: Optional[int],
    ) -> Optional[np.ndarray]:
        """Compute sampled physical positions on the full, unfiltered table.

        Mirrors :meth:`pandas.DataFrame.head` / ``tail`` / ``sample`` semantics
        (including the random seed), but de-duplicates by physical position
        instead of index label, so repeated labels and overlapping head/tail/
        sample windows cannot cause a row to be checked twice.
        """
        if head is None and tail is None and sample is None:
            return None
        n = self.n_rows
        positions = []
        if head is not None:
            positions.extend(range(min(head, n)))
        if tail is not None:
            positions.extend(range(max(0, n - tail), n))
        if sample is not None:
            if sample >= n:
                sampled = np.arange(n)
            else:
                # pd.DataFrame.sample uses a numpy RandomState seeded with
                # random_state; reuse it so the seed keeps its meaning.
                rng = np.random.RandomState(random_state)
                sampled = np.sort(rng.choice(n, size=sample, replace=False))
            positions.extend(sampled.tolist())
        return np.array(sorted(set(positions)), dtype=np.int64)

    def selection_for_check(
        self,
        check,
        head: Optional[int],
        tail: Optional[int],
        sample: Optional[int],
        random_state: Optional[int],
    ) -> np.ndarray:
        """Return the physical positions that enter a conditional check.

        The sampling window is computed against the full table and then
        intersected positionally with the ``where`` mask.
        """
        mask = self.evaluate_where(check)
        selected = np.flatnonzero(mask.to_numpy())
        sampled = self.sample_positions(head, tail, sample, random_state)
        if sampled is None:
            return selected
        if len(sampled) == 0:
            return sampled
        return np.intersect1d(selected, sampled, assume_unique=False)

    def assert_parser_preserves_rows(
        self, before: pd.DataFrame, after: pd.DataFrame, parser_name: str
    ) -> None:
        """Reject column parsers that change row count or the index sequence.

        Physical row positions would otherwise no longer map to the table the
        ``where`` masks describe.
        """
        if len(before) != len(after) or not before.index.equals(after.index):
            raise SchemaDefinitionError(
                f"Column parser {parser_name!r} changed the number of rows "
                "or the index sequence while conditional (`where`) checks "
                "are enabled. Conditional validation requires column "
                "parsers to preserve row count and index order; move row "
                "filtering into a schema-level parser."
            )
