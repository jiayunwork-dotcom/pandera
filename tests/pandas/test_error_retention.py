"""Regression tests for error retention and uninterrupted lazy aggregation.

These tests cover the case where structural errors, dtype-coercion failures and
parser execution failures used to be dropped or to abort the whole lazy
validation. The hand-computed expectation for every scenario is stated inline.

Scenarios combined here:

* ``strict`` / ``ordered`` structural errors with all-true/all-false ``where``
* dtype coercion failures retaining ``DATATYPE_COERCION`` and the raw bad value
* column/schema parsers that raise or return an illegal row layout
* independent column checks continuing when another branch fails
* regex columns validated independently per actual matched column
* ``drop_invalid_rows`` refusing to repair structural/parser failures while
  still precisely dropping ordinary, physically locatable row failures
* duplicate labels, ``MultiIndex``, ``n_failure_cases`` and warnings
* legal parser transforms, single-execution guarantees and state isolation
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.api.parsers import Parser
from pandera.errors import (
    SchemaError,
    SchemaErrorReason,
    SchemaErrors,
)


def _all_true_where(df):
    return pd.Series([True] * len(df), index=df.index)


def _all_false_where(df):
    return pd.Series([False] * len(df), index=df.index)


def _reason_cases(schema_errors):
    return [err.reason_code for err in schema_errors]


def _columns(schema_errors):
    return {err.column_name for err in schema_errors}


# ---------------------------------------------------------------------------
# Structural (strict / ordered) errors survive conditional preprocessing
# ---------------------------------------------------------------------------


def test_strict_extra_column_with_all_true_where_is_reported():
    # Hand table: x=[1,2], extra=[3,4]. Expect one COLUMN_NOT_IN_SCHEMA error
    # naming `extra`; the all-true where still runs its check exactly once.
    calls = {"where": 0, "check": 0}

    def where_fn(df):
        calls["where"] += 1
        return _all_true_where(df)

    def check_fn(series):
        calls["check"] += 1
        return series > 0

    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int,
                pa.Check(check_fn, name="c", where=where_fn),
            )
        },
        strict=True,
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(
            pd.DataFrame({"x": [1, 2], "extra": [3, 4]}), lazy=True
        )
    errors = exc.value.schema_errors
    assert _reason_cases(errors) == [
        SchemaErrorReason.COLUMN_NOT_IN_SCHEMA
    ]
    assert errors[0].failure_cases == "extra"
    assert calls == {"where": 1, "check": 1}


def test_strict_extra_column_with_all_false_where_is_reported():
    # The check function never runs (nothing selected) but the structural
    # strict error still has to be raised.
    seen = []

    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int,
                pa.Check(
                    lambda s: seen.append(1) or s > 0,
                    name="c",
                    where=_all_false_where,
                ),
            )
        },
        strict=True,
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(
            pd.DataFrame({"x": [1, 2], "extra": [3, 4]}), lazy=True
        )
    assert _reason_cases(exc.value.schema_errors) == [
        SchemaErrorReason.COLUMN_NOT_IN_SCHEMA
    ]
    assert seen == []


def test_ordered_schema_with_where_reports_out_of_order_column():
    # Schema column order is x, y; input is y, x. Expect COLUMN_NOT_ORDERED
    # naming the first out-of-order column (`y`).
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int, pa.Check(lambda s: s > 0, name="c", where=_all_true_where)
            ),
            "y": pa.Column(int),
        },
        ordered=True,
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(
            pd.DataFrame({"y": [1, 2], "x": [3, 4]}), lazy=True
        )
    errors = exc.value.schema_errors
    assert _reason_cases(errors) == [
        SchemaErrorReason.COLUMN_NOT_ORDERED
    ]
    assert errors[0].failure_cases == "y"


# ---------------------------------------------------------------------------
# Coercion failures keep DATATYPE_COERCION and the original bad value
# ---------------------------------------------------------------------------


def test_coercion_failure_retains_bad_value_without_where():
    schema = pa.DataFrameSchema(
        columns={"x": pa.Column(int, coerce=True)}, coerce=True
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(pd.DataFrame({"x": ["1", "bad", "3"]}), lazy=True)
    reasons = _reason_cases(exc.value.schema_errors)
    assert SchemaErrorReason.DATATYPE_COERCION in reasons
    coercion = next(
        e
        for e in exc.value.schema_errors
        if e.reason_code == SchemaErrorReason.DATATYPE_COERCION
    )
    # the raw bad value survives the report
    assert coercion.failure_cases["failure_case"].tolist() == ["bad"]


def test_coercion_failure_retains_bad_value_with_where():
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int,
                coerce=True,
                checks=pa.Check(
                    lambda s: s > 0, name="c", where=_all_true_where
                ),
            )
        },
        coerce=True,
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(pd.DataFrame({"x": ["1", "bad", "3"]}), lazy=True)
    reasons = _reason_cases(exc.value.schema_errors)
    # the later WRONG_DATATYPE must not replace the coercion failure
    assert SchemaErrorReason.DATATYPE_COERCION in reasons
    assert SchemaErrorReason.WRONG_DATATYPE in reasons
    coercion = next(
        e
        for e in exc.value.schema_errors
        if e.reason_code == SchemaErrorReason.DATATYPE_COERCION
    )
    assert coercion.failure_cases["failure_case"].tolist() == ["bad"]


def test_regex_coercion_reports_each_actual_column_and_bad_value():
    # A single regex declaration matches v1 and v2; each keeps its own raw
    # bad value and the report names the actual column, not the pattern.
    schema = pa.DataFrameSchema(
        columns={"v[0-9]": pa.Column(int, coerce=True, regex=True)},
        coerce=True,
    )
    df = pd.DataFrame({"v1": ["1", "bad1"], "v2": ["bad2", "2"]})
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    coercion = [
        e
        for e in exc.value.schema_errors
        if e.reason_code == SchemaErrorReason.DATATYPE_COERCION
    ]
    by_column = {e.column_name: e.failure_cases["failure_case"].tolist()
                 for e in coercion}
    assert by_column == {"v1": ["bad1"], "v2": ["bad2"]}


# ---------------------------------------------------------------------------
# Column parser failures: CHECK_ERROR, located to column/parser, branch isolation
# ---------------------------------------------------------------------------


def test_column_parser_returning_illegal_layout_reports_check_error():
    # x parser drops the last row (3 rows -> 2 rows); independent y check on
    # y=[-1,2,3] fails on its first row. Both diagnostics must be reported.
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int, parsers=[Parser(lambda s: s.iloc[:-1], name="shrink")]
            ),
            "y": pa.Column(
                int, checks=pa.Check(lambda s: s > 0, name="y_pos")
            ),
        }
    )
    df = pd.DataFrame({"x": [1, 2, 3], "y": [-1, 2, 3]})
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    errors = exc.value.schema_errors
    reasons = _reason_cases(errors)
    assert SchemaErrorReason.CHECK_ERROR in reasons
    assert SchemaErrorReason.DATAFRAME_CHECK in reasons

    parser_error = next(
        e
        for e in errors
        if e.reason_code == SchemaErrorReason.CHECK_ERROR
    )
    assert parser_error.column_name == "x"
    assert parser_error.check == "shrink"
    assert parser_error.parser is not None
    assert parser_error.parser.name == "shrink"

    # the independent y check still ran and is attributed to y
    y_error = next(
        e
        for e in errors
        if e.reason_code == SchemaErrorReason.DATAFRAME_CHECK
    )
    assert y_error.column_name == "y"


def test_column_parser_raising_exception_preserves_original_exc():
    def boom(series):
        raise ValueError("parser kaboom")

    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int, parsers=[Parser(boom, name="boom_parser")]
            ),
            "y": pa.Column(
                int, checks=pa.Check(lambda s: s > 0, name="y_pos")
            ),
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(
            pd.DataFrame({"x": [1, 2, 3], "y": [-1, 2, 3]}), lazy=True
        )
    parser_error = next(
        e
        for e in exc.value.schema_errors
        if e.reason_code == SchemaErrorReason.CHECK_ERROR
    )
    assert parser_error.column_name == "x"
    assert isinstance(parser_error.original_exc, ValueError)
    assert str(parser_error.original_exc) == "parser kaboom"
    # the user's exception is reachable either via original_exc or the chain
    reachable = parser_error.original_exc or parser_error.__context__
    assert isinstance(reachable, ValueError)
    # and the independent column check was not skipped
    assert any(
        e.column_name == "y"
        and e.reason_code == SchemaErrorReason.DATAFRAME_CHECK
        for e in exc.value.schema_errors
    )


def test_failed_parser_output_is_not_consumed_by_dependent_work():
    # The x check depends on the parser's output; when the parser raises the
    # check must not run and no record is fabricated for it. A second parser
    # chained after the failing one must not run either.
    order = []
    x_check_calls = []

    def p1(series):
        order.append("p1")
        raise RuntimeError("p1 boom")

    def p2(series):
        order.append("p2")
        return series

    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int,
                parsers=[Parser(p1, name="p1"), Parser(p2, name="p2")],
                checks=pa.Check(
                    lambda s: x_check_calls.append(1) or s > 0,
                    name="x_check",
                ),
            )
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(pd.DataFrame({"x": [1, 2, 3]}), lazy=True)
    assert order == ["p1"]
    assert x_check_calls == []
    # exactly one error (the parser), nothing fabricated for the check
    assert len(exc.value.schema_errors) == 1
    assert (
        exc.value.schema_errors[0].reason_code
        == SchemaErrorReason.CHECK_ERROR
    )


def test_illegal_parser_output_does_not_overwrite_valid_table():
    # Even though the parser returns a shorter Series, the table carried by
    # the exception must keep the original three rows (bad output discarded).
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int, parsers=[Parser(lambda s: s.iloc[:-1], name="shrink")]
            )
        }
    )
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    assert len(exc.value.data) == 3
    assert exc.value.data["x"].tolist() == [1, 2, 3]


def test_parser_returning_non_pandas_object_reports_check_error():
    # A scalar (neither Series nor DataFrame) is an illegal parser output.
    schema = pa.DataFrameSchema(
        columns={"x": pa.Column(int, parsers=[Parser(lambda s: 42)])}
    )
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    errors = exc.value.schema_errors
    assert _reason_cases(errors) == [SchemaErrorReason.CHECK_ERROR]
    # the validated table is untouched
    assert exc.value.data["x"].tolist() == [1, 2]


def test_series_schema_parser_returning_scalar_reports_check_error():
    series_schema = pa.SeriesSchema(int, parsers=[Parser(lambda s: 42)])
    with pytest.raises(SchemaErrors) as exc:
        series_schema.validate(pd.Series([1, 2]), lazy=True)
    assert (
        exc.value.schema_errors[0].reason_code
        == SchemaErrorReason.CHECK_ERROR
    )


def test_regex_columns_handle_parser_failures_independently():
    # One regex declaration matches actual columns a1 and a2. The a1 parser
    # raises, the a2 parser succeeds and takes effect; an independent b check
    # still runs. Errors are attributed to the actual column names.
    executed = []

    def maybe_boom(series):
        executed.append(series.name)
        if series.name == "a1":
            raise ValueError(f"boom {series.name}")
        return series * 10

    schema = pa.DataFrameSchema(
        columns={
            "a[0-9]": pa.Column(
                int, parsers=[Parser(maybe_boom, name="p")], regex=True
            ),
            "b": pa.Column(
                int, checks=pa.Check(lambda s: s > 100, name="b_big")
            ),
        }
    )
    df = pd.DataFrame({"a1": [1, 2], "a2": [3, 4], "b": [-1, -2]})
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    errors = exc.value.schema_errors
    # a1 failure ...
    a1_error = next(
        e
        for e in errors
        if e.reason_code == SchemaErrorReason.CHECK_ERROR
    )
    assert a1_error.column_name == "a1"
    assert isinstance(a1_error.original_exc, ValueError)
    # ... the other matched column was not exempted ...
    assert set(executed) == {"a1", "a2"}
    # ... its successful parse took effect while a1 keeps its raw values ...
    assert exc.value.data["a2"].tolist() == [30, 40]
    assert exc.value.data["a1"].tolist() == [1, 2]
    # ... and the independent b column check ran
    assert any(
        e.column_name == "b"
        and e.reason_code == SchemaErrorReason.DATAFRAME_CHECK
        for e in errors
    )


# ---------------------------------------------------------------------------
# drop_invalid_rows: structural / parser failures block dropping
# ---------------------------------------------------------------------------


def test_drop_invalid_rows_raises_on_coercion_and_keeps_row_diagnostics():
    # x = ["1","bad","-5"]: coercion fails on "bad" AND the value check would
    # reject the coerced -5. Both kinds of diagnostics are queryable and
    # dropping rows cannot make the batch succeed.
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int,
                coerce=True,
                checks=pa.Check(lambda s: s > 0, name="xpos"),
            )
        },
        drop_invalid_rows=True,
        coerce=True,
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(
            pd.DataFrame({"x": ["1", "bad", "-5"]}), lazy=True
        )
    reasons = _reason_cases(exc.value.schema_errors)
    assert SchemaErrorReason.DATATYPE_COERCION in reasons
    coercion = next(
        e
        for e in exc.value.schema_errors
        if e.reason_code == SchemaErrorReason.DATATYPE_COERCION
    )
    assert coercion.failure_cases["failure_case"].tolist() == ["bad"]


def test_drop_invalid_rows_raises_on_parser_failure():
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int, parsers=[Parser(lambda s: s.iloc[:-1], name="shrink")]
            )
        },
        drop_invalid_rows=True,
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(pd.DataFrame({"x": [1, 2, 3]}), lazy=True)
    assert any(
        e.reason_code == SchemaErrorReason.CHECK_ERROR
        for e in exc.value.schema_errors
    )


def test_drop_invalid_rows_still_drops_locatable_row_failures():
    # Only ordinary row checks fail; physical positions are reliable, so the
    # bad rows are removed precisely.
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int,
                pa.Check(lambda s: s > 0, name="xpos", where=_all_true_where),
            )
        },
        drop_invalid_rows=True,
    )
    out = schema.validate(
        pd.DataFrame({"x": [-1, 2, -3, 4]}), lazy=True
    )
    assert out["x"].tolist() == [2, 4]


def test_drop_invalid_rows_with_duplicate_labels_keeps_good_rows():
    # Repeated index labels must not cause good rows sharing a label to be
    # removed; positional dropping distinguishes them.
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int,
                pa.Check(lambda s: s > 0, name="xpos", where=_all_true_where),
            )
        },
        drop_invalid_rows=True,
    )
    duplicated = pd.DataFrame(
        {"x": [-1, 2, -3]}, index=["k", "k", "k"]
    )
    out = schema.validate(duplicated, lazy=True)
    assert out["x"].tolist() == [2]
    assert out.index.tolist() == ["k"]


def test_drop_invalid_rows_with_multiindex_keeps_good_rows():
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int,
                pa.Check(lambda s: s > 0, name="xpos", where=_all_true_where),
            )
        },
        drop_invalid_rows=True,
    )
    midx = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1)], names=["l1", "l2"]
    )
    out = schema.validate(
        pd.DataFrame({"x": [-1, 2, -3]}, index=midx), lazy=True
    )
    assert out["x"].tolist() == [2]


def test_n_failure_cases_limits_diagnostics_not_dropping():
    # Three rows fail but only one failure case is displayed; the drop set
    # still contains all three physical rows.
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int,
                pa.Check(
                    lambda s: s > 0,
                    name="xpos",
                    n_failure_cases=1,
                    where=_all_true_where,
                ),
            )
        },
        drop_invalid_rows=True,
    )
    out = schema.validate(
        pd.DataFrame({"x": [-1, -2, -3, 4]}), lazy=True
    )
    assert out["x"].tolist() == [4]


def test_warning_check_is_non_blocking_and_keeps_rows():
    schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int,
                pa.Check(
                    lambda s: s > 0,
                    name="warn_check",
                    raise_warning=True,
                    where=_all_true_where,
                ),
            )
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = schema.validate(
            pd.DataFrame({"x": [-1, -2]}), lazy=True
        )
    assert out["x"].tolist() == [-1, -2]
    assert any(
        issubclass(w.category, pa.errors.SchemaWarning) for w in caught
    )


# ---------------------------------------------------------------------------
# Legal parser transforms and single-execution guarantees
# ---------------------------------------------------------------------------


def test_schema_parser_reorders_then_positions_apply_to_result():
    def reorder(df):
        return df.sort_values("amount").reset_index(drop=True)

    where_calls, check_calls = [], []
    schema = pa.DataFrameSchema(
        columns={
            "amount": pa.Column(
                int,
                pa.Check(
                    lambda s: check_calls.append(1) or s > 0,
                    name="p",
                    where=lambda d: where_calls.append(len(d))
                    or d["settled"],
                ),
            )
        },
        parsers=[Parser(reorder)],
        drop_invalid_rows=True,
    )
    df = pd.DataFrame(
        {"settled": [True, True, False], "amount": [-5, 1, -9]},
        index=["x", "y", "z"],
    )
    out = schema.validate(df, lazy=True)
    # sorted amounts: [-9 (not selected), -5 (selected, fails), 1 (passes)]
    assert out["amount"].tolist() == [-9, 1]
    # the where mask and the check function each run exactly once
    assert where_calls == [3]
    assert len(check_calls) == 1


def test_value_changing_column_parser_runs_once_and_takes_effect():
    calls = {"n": 0}

    def negate(series):
        calls["n"] += 1
        return series * -1

    schema = pa.DataFrameSchema(
        columns={
            "v": pa.Column(
                int,
                parsers=[Parser(negate)],
                checks=pa.Check(
                    lambda s: s > -100, name="p", where=_all_true_where
                ),
            )
        }
    )
    out = schema.validate(
        pd.DataFrame({"v": [-5, 1, -9]}), lazy=True
    )
    assert out["v"].tolist() == [5, -1, 9]
    assert calls["n"] == 1


def test_schema_parser_filters_rows_before_checks():
    def keep_positive(df):
        return df[df["v"] > 0].reset_index(drop=True)

    schema = pa.DataFrameSchema(
        columns={
            "v": pa.Column(
                int,
                pa.Check(
                    lambda s: s > 0, name="pos", where=_all_true_where
                ),
            )
        },
        parsers=[Parser(keep_positive)],
    )
    out = schema.validate(
        pd.DataFrame({"v": [-1, 2, -3, 4]}), lazy=True
    )
    assert out["v"].tolist() == [2, 4]


# ---------------------------------------------------------------------------
# lazy=False first-failure semantics and state isolation
# ---------------------------------------------------------------------------


def test_lazy_false_parser_failure_raises_schema_error_with_chain():
    def boom(series):
        raise ValueError("kaboom")

    schema = pa.DataFrameSchema(
        columns={"x": pa.Column(int, parsers=[Parser(boom)])}
    )
    with pytest.raises(SchemaError) as exc:
        schema.validate(pd.DataFrame({"x": [1, 2]}), lazy=False)
    assert exc.value.reason_code == SchemaErrorReason.CHECK_ERROR
    assert isinstance(exc.value.original_exc, ValueError)


def test_lazy_false_schema_parser_failure_raises_schema_error():
    def boom(df):
        raise ValueError("df kaboom")

    schema = pa.DataFrameSchema(parsers=[Parser(boom)])
    with pytest.raises(SchemaError) as exc:
        schema.validate(pd.DataFrame({"x": [1]}), lazy=False)
    assert exc.value.reason_code == SchemaErrorReason.CHECK_ERROR
    assert isinstance(exc.value.original_exc, ValueError)


def test_repeated_calls_do_not_leak_errors_or_mutate_input():
    def shrink(series):
        return series.iloc[:-1]

    failing_schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int, parsers=[Parser(shrink, name="shrink")]
            ),
            "y": pa.Column(
                int, checks=pa.Check(lambda s: s > 0, name="ypos")
            ),
        }
    )
    bad = pd.DataFrame({"x": [1, 2, 3], "y": [-1, 2, 3]})
    for _ in range(2):
        with pytest.raises(SchemaErrors) as exc:
            failing_schema.validate(bad, lazy=True)
        # stable diagnostics: nothing accumulates on the schema across calls
        assert len(exc.value.schema_errors) == 2

    # a legal schema on a legal input still works after a prior failure
    ok_schema = pa.DataFrameSchema(
        columns={
            "x": pa.Column(
                int, parsers=[Parser(lambda s: s * 2, name="double")]
            )
        }
    )
    out = ok_schema.validate(
        pd.DataFrame({"x": [1, 2]}), lazy=True
    )
    assert out["x"].tolist() == [2, 4]

    # inplace=False leaves the caller's frame untouched
    df = pd.DataFrame({"x": [1, 2]})
    before = df.copy()
    ok_schema.validate(df, lazy=True, inplace=False)
    pd.testing.assert_frame_equal(df, before)


def test_failed_parser_does_not_corrupt_schema_configuration():
    def shrink(series):
        return series.iloc[:-1]

    column = pa.Column(int, parsers=[Parser(shrink)])
    schema = pa.DataFrameSchema(columns={"x": column})
    with pytest.raises(SchemaErrors):
        schema.validate(pd.DataFrame({"x": [1, 2, 3]}), lazy=True)
    # the parser list is still exactly the one the user configured
    assert len(schema.columns["x"].parsers) == 1


# ---------------------------------------------------------------------------
# Wide-table memory discipline
# ---------------------------------------------------------------------------


def test_errors_do_not_retain_wide_table_copies():
    n_rows, n_cols = 100, 40
    data = {
        f"c{i}": ([-1] * n_rows if i % 2 else [1] * n_rows)
        for i in range(n_cols)
    }
    df = pd.DataFrame(data)
    schema = pa.DataFrameSchema(
        {
            f"c{i}": pa.Column(
                int, pa.Check(lambda s: s > 0, name=f"chk{i}")
            )
            for i in range(n_cols)
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    # 20 failing columns, and none retains the validated table object
    assert len(exc.value.schema_errors) == n_cols // 2
    assert all(err.data is None for err in exc.value.schema_errors)
    # each diagnostic is a narrow canonical frame, never a full wide table
    assert all(
        not hasattr(err.failure_cases, "shape")
        or err.failure_cases.shape[1] <= 4
        for err in exc.value.schema_errors
    )


# ---------------------------------------------------------------------------
# Standalone SeriesSchema parser failures
# ---------------------------------------------------------------------------


def test_series_schema_parser_raising_reports_check_error():
    def boom(series):
        raise ValueError("series boom")

    series_schema = pa.SeriesSchema(int, parsers=[Parser(boom)])
    with pytest.raises(SchemaErrors) as exc:
        series_schema.validate(pd.Series([1, 2]), lazy=True)
    error = exc.value.schema_errors[0]
    assert error.reason_code == SchemaErrorReason.CHECK_ERROR
    assert isinstance(error.original_exc, ValueError)


def test_series_schema_parser_illegal_layout_reports_check_error():
    series_schema = pa.SeriesSchema(
        int, parsers=[Parser(lambda s: s.iloc[:-1])]
    )
    with pytest.raises(SchemaErrors) as exc:
        series_schema.validate(pd.Series([1, 2, 3]), lazy=True)
    assert (
        exc.value.schema_errors[0].reason_code
        == SchemaErrorReason.CHECK_ERROR
    )


def test_series_schema_legal_parser_runs_once():
    calls = {"n": 0}

    def twice(series):
        calls["n"] += 1
        return series * 2

    series_schema = pa.SeriesSchema(int, parsers=[Parser(twice)])
    out = series_schema.validate(pd.Series([1, 2]))
    assert out.tolist() == [2, 4]
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Configuration contracts that must remain unchanged
# ---------------------------------------------------------------------------


def test_where_on_standalone_series_schema_still_rejected():
    with pytest.raises(pa.errors.SchemaDefinitionError):
        pa.SeriesSchema(
            int,
            checks=pa.Check(lambda s: s > 0, where=_all_true_where),
        ).validate(pd.Series([1]))


def test_drop_invalid_rows_without_lazy_still_rejected():
    schema = pa.DataFrameSchema(
        columns={"x": pa.Column(int, pa.Check.ge(0))},
        drop_invalid_rows=True,
    )
    with pytest.raises(pa.errors.SchemaDefinitionError):
        schema.validate(pd.DataFrame({"x": [-1]}), lazy=False)


def test_wrong_dtype_with_drop_invalid_rows_raises():
    # A dtype mismatch is structural and cannot be repaired by row deletion.
    schema = pa.DataFrameSchema(
        columns={"numbers": pa.Column(str)}, drop_invalid_rows=True
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(
            pd.DataFrame({"numbers": [1, 2, 3]}), lazy=True
        )
    assert any(
        e.reason_code == SchemaErrorReason.WRONG_DATATYPE
        for e in exc.value.schema_errors
    )


# ---------------------------------------------------------------------------
# Numerical sanity helper kept to make numpy usage explicit
# ---------------------------------------------------------------------------


def test_numpy_environment_matches_supported_versions():
    assert np.__version__.startswith("1.26")
    assert pd.__version__.startswith("2.2.3")
