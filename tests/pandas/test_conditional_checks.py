"""Tests for conditional validation via the ``where`` Check argument."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.api.parsers import Parser
from pandera.errors import (
    SchemaDefinitionError,
    SchemaErrors,
)


# ---------------------------------------------------------------------------
# Hand-computed reference tables used throughout the suite
# ---------------------------------------------------------------------------
#
# base table (physical positions 0..4):
#
#   position | index label | settled | amount
#   ---------+--------------+---------+--------
#        0    |     "a"      |  True   |   10
#        1    |     "b"      |  False  |   -5   (not selected, never checked)
#        2    |     "c"      |  True   |  -20   (selected, FAILS amount > 0)
#        3    |     "d"      |  False  |   -1   (not selected, never checked)
#        4    |     "e"      |  True   |   30
#
# selected positions for ``settled`` mask: {0, 2, 4}
# failing physical positions: {2}
# retained rows after drop_invalid_rows: positions {0, 1, 3, 4}


def _base_frame():
    return pd.DataFrame(
        {
            "settled": [True, False, True, False, True],
            "amount": [10.0, -5.0, -20.0, -1.0, 30.0],
        },
        index=["a", "b", "c", "d", "e"],
    )


def _column_schema(**overrides):
    check = pa.Check(
        lambda s: s > 0,
        name="positive_when_settled",
        where=lambda d: d["settled"],
        **overrides,
    )
    return pa.DataFrameSchema(
        {
            "amount": pa.Column(float, check),
            "settled": pa.Column(bool),
        }
    )


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_where_must_be_callable():
    with pytest.raises(ValueError, match="`where` must be a callable"):
        pa.Check(lambda s: s > 0, where="settled")


def test_where_and_groupby_are_mutually_exclusive():
    with pytest.raises(SchemaDefinitionError, match="`where` and `groupby`"):
        pa.Check(
            lambda d: d,
            groupby="g",
            where=lambda d: d["x"],
        )


def test_where_is_not_forwarded_to_check_fn():
    received = {}

    def fn(series, **kwargs):
        received.update(kwargs)
        return series.notna()

    schema = pa.DataFrameSchema(
        {
            "amount": pa.Column(
                float, pa.Check(fn, name="kw", where=lambda d: d["settled"])
            ),
            "settled": pa.Column(bool),
        }
    )
    schema.validate(_base_frame(), lazy=True)
    assert "where" not in received


def test_series_schema_rejects_where():
    schema = pa.SeriesSchema(
        float,
        checks=pa.Check(lambda s: s > 0, name="p", where=lambda d: d > 0),
    )
    with pytest.raises(SchemaDefinitionError, match="SeriesSchema"):
        schema.validate(pd.Series([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Mask semantics
# ---------------------------------------------------------------------------


def test_mask_all_true_checks_every_row():
    df = pd.DataFrame({"flag": [True, True], "v": [1, -2]})
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int, pa.Check(lambda s: s > 0, name="p", where=lambda d: d["flag"])
            ),
            "flag": pa.Column(bool),
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    assert exc.value.failure_cases["row_position"].tolist() == [1]


def test_mask_all_false_passes_without_calling_check_fn():
    calls = {"n": 0}

    def fn(s):
        calls["n"] += 1
        return s > 0

    df = pd.DataFrame({"flag": [False, False], "v": [-1, -2]})
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int, pa.Check(fn, name="p", where=lambda d: d["flag"])
            ),
            "flag": pa.Column(bool),
        }
    )
    result = schema.validate(df, lazy=True)
    assert len(result) == 2
    assert calls["n"] == 0


def test_nullable_boolean_mask_treats_na_as_not_selected():
    df = pd.DataFrame(
        {
            "settled": pd.array([True, False, None, True], dtype="boolean"),
            "amount": [10.0, -5.0, -9.0, -1.0],
        }
    )
    schema = pa.DataFrameSchema(
        {
            "amount": pa.Column(
                float,
                pa.Check(lambda s: s > 0, name="p", where=lambda d: d["settled"]),
            ),
            "settled": pa.Column(bool, nullable=True),
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    # only position 3 is selected and failing; position 2 (NA mask) is ignored
    cond_rows = exc.value.failure_cases[
        exc.value.failure_cases["check"].eq("p")
    ]
    assert cond_rows["row_position"].tolist() == [3]


def test_column_check_where_can_read_other_columns():
    # the column check's condition reads a different column; the checked
    # column itself must remain independently valid.
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 0, 30]})
    schema = pa.DataFrameSchema(
        {
            "a": pa.Column(
                int,
                pa.Check(
                    lambda s: s > 0,
                    name="cross",
                    where=lambda d: d["b"] > 5,
                ),
            ),
            "b": pa.Column(int),
        }
    )
    # selected rows (b > 5): positions 0 and 2, both positive -> passes;
    # without cross-column access the mask could not be evaluated at all.
    schema.validate(df, lazy=True)


@pytest.mark.parametrize(
    "bad_mask,expected_fragment",
    [
        (lambda d: True, "scalar"),
        (lambda d: pd.Series([True, False]), "length"),
        (
            lambda d: pd.Series([True, False, True, False, True],
                                index=[9, 8, 7, 6, 5]),
            "index",
        ),
        (
            # matching length and default RangeIndex, wrong dtype
            lambda d: pd.Series([1, 0, 1, 0, 1]),
            "dtype",
        ),
        (lambda d: pd.DataFrame({"x": [True] * 5}), "DataFrame"),
    ],
)
def test_invalid_mask_shapes_are_reported_as_validation_errors(
    bad_mask, expected_fragment
):
    # use a default RangeIndex frame so index mismatches don't mask the
    # other, independently reported shape/dtype errors.
    frame = pd.DataFrame(
        {"amount": [10.0, -5.0, -20.0, -1.0, 30.0]}
    )
    schema = pa.DataFrameSchema(
        checks=pa.Check(lambda d: d["amount"] > 0, name="x", where=bad_mask)
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(frame, lazy=True)
    assert any(
        expected_fragment in str(err)
        for err in exc.value.schema_errors
    )


def test_where_exception_is_a_check_error_not_a_pass():
    schema = pa.DataFrameSchema(
        checks=pa.Check(
            lambda d: d["amount"] > 0,
            name="boom",
            where=lambda d: 1 / 0,
        )
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(_base_frame(), lazy=True)
    assert exc.value.schema_errors[0].reason_code.name == "CHECK_ERROR"


def test_where_evaluated_once_per_validate():
    counts = {"n": 0}

    def where_fn(d):
        counts["n"] += 1
        return d["settled"]

    schema = pa.DataFrameSchema(
        checks=[
            pa.Check(lambda d: d["amount"] < 100, name="c1", where=where_fn),
            pa.Check(lambda d: d["amount"] < 100, name="c2", where=where_fn),
        ],
        columns={
            "amount": pa.Column(float),
            "settled": pa.Column(bool),
        },
    )
    schema.validate(_base_frame(), lazy=True)
    assert counts["n"] == 2  # once per check, not once per row/re-evaluation


def test_user_where_callable_gets_a_read_only_table():
    def mutating(d):
        d["__hack__"] = 1
        return d["settled"]

    df = _base_frame()
    schema = pa.DataFrameSchema(
        {
            "amount": pa.Column(
                float, pa.Check(lambda s: s.notna(), name="p", where=mutating)
            ),
            "settled": pa.Column(bool),
        }
    )
    schema.validate(df, lazy=True)
    assert "__hack__" not in df.columns


# ---------------------------------------------------------------------------
# Both check granularities and both vectorized output shapes
# ---------------------------------------------------------------------------


def test_column_level_vectorized_check_honors_selection():
    with pytest.raises(SchemaErrors) as exc:
        _column_schema().validate(_base_frame(), lazy=True)
    fc = exc.value.failure_cases
    assert fc["row_position"].tolist() == [2]
    assert fc["index"].tolist() == ["c"]
    assert fc["failure_case"].tolist() == [-20.0]


def test_column_level_element_wise_check_honors_selection():
    schema = pa.DataFrameSchema(
        {
            "amount": pa.Column(
                float,
                pa.Check(
                    lambda x: x > 0,
                    element_wise=True,
                    name="ew",
                    where=lambda d: d["settled"],
                ),
            ),
            "settled": pa.Column(bool),
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(_base_frame(), lazy=True)
    assert exc.value.failure_cases["row_position"].tolist() == [2]


def test_table_level_series_output():
    schema = pa.DataFrameSchema(
        checks=pa.Check(
            lambda d: d["amount"] > 0,
            name="tbl",
            where=lambda d: d["settled"],
        )
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(_base_frame(), lazy=True)
    assert exc.value.failure_cases["row_position"].tolist() == [2]


def test_table_level_dataframe_output_attributes_cells_and_rows():
    df = pd.DataFrame(
        {"a": [1, -2, 3], "b": [-3, 4, 5], "flag": [True, True, False]}
    )
    schema = pa.DataFrameSchema(
        checks=pa.Check(
            lambda d: d[["a", "b"]] > 0,
            name="wide",
            where=lambda d: d["flag"],
        )
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    fc = exc.value.failure_cases
    assert set(zip(fc["row_position"], fc["column"])) == {
        (0, "b"),
        (1, "a"),
    }


def test_table_level_element_wise_check_honors_selection():
    df = pd.DataFrame(
        {"flag": [True, False, True], "a": [10, -9, -3], "b": [2, 8, 4]}
    )
    schema = pa.DataFrameSchema(
        checks=pa.Check(
            lambda row: row["a"] < row["b"],
            element_wise=True,
            name="rowwise",
            where=lambda d: d["flag"],
        )
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    fc = exc.value.failure_cases
    # only selected rows 0 and 2 were checked; row 0 violates a < b, row 2
    # passes. Row 1 was excluded by the mask.
    assert fc["row_position"].tolist() == [0]
    assert fc["failure_case"].tolist() == [False]


def test_scalar_false_does_not_attribute_individual_rows():
    schema = pa.DataFrameSchema(
        checks=pa.Check(
            lambda d: (d["amount"] > 0).all(),
            name="scalar",
            where=lambda d: d["settled"],
        )
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(_base_frame(), lazy=True)
    err = exc.value.schema_errors[0]
    assert err.row_positions is None
    assert err.failure_cases is False  # scalar failure marker


def test_scalar_false_with_drop_mode_still_raises():
    schema = pa.DataFrameSchema(
        checks=pa.Check(
            lambda d: (d["amount"] > 0).all(),
            name="scalar",
            where=lambda d: d["settled"],
        ),
        drop_invalid_rows=True,
    )
    with pytest.raises(SchemaErrors):
        schema.validate(_base_frame(), lazy=True)


def test_empty_table_passes():
    schema = _column_schema()
    empty = _base_frame().iloc[[]]
    assert len(schema.validate(empty, lazy=True)) == 0


def test_ignore_na_applies_after_selection():
    df = pd.DataFrame(
        {
            "flag": [True, True, True, False],
            "v": pd.array([1.0, None, -3.0, -9.0], dtype="Float64"),
        }
    )
    schema_ignore = pa.DataFrameSchema(
        {
            "v": pa.Column(
                float,
                pa.Check(
                    lambda s: s > 0,
                    name="p",
                    ignore_na=True,
                    where=lambda d: d["flag"],
                ),
                nullable=True,
            ),
            "flag": pa.Column(bool),
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema_ignore.validate(df, lazy=True)
    assert exc.value.failure_cases["row_position"].tolist() == [2]

    schema_strict = pa.DataFrameSchema(
        {
            "v": pa.Column(
                float,
                pa.Check(
                    lambda s: s > 0,
                    name="p",
                    ignore_na=False,
                    where=lambda d: d["flag"],
                ),
                nullable=True,
            ),
            "flag": pa.Column(bool),
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema_strict.validate(df, lazy=True)
    assert sorted(exc.value.failure_cases["row_position"]) == [1, 2]


# ---------------------------------------------------------------------------
# Duplicate labels / identical rows / MultiIndex
# ---------------------------------------------------------------------------


def test_duplicate_index_labels_keep_distinct_records():
    df = pd.DataFrame(
        {"flag": [True] * 4, "v": [1, -2, -2, 4]}, index=[0, 1, 1, 3]
    )
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int, pa.Check(lambda s: s > 0, name="p", where=lambda d: d["flag"])
            ),
            "flag": pa.Column(bool),
        },
        drop_invalid_rows=True,
    )
    out = schema.validate(df, lazy=True)
    # both physical rows 1 and 2 share label 1 and identical value -2; both
    # fail independently, the passing twins elsewhere are untouched.
    assert out["v"].tolist() == [1, 4]
    assert out.index.tolist() == [0, 3]


def test_duplicate_multiindex_labels():
    midx = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 1), ("b", 2), ("b", 2)]
    )
    df = pd.DataFrame(
        {"flag": [True] * 4, "v": [1, -2, 3, -4]}, index=midx
    )
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int, pa.Check(lambda s: s > 0, name="p", where=lambda d: d["flag"])
            ),
            "flag": pa.Column(bool),
        },
        drop_invalid_rows=True,
    )
    out = schema.validate(df, lazy=True)
    assert out["v"].tolist() == [1, 3]


def test_identical_rows_are_separate_records_in_diagnostics():
    df = pd.DataFrame(
        {"flag": [True, True], "v": [-5, -5]}, index=["x", "x"]
    )
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int, pa.Check(lambda s: s > 0, name="p", where=lambda d: d["flag"])
            ),
            "flag": pa.Column(bool),
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True)
    assert exc.value.failure_cases["row_position"].tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_head_intersects_where_on_full_table():
    df = pd.DataFrame(
        {"flag": [True] * 10, "v": list(range(10))}
    )
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int, pa.Check(lambda s: s < 1, name="lt1", where=lambda d: d["flag"])
            ),
            "flag": pa.Column(bool),
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(df, lazy=True, head=3)
    positions = sorted(exc.value.failure_cases["row_position"])
    assert positions == [1, 2]  # head=3 window, position 0 passes


def test_sample_seed_is_reproducible_and_intersects_mask():
    df = pd.DataFrame(
        {"flag": [i % 2 == 0 for i in range(10)], "v": list(range(10))}
    )
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int, pa.Check(lambda s: s < 1, name="lt1", where=lambda d: d["flag"])
            ),
            "flag": pa.Column(bool),
        }
    )

    def collect(seed):
        with pytest.raises(SchemaErrors) as exc:
            schema.validate(df, lazy=True, sample=4, random_state=seed)
        return sorted(exc.value.failure_cases["row_position"])

    first = collect(42)
    second = collect(42)
    assert first == second
    # only even (selected) positions can ever appear
    assert all(p % 2 == 0 and p != 0 for p in first)


def test_overlapping_head_tail_sample_checks_each_row_once():
    seen = []

    def ew(x):
        seen.append(x)
        return True

    df = pd.DataFrame({"flag": [True] * 8, "v": list(range(8))})
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                pa.Check(
                    ew, element_wise=True, name="ew", where=lambda d: d["flag"]
                ),
            ),
            "flag": pa.Column(bool),
        }
    )
    schema.validate(df, head=4, tail=4, sample=4, random_state=0)
    assert len(seen) == len(set(seen))


def test_unchecked_rows_remain_in_returned_frame():
    # sampling only narrows what is checked; rows outside the sample are not
    # validated but remain present in the returned frame, in original order.
    passing = pd.DataFrame(
        {"flag": [True] * 6, "v": list(range(1, 7))}
    )
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                pa.Check(lambda s: s > 0, name="p", where=lambda d: d["flag"]),
            ),
            "flag": pa.Column(bool),
        },
    )
    out = schema.validate(passing, sample=3, random_state=1, lazy=True)
    assert len(out) == 6
    assert out["v"].tolist() == list(range(1, 7))


# ---------------------------------------------------------------------------
# Diagnostics vs the full failure set / warnings / dedup
# ---------------------------------------------------------------------------


def test_n_failure_cases_limits_diagnostics_but_not_drop_set():
    df = pd.DataFrame(
        {"flag": [True] * 6, "v": [-1, -2, -3, -4, -5, -6]}
    )
    schema_drop = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                pa.Check(
                    lambda s: s > 0,
                    name="p",
                    n_failure_cases=1,
                    where=lambda d: d["flag"],
                ),
            ),
            "flag": pa.Column(bool),
        },
        drop_invalid_rows=True,
    )
    out = schema_drop.validate(df, lazy=True)
    assert len(out) == 0  # all six physical rows removed

    schema_report = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                pa.Check(
                    lambda s: s > 0,
                    name="p",
                    n_failure_cases=1,
                    where=lambda d: d["flag"],
                ),
            ),
            "flag": pa.Column(bool),
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema_report.validate(df, lazy=True)
    assert len(exc.value.failure_cases) == 1


def test_warning_check_keeps_all_rows():
    df = pd.DataFrame(
        {"flag": [True] * 3, "v": [-1, 2, -3]}
    )
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                pa.Check(
                    lambda s: s > 0,
                    name="warn",
                    raise_warning=True,
                    where=lambda d: d["flag"],
                ),
            ),
            "flag": pa.Column(bool),
        },
        drop_invalid_rows=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = schema.validate(df, lazy=True)
        assert any(
            issubclass(w.category, pa.errors.SchemaWarning) for w in caught
        )
    assert len(out) == 3


def test_row_failing_several_checks_is_dropped_once():
    df = pd.DataFrame(
        {"flag": [True] * 3, "v": [-1, 2, -3], "w": [-9, 8, -7]}
    )
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                pa.Check(lambda s: s > 0, name="c1", where=lambda d: d["flag"]),
            ),
            "w": pa.Column(
                int,
                pa.Check(lambda s: s > 0, name="c2", where=lambda d: d["flag"]),
            ),
            "flag": pa.Column(bool),
        },
        drop_invalid_rows=True,
    )
    out = schema.validate(df, lazy=True)
    assert out["v"].tolist() == [2]
    # original order and index labels preserved
    assert out.index.tolist() == [1]


def test_mixed_where_and_plain_checks_share_row_attribution():
    # a plain (non-where) row-wise check in a schema that also has a where
    # check: drop must not remove same-label good rows.
    df = pd.DataFrame(
        {"flag": [True, False, True], "v": [-1, 5, -3]}, index=[0, 0, 1]
    )
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                [
                    pa.Check(lambda s: s > -100, name="plain"),
                    pa.Check(
                        lambda s: s > 0,
                        name="conditional",
                        where=lambda d: d["flag"],
                    ),
                ],
            ),
            "flag": pa.Column(bool),
        },
        drop_invalid_rows=True,
    )
    out = schema.validate(df, lazy=True)
    # plain check passes everywhere; conditional fails physical rows 0 and 2
    assert out["v"].tolist() == [5]
    assert out.index.tolist() == [0]


# ---------------------------------------------------------------------------
# Missing columns and unlocatable errors
# ---------------------------------------------------------------------------


def test_missing_column_in_drop_mode_raises_instead_of_guessing():
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                pa.Check(lambda s: s > 0, name="p", where=lambda d: d["flag"]),
            )
        },
        drop_invalid_rows=True,
    )
    with pytest.raises(SchemaErrors):
        schema.validate(pd.DataFrame({"other": [1, 2]}), lazy=True)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_schema_level_parser_reorders_then_positions_apply_to_result():
    def reorder(d):
        return d.sort_values("amount").reset_index(drop=True)

    df = pd.DataFrame(
        {
            "settled": [True, True, False],
            "amount": [-5, 1, -9],
        },
        index=["x", "y", "z"],
    )
    schema = pa.DataFrameSchema(
        columns={
            "amount": pa.Column(
                int,
                pa.Check(
                    lambda s: s > 0,
                    name="p",
                    where=lambda d: d["settled"],
                ),
            ),
            "settled": pa.Column(bool),
        },
        parsers=[Parser(reorder)],
        drop_invalid_rows=True,
    )
    out = schema.validate(df, lazy=True)
    # after sorting by amount: [-9(False), -5(True fail), 1(True pass)]
    assert out["amount"].tolist() == [-9, 1]


def test_value_changing_column_parser_is_allowed():
    df = pd.DataFrame(
        {"flag": [True, True, False], "v": [-5, 1, -9]}
    )
    schema = pa.DataFrameSchema(
        columns={
            "v": pa.Column(
                int,
                parsers=[Parser(lambda s: s * 2)],
                checks=pa.Check(
                    lambda s: s > -100, name="p", where=lambda d: d["flag"]
                ),
            ),
            "flag": pa.Column(bool),
        }
    )
    out = schema.validate(df, lazy=True)
    assert out["v"].tolist() == [-10, 2, -18]


def test_row_changing_column_parser_is_rejected():
    df = pd.DataFrame(
        {"flag": [True, True, False], "v": [-5, 1, -9]}
    )
    schema = pa.DataFrameSchema(
        columns={
            "v": pa.Column(
                int,
                parsers=[Parser(lambda s: s.iloc[:-1])],
                checks=pa.Check(
                    lambda s: s > 0, name="p", where=lambda d: d["flag"]
                ),
            ),
            "flag": pa.Column(bool),
        }
    )
    with pytest.raises(SchemaDefinitionError, match="row"):
        schema.validate(df, lazy=True)


# ---------------------------------------------------------------------------
# State isolation and non-mutation
# ---------------------------------------------------------------------------


def test_repeated_and_interleaved_validations_do_not_leak_state():
    where_lengths = []

    def where_fn(d):
        where_lengths.append(len(d))
        return d["flag"]

    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int, pa.Check(lambda s: s > 0, name="p", where=where_fn)
            ),
            "flag": pa.Column(bool),
        }
    )
    good = pd.DataFrame({"flag": [True], "v": [1]})
    bad = pd.DataFrame({"flag": [True], "v": [-1]})

    schema.validate(good)
    with pytest.raises(SchemaErrors):
        schema.validate(bad, lazy=True)
    schema.validate(good)  # must not remember the earlier failure

    assert where_lengths == [1, 1, 1]


def test_inplace_false_does_not_mutate_caller():
    df = pd.DataFrame({"flag": [True, False], "v": [1.0, 2.0]})
    before = df.copy()
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                float,
                pa.Check(lambda s: s > 0, name="p", where=lambda d: d["flag"]),
            ),
            "flag": pa.Column(bool),
        }
    )
    schema.validate(df, lazy=True, inplace=False)
    pd.testing.assert_frame_equal(df, before)


def test_schema_without_where_has_no_row_position_column():
    schema = pa.DataFrameSchema(
        {"v": pa.Column(int, pa.Check(lambda s: s > 0, name="p"))}
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(pd.DataFrame({"v": [-1]}), lazy=True)
    assert "row_position" not in exc.value.failure_cases.columns


def test_check_fn_and_where_are_not_re_executed():
    where_counts = {"n": 0}
    check_counts = {"n": 0}

    def condition(d):
        where_counts["n"] += 1
        return d["flag"]

    def fn(s):
        check_counts["n"] += 1
        return s > 0

    df = pd.DataFrame(
        {"flag": [True, True, False, True], "v": [1, -2, 3, -4]}
    )
    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(int, pa.Check(fn, name="p", where=condition)),
            "flag": pa.Column(bool),
        },
        drop_invalid_rows=True,
    )
    # one validate: where evaluated once and the vectorized check runs once
    # against the three selected rows (not once per row, not re-run to
    # reconstruct errors or drop rows).
    schema.validate(df, lazy=True)
    assert where_counts["n"] == 1
    assert check_counts["n"] == 1

    schema.validate(df, lazy=True)
    assert where_counts["n"] == 2
    assert check_counts["n"] == 2


def test_where_checks_are_thread_safe():
    from joblib import Parallel, delayed

    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                pa.Check(lambda s: s > 0, name="p", where=lambda d: d["flag"]),
            ),
            "flag": pa.Column(bool),
        }
    )
    df = pd.DataFrame({"flag": [True, False, True], "v": [1, -9, 3]})

    def run():
        return schema.validate(df, lazy=True)

    results = Parallel(n_jobs=4, prefer="threads")(
        delayed(run)() for _ in range(12)
    )
    assert all(len(r) == 3 for r in results)


def test_unlocatable_errors_still_carry_row_position_column():
    # scalar False cannot attribute rows, but the schema is conditional so
    # the consolidated frame must still expose row_position (NA-filled).
    schema = pa.DataFrameSchema(
        checks=pa.Check(
            lambda d: (d["v"] > 0).all(),
            name="scalar",
            where=lambda d: d["flag"],
        )
    )
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(
            pd.DataFrame({"v": [1, -2], "flag": [True, True]}), lazy=True
        )
    assert "row_position" in exc.value.failure_cases.columns
    assert exc.value.failure_cases["row_position"].isna().all()

    # same guarantee for a missing required column
    schema_missing = pa.DataFrameSchema(
        columns={
            "v": pa.Column(
                int,
                pa.Check(lambda s: s > 0, name="p", where=lambda d: d["flag"]),
            )
        }
    )
    with pytest.raises(SchemaErrors) as exc:
        schema_missing.validate(
            pd.DataFrame({"other": [1]}), lazy=True
        )
    assert "row_position" in exc.value.failure_cases.columns


def test_check_fn_exception_is_reportable_and_blocks_dropping():
    def boom(_):
        raise ValueError("kaboom")

    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                pa.Check(boom, name="boom_check", where=lambda d: d["flag"]),
            ),
            "flag": pa.Column(bool),
        },
        drop_invalid_rows=True,
    )
    # cannot attribute rows to a check that raised: raise rather than guess
    with pytest.raises(SchemaErrors) as exc:
        schema.validate(
            pd.DataFrame({"flag": [True], "v": [1]}), lazy=True
        )
    assert exc.value.schema_errors[0].reason_code.name == "CHECK_ERROR"


def test_malformed_check_output_index_is_a_validation_error():
    schema = pa.DataFrameSchema(
        checks=pa.Check(
            lambda d: pd.Series(
                [True, False, True], index=[10, 11, 12]
            ),
            name="bad_output",
            where=lambda d: d["flag"],
        )
    )
    df = pd.DataFrame({"flag": [True, True, True], "v": [1, 2, 3]})
    with pytest.raises(SchemaErrors):
        schema.validate(df, lazy=True)


def test_all_na_mask_selects_no_rows_and_skips_check_fn():
    calls = {"n": 0}

    def fn(s):
        calls["n"] += 1
        return s > 0

    schema = pa.DataFrameSchema(
        {
            "v": pa.Column(
                int,
                pa.Check(
                    fn,
                    name="p",
                    where=lambda d: pd.Series(
                        pd.array([None, None], dtype="boolean")
                    ),
                ),
            )
        }
    )
    out = schema.validate(pd.DataFrame({"v": [-1, -2]}), lazy=True)
    assert len(out) == 2
    assert calls["n"] == 0


def test_context_does_not_accumulate_across_calls_or_store_wide_tables():
    import gc

    from pandera.backends.pandas.conditional import (
        ConditionalValidationContext,
    )

    n_rows, n_cols = 2_000, 30
    df = pd.DataFrame(
        {f"c{i}": np.arange(n_rows) for i in range(n_cols)}
    )
    df["flag"] = True
    checks = [
        pa.Check(
            lambda s: s >= 0, name=f"k{i}", where=lambda d: d["flag"]
        )
        for i in range(8)
    ]
    schema = pa.DataFrameSchema(
        columns={
            "c0": pa.Column(int, checks),
            **{f"c{i}": pa.Column(int) for i in range(1, n_cols)},
            "flag": pa.Column(bool),
        }
    )
    for _ in range(3):
        schema.validate(df, lazy=True)
    gc.collect()
    contexts = [
        o
        for o in gc.get_objects()
        if isinstance(o, ConditionalValidationContext)
    ]
    assert contexts == []  # no per-call history retained anywhere

    # a live context holds one full table plus one 1-D boolean mask per
    # conditional check, never one wide table per check.
    ctx = ConditionalValidationContext(df)
    ctx.evaluate_where(checks[0])
    assert len(ctx._mask_cache) == 1
    mask = next(iter(ctx._mask_cache.values()))
    assert mask.shape == (n_rows,)
    assert mask.memory_usage(deep=True) < df.memory_usage(deep=True).sum()


# ---------------------------------------------------------------------------
# Backend rejection
# ---------------------------------------------------------------------------


def test_polars_backend_rejects_where():
    pl = pytest.importorskip("polars")
    import pandera.polars as pla

    schema = pla.DataFrameSchema(
        checks=pla.Check(
            lambda d: d, name="c", where=lambda d: d["x"]
        )
    )
    with pytest.raises(SchemaDefinitionError, match="polars"):
        schema.validate(pl.DataFrame({"x": [1, 2]}), lazy=True)
