"""Integration tests for distribution profiles across pandera."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest

import pandera.pandas as pa
from pandera.errors import (
    DistributionCheckError,
    DistributionUncertainError,
    SchemaDefinitionError,
    SchemaErrors,
    SchemaWarning,
)
from pandera.numeric_profile import NumericProfile
from pandera.schema_inference.pandas import infer_schema
from pandera.schema_statistics.pandas import (
    infer_dataframe_statistics,
    parse_check_statistics,
)


def _reference(values=None):
    values = list(range(1000)) if values is None else values
    return pd.Series(values, dtype="float64")


def _drifted():
    return _reference() + 5000.0


def _uncertain():
    values = _reference().to_numpy().copy()
    values[:100] += 10000.0  # true distance == 0.1
    return pd.Series(values, dtype="float64")


def _schema(profile, tolerance, **kwargs):
    check = pa.Check.distribution_within(profile, tolerance=tolerance)
    return pa.DataFrameSchema({"x": pa.Column("float64", check)}, **kwargs)


# ---------------------------------------------------------------------------
# Check execution semantics
# ---------------------------------------------------------------------------


def test_check_passes_for_same_distribution():
    schema = _schema(NumericProfile.from_batches([_reference()]), 0.02)
    out = schema.validate(pd.DataFrame({"x": _reference()}))
    assert len(out) == 1000


def test_confirmed_drift_raises_distribution_error():
    schema = _schema(NumericProfile.from_batches([_reference()]), 0.02)
    with pytest.raises(DistributionCheckError) as excinfo:
        schema.validate(pd.DataFrame({"x": _drifted()}))
    error = excinfo.value
    assert error.status == "fail"
    assert error.distance_lower > 0.02
    assert error.n == 1000


def test_uncertain_raises_public_error():
    schema = _schema(NumericProfile.from_batches([_reference()]), 0.105)
    with pytest.raises(DistributionUncertainError) as excinfo:
        schema.validate(pd.DataFrame({"x": _uncertain()}))
    error = excinfo.value
    assert error.status == "uncertain"
    assert error.distance_lower <= 0.105 < error.distance_upper
    assert error.n == 1000


def test_report_exposes_distance_threshold_and_n():
    profile = NumericProfile.from_batches([_reference()])
    schema = _schema(profile, 0.02)
    with pytest.raises(SchemaErrors) as excinfo:
        schema.validate(pd.DataFrame({"x": _drifted()}), lazy=True)
    error = excinfo.value.schema_errors[0]
    assert isinstance(error, DistributionCheckError)
    assert error.tolerance == 0.02
    assert error.n == profile.n
    assert 0.0 <= error.distance_lower <= error.distance_upper <= 1.0


def test_batch_failure_is_not_reported_as_per_row():
    schema = _schema(NumericProfile.from_batches([_reference()]), 0.02)
    with pytest.raises(SchemaErrors) as excinfo:
        schema.validate(pd.DataFrame({"x": _drifted()}), lazy=True)
    error = excinfo.value.schema_errors[0]
    # failure_cases is a scalar message, not one failure per row.
    assert isinstance(error.failure_cases, str)
    assert "distance" in error.failure_cases


def test_lazy_aggregation_distinguishes_uncertain():
    profile = NumericProfile.from_batches([_reference()])
    schema = _schema(profile, 0.105)
    with pytest.raises(SchemaErrors) as excinfo:
        schema.validate(pd.DataFrame({"x": _uncertain()}), lazy=True)
    errors = excinfo.value.schema_errors
    assert len(errors) == 1
    assert isinstance(errors[0], DistributionUncertainError)
    assert errors[0].reason_code.name == "CHECK_ERROR"
    assert excinfo.value.error_counts["CHECK_ERROR"] == 1


def test_raise_warning_only_downgrades_confirmed_failures():
    profile = NumericProfile.from_batches([_reference()])
    warned = pa.Check.distribution_within(
        profile, tolerance=0.02, raise_warning=True
    )
    schema = pa.DataFrameSchema({"x": pa.Column("float64", warned)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema.validate(pd.DataFrame({"x": _drifted()}))
    assert [item.category for item in caught] == [SchemaWarning]

    uncertain = pa.Check.distribution_within(
        profile, tolerance=0.105, raise_warning=True
    )
    schema = pa.DataFrameSchema({"x": pa.Column("float64", uncertain)})
    with pytest.raises(DistributionUncertainError):
        schema.validate(pd.DataFrame({"x": _uncertain()}))


# ---------------------------------------------------------------------------
# drop_invalid_rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["schema", "column"])
def test_drop_invalid_rows_never_guesses(where):
    profile = NumericProfile.from_batches([_reference()])
    check = pa.Check.distribution_within(profile, tolerance=0.02)
    if where == "schema":
        schema = pa.DataFrameSchema(
            {"x": pa.Column("float64", check)}, drop_invalid_rows=True
        )
    else:
        schema = pa.DataFrameSchema(
            {"x": pa.Column("float64", check, drop_invalid_rows=True)}
        )
    with pytest.raises(SchemaErrors) as excinfo:
        schema.validate(pd.DataFrame({"x": _drifted()}), lazy=True)
    assert isinstance(excinfo.value.schema_errors[0], DistributionCheckError)


@pytest.mark.parametrize("where", ["schema", "column"])
def test_drop_invalid_rows_blocks_uncertain(where):
    profile = NumericProfile.from_batches([_reference()])
    check = pa.Check.distribution_within(profile, tolerance=0.105)
    if where == "schema":
        schema = pa.DataFrameSchema(
            {"x": pa.Column("float64", check)}, drop_invalid_rows=True
        )
    else:
        schema = pa.DataFrameSchema(
            {"x": pa.Column("float64", check, drop_invalid_rows=True)}
        )
    with pytest.raises(SchemaErrors) as excinfo:
        schema.validate(pd.DataFrame({"x": _uncertain()}), lazy=True)
    assert isinstance(
        excinfo.value.schema_errors[0], DistributionUncertainError
    )


# ---------------------------------------------------------------------------
# preprocessing / coercion / parsers
# ---------------------------------------------------------------------------


def test_check_sees_coerced_column():
    profile = NumericProfile.from_batches([_reference()])
    check = pa.Check.distribution_within(profile, tolerance=0.02)
    schema = pa.DataFrameSchema(
        {"x": pa.Column("float64", check, coerce=True)}
    )
    data = pd.DataFrame({"x": np.arange(1000, dtype="int64")})
    assert len(schema.validate(data)) == 1000


def test_check_sees_parser_modified_values():
    profile = NumericProfile.from_batches([_reference()])
    check = pa.Check.distribution_within(profile, tolerance=0.02)
    schema = pa.DataFrameSchema(
        {"x": pa.Column("float64", check, parsers=pa.Parser(lambda s: s * 2))}
    )
    with pytest.raises(SchemaErrors) as excinfo:
        schema.validate(pd.DataFrame({"x": _reference()}), lazy=True)
    assert isinstance(excinfo.value.schema_errors[0], DistributionCheckError)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"groupby": "g"},
        {"element_wise": True},
        {"where": lambda df: df["x"] > 0},
        {"head": 10},
        {"tail": 10},
        {"sample": 10},
    ],
)
def test_population_changing_options_are_rejected(kwargs):
    profile = NumericProfile.from_batches([_reference()])
    if "head" in kwargs or "tail" in kwargs or "sample" in kwargs:
        check = pa.Check.distribution_within(profile, tolerance=0.02)
        schema = pa.DataFrameSchema({"x": pa.Column("float64", check)})
        data = pd.DataFrame({"x": _reference()})
        with pytest.raises(SchemaDefinitionError):
            schema.validate(data, **kwargs)
    else:
        with pytest.raises(SchemaDefinitionError):
            pa.Check.distribution_within(profile, tolerance=0.02, **kwargs)


def test_where_check_inside_dataframe_is_rejected():
    profile = NumericProfile.from_batches([_reference()])
    check = pa.Check.distribution_within(profile, tolerance=0.02)
    check.where = lambda df: df["x"] > 0
    schema = pa.DataFrameSchema({"x": pa.Column("float64", check)})
    with pytest.raises(SchemaDefinitionError):
        schema.validate(pd.DataFrame({"x": _reference()}))


def test_old_checks_still_work():
    schema = pa.DataFrameSchema({"x": pa.Column("float64", pa.Check.gt(0))})
    assert len(schema.validate(pd.DataFrame({"x": [1.0, 2.0]}))) == 2


# ---------------------------------------------------------------------------
# statistics -> checks
# ---------------------------------------------------------------------------


def test_statistics_reconstruct_the_same_check():
    frame = pd.DataFrame({"a": np.arange(300, dtype="int64")})
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.05}
    statistics = infer_dataframe_statistics(frame, distribution=config)
    checks = parse_check_statistics(statistics["columns"]["a"]["checks"])
    schema = infer_schema(frame, distribution=config)
    inferred = schema.columns["a"].checks[0]
    assert len(checks) == 1
    assert checks[0].name == inferred.name == "distribution_within"
    assert checks[0].statistics == inferred.statistics
    # The statistics-built check executes.
    built = pa.DataFrameSchema({"a": pa.Column("int64", checks)})
    assert len(built.validate(frame)) == 300


def test_statistics_parsing_does_not_mutate_input():
    frame = pd.DataFrame({"a": np.arange(50, dtype="int64")})
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.05}
    statistics = infer_dataframe_statistics(frame, distribution=config)
    before = json.dumps(statistics["columns"]["a"]["checks"], sort_keys=True)
    parse_check_statistics(statistics["columns"]["a"]["checks"])
    parse_check_statistics(statistics["columns"]["a"]["checks"])
    after = json.dumps(statistics["columns"]["a"]["checks"], sort_keys=True)
    assert before == after


# ---------------------------------------------------------------------------
# inference -> validation
# ---------------------------------------------------------------------------


def test_infer_schema_distribution_round_trip():
    frame = pd.DataFrame({"a": np.arange(500, dtype="float64")})
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.05}
    schema = infer_schema(frame, distribution=config)
    assert len(schema.validate(frame)) == 500
    with pytest.raises(SchemaErrors):
        schema.validate(frame.assign(a=frame["a"] + 1000), lazy=True)


def test_infer_schema_selected_column_drops_min_max_checks():
    frame = pd.DataFrame({"a": np.arange(100, dtype="int64")})
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.05}
    schema = infer_schema(frame, distribution=config)
    names = [check.name for check in schema.columns["a"].checks]
    assert names == ["distribution_within"]
    # Within-range drift is still caught even though min/max are unchanged.
    drifted = frame.assign(
        a=np.concatenate([np.arange(50), np.arange(50) + 150])
    )
    with pytest.raises(SchemaErrors):
        schema.validate(drifted, lazy=True)


def test_infer_schema_other_columns_keep_old_behaviour():
    frame = pd.DataFrame(
        {
            "a": np.arange(100, dtype="int64"),
            "b": np.arange(100, dtype="float64"),
            "c": ["x"] * 100,
        }
    )
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.05}
    schema = infer_schema(frame, distribution=config)
    assert [c.name for c in schema.columns["b"].checks] == [
        "greater_than_or_equal_to",
        "less_than_or_equal_to",
    ]
    assert not schema.columns["c"].checks
    assert schema.index is not None


@pytest.mark.parametrize(
    "config",
    [
        {"columns": ["a"], "epsilon": 0.01},
        {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.1, "x": 1},
        {"columns": [], "epsilon": 0.01, "tolerance": 0.1},
        {"columns": ["a", "a"], "epsilon": 0.01, "tolerance": 0.1},
        {"columns": ["missing"], "epsilon": 0.01, "tolerance": 0.1},
        {"columns": "a", "epsilon": 0.01, "tolerance": 0.1},
        {"columns": [1], "epsilon": 0.01, "tolerance": 0.1},
        {"columns": ["a"], "epsilon": 0.5, "tolerance": 0.1},
        {"columns": ["a"], "epsilon": 0.01, "tolerance": 2},
    ],
)
def test_invalid_distribution_config_is_rejected(config):
    frame = pd.DataFrame({"a": np.arange(10, dtype="int64")})
    with pytest.raises((TypeError, ValueError, SchemaDefinitionError)):
        infer_schema(frame, distribution=config)


def test_ambiguous_column_names_are_rejected():
    frame = pd.DataFrame(np.arange(20).reshape(10, 2), columns=["a", "a"])
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.1}
    with pytest.raises((SchemaDefinitionError, pa.errors.SchemaInitError)):
        infer_schema(frame, distribution=config)


def test_non_numeric_selected_column_is_rejected():
    frame = pd.DataFrame({"a": ["x", "y"]})
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.1}
    with pytest.raises(SchemaDefinitionError, match="only supports"):
        infer_schema(frame, distribution=config)


def test_old_inference_is_unchanged_when_distribution_is_none():
    frame = pd.DataFrame({"a": np.arange(10, dtype="int64")})
    schema = infer_schema(frame, distribution=None)
    assert [c.name for c in schema.columns["a"].checks] == [
        "greater_than_or_equal_to",
        "less_than_or_equal_to",
    ]


def test_infer_series_rejects_distribution():
    with pytest.raises(TypeError):
        infer_schema(
            pd.Series([1, 2, 3]),
            distribution={"columns": ["x"], "epsilon": 0.01, "tolerance": 0.1},
        )


# ---------------------------------------------------------------------------
# IO and cross-process reload
# ---------------------------------------------------------------------------


def test_json_and_yaml_round_trip(tmp_path):
    frame = pd.DataFrame({"a": np.arange(400, dtype="int64")})
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.05}
    schema = infer_schema(frame, distribution=config)
    profile = NumericProfile.from_dict(
        schema.columns["a"].checks[0].statistics["profile"]
    )

    json_path = tmp_path / "schema.json"
    yaml_path = tmp_path / "schema.yaml"
    schema.to_json(json_path)
    schema.to_yaml(yaml_path)

    for loaded in (
        pa.DataFrameSchema.from_json(json_path),
        pa.DataFrameSchema.from_yaml(yaml_path),
    ):
        check = loaded.columns["a"].checks[0]
        assert check.name == "distribution_within"
        assert check.statistics["tolerance"] == 0.05
        restored = NumericProfile.from_dict(check.statistics["profile"])
        assert restored.n == profile.n
        assert restored.cdf_bounds(200) == profile.cdf_bounds(200)
        assert loaded.validate(pd.DataFrame({"a": np.arange(400)})) is not None
        with pytest.raises(SchemaErrors):
            loaded.validate(
                pd.DataFrame({"a": np.arange(400) + 1000}), lazy=True
            )


def test_corrupt_schema_load_is_rejected_not_silently_dropped():
    frame = pd.DataFrame({"a": np.arange(100, dtype="int64")})
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.05}
    schema = infer_schema(frame, distribution=config)
    payload = json.loads(schema.to_json())
    payload["columns"]["a"]["checks"][0]["profile"]["version"] = 42
    with pytest.raises(ValueError, match="version"):
        pa.DataFrameSchema.from_json(json.dumps(payload))
    payload = json.loads(schema.to_json())
    payload["columns"]["a"]["checks"][0]["profile"]["n"] += 1
    with pytest.raises(ValueError, match="counts"):
        pa.DataFrameSchema.from_json(json.dumps(payload))


def test_new_process_reload(tmp_path):
    frame = pd.DataFrame({"a": np.arange(500, dtype="int64")})
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.05}
    schema = infer_schema(frame, distribution=config)
    path = tmp_path / "schema.json"
    schema.to_json(path)
    script = textwrap.dedent(f"""
        import numpy as np
        import pandas as pd
        import pandera.pandas as pa
        from pandera.errors import SchemaErrors

        schema = pa.DataFrameSchema.from_json({str(path)!r})
        check = schema.columns["a"].checks[0]
        profile = check.statistics["profile"]
        print("N", profile["n"])
        print("TOL", check.statistics["tolerance"])
        print("PASS", len(schema.validate(pd.DataFrame({{"a": np.arange(500)}}))))
        try:
            schema.validate(
                pd.DataFrame({{"a": np.arange(500) + 1000}}), lazy=True
            )
            print("DRIFT", "none")
        except SchemaErrors as err:
            print("DRIFT", type(err.schema_errors[0]).__name__)
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "N 500" in result.stdout
    assert "TOL 0.05" in result.stdout
    assert "PASS 500" in result.stdout
    assert "DRIFT DistributionCheckError" in result.stdout


# ---------------------------------------------------------------------------
# immutability and isolation
# ---------------------------------------------------------------------------


def test_repeated_and_interleaved_validations_do_not_leak():
    profile = NumericProfile.from_batches([_reference()])
    schema = _schema(profile, 0.02)
    good = pd.DataFrame({"x": _reference()})
    bad = pd.DataFrame({"x": _drifted()})
    for _ in range(3):
        assert len(schema.validate(good)) == 1000
        with pytest.raises(SchemaErrors):
            schema.validate(bad, lazy=True)
        assert len(schema.validate(good)) == 1000


def test_profile_and_inputs_are_not_mutated():
    reference = _reference()
    original = reference.copy()
    profile = NumericProfile.from_batches([reference])
    state = profile.to_dict()
    schema = _schema(profile, 0.02)
    batch = _reference()
    batch_original = batch.copy()
    schema.validate(pd.DataFrame({"x": batch}))
    pd.testing.assert_series_equal(reference, original)
    pd.testing.assert_series_equal(batch, batch_original)
    assert profile.to_dict() == state


def test_schema_serialisation_does_not_mutate_check_statistics():
    frame = pd.DataFrame({"a": np.arange(60, dtype="int64")})
    config = {"columns": ["a"], "epsilon": 0.01, "tolerance": 0.05}
    schema = infer_schema(frame, distribution=config)
    before = json.dumps(
        schema.columns["a"].checks[0].statistics, sort_keys=True
    )
    for _ in range(3):
        schema.to_json()
    after = json.dumps(
        schema.columns["a"].checks[0].statistics, sort_keys=True
    )
    assert before == after
