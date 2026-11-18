"""Module for inferring the statistics of pandas objects."""

import warnings
from typing import Any, Dict, List, Mapping, Union

import pandas as pd

from pandera import dtypes
from pandera.api.checks import Check
from pandera.engines import pandas_engine
from pandera.errors import SchemaDefinitionError
from pandera.numeric_profile import (
    NumericProfile,
    _is_supported_numeric_dtype,
    validate_epsilon,
    validate_tolerance,
)

DISTRIBUTION_KEYS = frozenset({"columns", "epsilon", "tolerance"})


def _validate_distribution_config(config: Any) -> Dict[str, Any]:
    """Validate the ``distribution`` inference configuration."""
    if not isinstance(config, Mapping):
        raise TypeError(
            "distribution must be a mapping with keys 'columns', 'epsilon' "
            f"and 'tolerance', got {type(config).__name__}."
        )
    keys = set(config.keys())
    if keys != DISTRIBUTION_KEYS:
        raise ValueError(
            "distribution config must contain exactly the keys 'columns', "
            f"'epsilon' and 'tolerance', got {sorted(keys)}."
        )
    columns = config["columns"]
    if isinstance(columns, (str, bytes)) or not isinstance(
        columns, (list, tuple)
    ):
        raise TypeError(
            "distribution['columns'] must be a sequence of column names, "
            f"got {type(columns).__name__}."
        )
    if not columns:
        raise ValueError("distribution['columns'] must not be empty.")
    seen = set()
    for name in columns:
        if not isinstance(name, str):
            raise TypeError(
                "distribution['columns'] must contain strings, got "
                f"{type(name).__name__}."
            )
        if name in seen:
            raise ValueError(
                f"distribution['columns'] contains duplicate name {name!r}."
            )
        seen.add(name)
    return {
        "columns": list(columns),
        "epsilon": validate_epsilon(config["epsilon"]),
        "tolerance": validate_tolerance(config["tolerance"]),
    }


def _selected_column_profile(
    series: pd.Series, column_name: str, epsilon: float
) -> NumericProfile:
    if not _is_supported_numeric_dtype(series.dtype):
        raise SchemaDefinitionError(
            "distribution profiling only supports int64, float64, nullable "
            f"Int64 and Float64 columns; column {column_name!r} has dtype "
            f"{series.dtype!r}."
        )
    try:
        return NumericProfile.from_batches([series], epsilon=epsilon)
    except (TypeError, ValueError) as exc:
        raise SchemaDefinitionError(
            f"cannot build a distribution profile for column "
            f"{column_name!r}: {exc}"
        ) from exc


def infer_dataframe_statistics(
    df: pd.DataFrame, *, distribution: Any = None
) -> Dict[str, Any]:
    """Infer column and index statistics from a pandas DataFrame.

    :param df: dataframe to infer statistics from.
    :param distribution: optional keyword-only configuration with exactly the
        keys ``columns``, ``epsilon`` and ``tolerance``. For each selected
        numeric column the inferred checks are replaced by a single
        distribution-profile check instead of the usual min/max bounds.
    """
    config = None
    if distribution is not None:
        config = _validate_distribution_config(distribution)
        column_names = list(df.columns)
        duplicated = {
            name for name in config["columns"] if column_names.count(name) > 1
        }
        if duplicated:
            raise SchemaDefinitionError(
                "distribution columns must be unambiguous; the following "
                f"names occur more than once: {sorted(duplicated)}."
            )
        for name in config["columns"]:
            if name not in column_names:
                raise SchemaDefinitionError(
                    f"distribution column {name!r} does not exist in the "
                    "dataframe."
                )

    nullable_columns = df.isna().any()
    inferred_column_dtypes = {col: _get_array_type(df[col]) for col in df}
    column_statistics = {
        col: {
            "dtype": dtype,
            "nullable": bool(nullable_columns[col]),  # type: ignore
            "checks": _get_array_check_statistics(df[col], dtype),
        }
        for col, dtype in inferred_column_dtypes.items()
    }

    if config is not None:
        for name in config["columns"]:
            profile = _selected_column_profile(
                df[name], name, config["epsilon"]
            )
            column_statistics[name]["checks"] = {
                "distribution_within": {
                    "profile": profile.to_dict(),
                    "tolerance": config["tolerance"],
                }
            }

    return {
        "columns": column_statistics if column_statistics else None,
        "index": infer_index_statistics(df.index),
    }


def infer_series_statistics(
    series: pd.Series, *, distribution: Any = None
) -> Dict[str, Any]:
    """Infer column and index statistics from a pandas Series."""
    if distribution is not None:
        raise SchemaDefinitionError(
            "distribution profiling is only supported for DataFrameSchema "
            "inference; a Series has no column names to select."
        )
    dtype = _get_array_type(series)
    return {
        "dtype": dtype,
        "nullable": bool(series.isna().any()),
        "checks": _get_array_check_statistics(series, dtype),
        "name": series.name,
    }


def infer_index_statistics(index: Union[pd.Index, pd.MultiIndex]):
    """Infer index statistics given a pandas Index object."""

    def _index_stats(index_level):
        dtype = _get_array_type(index_level)
        return {
            "dtype": dtype,
            "nullable": bool(index_level.isna().any()),
            "checks": _get_array_check_statistics(index_level, dtype),
            "name": index_level.name,
        }

    if isinstance(index, pd.MultiIndex):
        index_statistics = [
            _index_stats(index.get_level_values(i))
            for i in range(index.nlevels)
        ]
    elif isinstance(index, pd.Index):
        index_statistics = [_index_stats(index)]
    else:
        warnings.warn(
            f"index type {type(index)} not recognized, skipping index inference",
            UserWarning,
        )
        index_statistics = []
    return index_statistics if index_statistics else None


def parse_check_statistics(check_stats: Union[Dict[str, Any], None]):
    """Convert check statistics to a list of Check objects, including their options.

    The input mapping is never modified: options are read from a copy, so a
    statistics mapping can be parsed more than once (for example to construct
    an identical schema repeatedly).
    """
    if check_stats is None:
        return None
    checks = []
    for check_name, stats in check_stats.items():
        check = getattr(Check, check_name)
        try:
            # Extract options if present, without mutating the input.
            if isinstance(stats, dict):
                stats = dict(stats)
                options = stats.pop("options", {}) or {}
                if stats:  # If there are remaining stats
                    check_instance = check(**stats)
                else:  # Handle case where all stats were in options
                    check_instance = check()
                # Apply options to the check instance
                for option_name, option_value in options.items():
                    if option_name == "check_name":
                        continue
                    setattr(check_instance, option_name, option_value)
                checks.append(check_instance)
            else:
                # Handle unary check case
                checks.append(check(stats))
        except TypeError:
            # if stats cannot be unpacked as key-word args, assume unary check.
            checks.append(check(stats))
    return checks if checks else None


def get_dataframe_schema_statistics(dataframe_schema):
    """Get statistical properties from dataframe schema."""
    statistics = {
        "columns": {
            col_name: {
                "dtype": column.dtype,
                "nullable": column.nullable,
                "coerce": column.coerce,
                "required": column.required,
                "regex": column.regex,
                "checks": parse_checks(column.checks),
                "unique": column.unique,
                "description": column.description,
                "title": column.title,
            }
            for col_name, column in dataframe_schema.columns.items()
        },
        "checks": parse_checks(dataframe_schema.checks),
        "index": (
            None
            if dataframe_schema.index is None
            else get_index_schema_statistics(dataframe_schema.index)
        ),
        "coerce": dataframe_schema.coerce,
    }
    return statistics


def _get_series_base_schema_statistics(series_schema_base):
    return {
        "dtype": series_schema_base.dtype,
        "nullable": series_schema_base.nullable,
        "checks": parse_checks(series_schema_base.checks),
        "coerce": series_schema_base.coerce,
        "name": series_schema_base.name,
        "unique": series_schema_base.unique,
        "title": series_schema_base.title,
        "description": series_schema_base.description,
    }


def get_index_schema_statistics(index_schema_component):
    """Get statistical properties of index schema component."""
    try:
        # get index components from MultiIndex
        index_components = index_schema_component.indexes
    except AttributeError:
        index_components = [index_schema_component]
    return [
        _get_series_base_schema_statistics(index_component)
        for index_component in index_components
    ]


def get_series_schema_statistics(series_schema):
    """Get statistical properties from series schema."""
    return _get_series_base_schema_statistics(series_schema)


def parse_checks(checks) -> Union[List[Dict[str, Any]], None]:
    """Convert Check object to check statistics including options."""
    check_statistics = []

    for check in checks:
        if check not in Check:
            warnings.warn(
                "Only registered checks may be serialized to statistics. "
                "Did you forget to register it with the extension API? "
                f"Check `{check.name}` will be skipped."
            )
            continue

        # Get base statistics. Copy the mapping so that serialising a schema
        # never mutates the Check (and, in particular, never mutates a stored
        # distribution profile or its configuration).
        base_stats = {} if check.statistics is None else dict(check.statistics)

        # Collect check options
        check_options = {
            "check_name": check.name,
            "raise_warning": check.raise_warning,
            "n_failure_cases": check.n_failure_cases,
            "ignore_na": check.ignore_na,
        }

        # Filter out None values from options
        check_options = {
            k: v for k, v in check_options.items() if v is not None
        }

        # Combine statistics with options
        if check_options:
            base_stats["options"] = check_options
            check_statistics.append(base_stats)

    # Check for incompatible checks
    incompatibile_checks = {
        "equal_to": "eq",
        "greater_than": "gt",
        "less_than": "lt",
        "greater_than_or_equal_to": "lt",
        "less_than_or_equal_to": "le",
        "in_range": "between",
    }

    incompatibile_checks_count = sum(
        map(
            lambda check: check["options"]["check_name"]
            in incompatibile_checks,
            check_statistics,
        )
    )
    if incompatibile_checks_count > 1:
        error_message = ", ".join(
            [f"{key} ({value})" for key, value in incompatibile_checks.items()]
        )
        warnings.warn(
            "You do not need more than one check out of " f"{error_message}."
        )

    return check_statistics if check_statistics else None


def _get_array_type(x):
    # get most granular type possible

    data_type = pandas_engine.Engine.dtype(x.dtype)
    # for object arrays, try to infer dtype
    if data_type is pandas_engine.Engine.dtype("object"):
        inferred_alias = pd.api.types.infer_dtype(x, skipna=False)
        if inferred_alias != "string":
            data_type = pandas_engine.Engine.dtype(inferred_alias)
    return data_type


def _get_array_check_statistics(
    x, data_type: dtypes.DataType
) -> Union[Dict[str, Any], None]:
    """Get check statistics from an array-like object."""
    if x.isna().all():
        return None
    if dtypes.is_datetime(data_type):
        check_stats = {
            "greater_than_or_equal_to": x.min(),
            "less_than_or_equal_to": x.max(),
        }
    elif dtypes.is_numeric(data_type) and not dtypes.is_bool(data_type):
        check_stats = {
            "greater_than_or_equal_to": float(x.min()),
            "less_than_or_equal_to": float(x.max()),
        }
    elif dtypes.is_category(data_type):
        try:
            categories = x.cat.categories
        except AttributeError:
            categories = x.categories
        check_stats = {
            "isin": categories.tolist(),
        }
    else:
        check_stats = {}
    return check_stats if check_stats else None
