---
file_format: mystnb
---

```{currentmodule} pandera
```

(schema-inference)=

# Schema Inference and Persistence

*New in version 0.4.0*

With simple use cases, writing a schema definition manually is pretty
straight-forward with pandera. However, it can get tedious to do this with
dataframes that have many columns of various data types.

## Inferring a schema from data

To help you handle these cases, the {func}`~pandera.schema_inference.pandas.infer_schema` function enables
you to quickly infer a draft schema from a pandas dataframe or series. Below
is a simple example:

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

df = pd.DataFrame({
    "column1": [5, 10, 20],
    "column2": ["a", "b", "c"],
    "column3": pd.to_datetime(["2010", "2011", "2012"]),
})
schema = pa.infer_schema(df)
print(schema)
```

These inferred schemas are **rough drafts** that shouldn't be used for
validation without modification. You can modify the inferred schema to
obtain the schema definition that you're satisfied with.

For {class}`~pandera.api.pandas.container.DataFrameSchema` objects, the following methods create
modified copies of the schema:

- {func}`~pandera.api.pandas.container.DataFrameSchema.add_columns`
- {func}`~pandera.api.pandas.container.DataFrameSchema.remove_columns`
- {func}`~pandera.api.pandas.container.DataFrameSchema.update_column`

For {class}`~pandera.api.pandas.array.SeriesSchema` objects:

- {func}`~pandera.api.pandas.array.SeriesSchema.set_checks`

The section below describes two workflows for persisting and modifying an
inferred schema.

(schema-persistence)=

## Persisting a schema

The schema persistence feature requires a pandera installation with the `io`
extension. See the {ref}`installation<installation>` instructions for more
details.

There are two ways of persisting schemas, inferred or otherwise.

### Write to a Python script

You can also write your schema to a python script with {func}`~pandera.io.to_script`:

```{code-cell} python
# supply a file-like object, Path, or str to write to a file. If not
# specified, to_script will output the code as a string.
schema_script = schema.to_script()
print(schema_script)
```

As a python script, you can iterate on an inferred schema and use it to
validate data once you are satisfied with your schema definition.

### Write to YAML

You can also write the schema object to a yaml file with {func}`~pandera.io.to_yaml`,
and you can then read it into memory with {func}`~pandera.io.from_yaml`. The
{func}`~pandera.api.pandas.container.DataFrameSchema.to_yaml` and {func}`~pandera.api.pandas.container.DataFrameSchema.from_yaml`
is a convenience method for this functionality.

```{code-cell} python
# supply a file-like object, Path, or str to write to a file. If not
# specified, to_yaml will output a yaml string.
yaml_schema = schema.to_yaml()
print(yaml_schema)
```

You can edit this yaml file to modify the schema. For example, you can specify
new column names under the `column` key, and the respective values map onto
key-word arguments in the {class}`~pandera.api.pandas.components.Column` class.

```{note}
Currently, only built-in {class}`~pandera.api.checks.Check` methods are supported under the
`checks` key.
```

### Write to JSON

Finally, you can also write the schema object to a json file with {func}`~pandera.io.to_json`,
and you can then read it into memory with {func}`~pandera.io.from_json`. The
{func}`~pandera.api.pandas.container.DataFrameSchema.to_json` and {func}`~pandera.api.pandas.container.DataFrameSchema.from_json`
is a convenience method for this functionality.

```{code-cell} python
# supply a file-like object, Path, or str to write to a file. If not
# specified, to_yaml will output a yaml string.
json_schema = schema.to_json(indent=4)
print(json_schema)
```

You can edit this json file to update the schema as needed, and then load
it back into a pandera schema object with {func}`~pandera.io.from_json` or
{func}`~pandera.api.pandas.container.DataFrameSchema.from_json`.

(distribution-profiles)=

## Distribution profiles

The min/max bounds that `infer_schema` infers for numeric columns are cheap to
store but blind to everything between the extremes: a column whose minimum and
maximum are unchanged can still have drifted substantially. When you want to
detect that kind of drift without keeping full reference data around, use a
{class}`~pandera.numeric_profile.NumericProfile`.

A profile summarises a stream of numeric observations with a **deterministic**
error bound on the empirical CDF. It retains only a compact sketch (never the
observations), and it can be serialised alongside the rest of a schema.

### Build a profile from reference batches

The reference data may arrive in several batches. `from_batches` traverses the
source exactly once and only accepts `int64`, `float64` and pandas' nullable
`Int64`/`Float64` dtypes. Missing values are ignored; all other values must be
finite.

```{code-cell} python
import numpy as np
import pandas as pd
import pandera.pandas as pa

reference_batches = [
    pd.Series(np.arange(0, 50_000, dtype="float64")),
    pd.Series(np.arange(50_000, 100_000, dtype="float64")),
]
profile = pa.NumericProfile.from_batches(reference_batches, epsilon=0.01)
print(profile)
print("F(50_000) is in", profile.cdf_bounds(50_000))
```

For any finite threshold `t`, `cdf_bounds(t)` returns `[L, U]` with
`0 <= L <= F(t) <= U <= 1` and `U - L <= 2 * epsilon`, where `F` is the
empirical CDF of the observations actually fed to `from_batches`. These are
deterministic guarantees about the real input, not random confidence
intervals. Different batch layouts or row orders may produce different
summaries, but they all satisfy the same guarantee.

### Compare a follow-up batch

`compare` scores a complete follow-up batch. The true distance `D` is the
maximum absolute difference between the two empirical CDFs; the returned
mapping contains `distance_lower`, `distance_upper` and a `status`:

```{code-cell} python
follow_up = pd.Series(np.arange(0, 100_000, dtype="float64"))
result = profile.compare(follow_up, tolerance=0.02)
print(result)
```

* `pass` when `distance_upper <= tolerance`,
* `fail` when `distance_lower > tolerance`,
* `uncertain` otherwise.

The bounds always satisfy `0 <= distance_lower <= D <= distance_upper <= 1`
and their width is at most `4 * epsilon`. No p-value is computed.

### Encode the check in a schema

`Check.distribution_within` turns a profile into a regular column check. It is a
*batch-level* check: it passes as a whole, fails as a whole, or reports an
`uncertain` outcome as a {class}`~pandera.errors.DistributionUncertainError`.
It sees the column after the usual schema preprocessing, dtype coercion and
column parsers, so it compares the distribution that is actually validated.

```{code-cell} python
check = pa.Check.distribution_within(profile, tolerance=0.02)
schema = pa.DataFrameSchema({"value": pa.Column("float64", check)})
```

The check deliberately rejects options that would silently change the compared
population (`where`, `groupby`, `element_wise`, and non-default
`head`/`tail`/`sample`). A confirmed `fail` respects `raise_warning=True`;
an `uncertain` outcome always blocks, and `drop_invalid_rows=True` raises
instead of guessing which rows to remove.

You can also ask `infer_schema` to build the check for you. Pass the
keyword-only `distribution` configuration, which must contain exactly
`columns`, `epsilon` and `tolerance`:

```{code-cell} python
inferred = pa.infer_schema(
    pd.DataFrame({"value": np.arange(0, 100_000, dtype="float64")}),
    distribution={
        "columns": ["value"],
        "epsilon": 0.01,
        "tolerance": 0.02,
    },
)
print(inferred)
```

For the selected columns the profile check replaces the usual min/max checks;
other columns, the index, nullability and dtype inference are unchanged. The
same statistics can be reconstructed with
{func}`~pandera.schema_statistics.pandas.parse_check_statistics`, and the
profile travels with the schema through YAML/JSON export and import. A saved
schema can be reloaded in a new process and will reproduce the same profile
counts, bounds and decisions without the reference data.

```{code-cell} python
json_schema = inferred.to_json()
reloaded = pa.DataFrameSchema.from_json(json_schema)
print(reloaded.columns["value"].checks[0].name)
```
