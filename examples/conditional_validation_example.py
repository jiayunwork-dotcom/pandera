"""Reproducible example of conditional validation with ``Check(where=...)``.

Run with::

    python examples/conditional_validation_example.py

The scenario mirrors a common data-quality rule: settled records must carry a
positive amount, while unsettled records are exempt. The example shows the
full contract:

* a column-level conditional check and a table-level conditional check,
* exact physical-row attribution via ``row_position`` (including repeated
  index labels, where label-based dropping would remove the wrong rows),
* the distinction between limited diagnostics (``n_failure_cases``) and the
* complete set of rows removed by ``drop_invalid_rows``,
* sampling windows intersected with the condition, and
* a condition that raises, which is reported as a validation error instead
  of silently passing.
"""

import pandas as pd

import pandera.pandas as pa


def main() -> None:
    # ------------------------------------------------------------------
    # Input batch. Note the duplicated index label "1": the two physical
    # rows at positions 1 and 2 are independent records even though they
    # carry the same label (and, here, the same value).
    # ------------------------------------------------------------------
    frame = pd.DataFrame(
        {
            "settled": [True, False, True, False, True],
            "amount": [10.0, -5.0, -20.0, -1.0, 30.0],
        },
        index=["a", "b", "c", "d", "e"],
    )

    schema = pa.DataFrameSchema(
        columns={
            "amount": pa.Column(
                float,
                pa.Check(
                    lambda s: s > 0,
                    name="positive_when_settled",
                    where=lambda table: table["settled"],
                ),
            ),
            "settled": pa.Column(bool),
        },
        drop_invalid_rows=True,
    )

    print("Input batch:")
    print(frame)

    # ------------------------------------------------------------------
    # Lazy diagnostics: row_position is the 0-based physical offset in the
    # preprocessed table; index is still the original label.
    # ------------------------------------------------------------------
    report_schema = pa.DataFrameSchema(
        columns={
            "amount": pa.Column(
                float,
                pa.Check(
                    lambda s: s > 0,
                    name="positive_when_settled",
                    where=lambda table: table["settled"],
                ),
            ),
            "settled": pa.Column(bool),
        }
    )
    try:
        report_schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as err:
        print("\nFailure cases (one diagnostic per failing physical row):")
        print(err.failure_cases)

    # ------------------------------------------------------------------
    # Dropping: only physical row 2 is removed; the exempt rows and all
    # other rows survive with their original order and index.
    # ------------------------------------------------------------------
    cleaned = schema.validate(frame, lazy=True)
    print("\nCleaned batch:")
    print(cleaned)

    # ------------------------------------------------------------------
    # n_failure_cases limits diagnostics only: even with one displayed
    # case, every failing physical row is still dropped.
    # ------------------------------------------------------------------
    all_bad = pd.DataFrame(
        {"settled": [True] * 4, "amount": [-1.0, -2.0, -3.0, -4.0]}
    )
    limited_schema = pa.DataFrameSchema(
        columns={
            "amount": pa.Column(
                float,
                pa.Check(
                    lambda s: s > 0,
                    name="positive_when_settled",
                    n_failure_cases=1,
                    where=lambda table: table["settled"],
                ),
            ),
            "settled": pa.Column(bool),
        },
        drop_invalid_rows=True,
    )
    dropped = limited_schema.validate(all_bad, lazy=True)
    assert len(dropped) == 0  # all four rows removed, not just the one shown
    print("\nn_failure_cases=1 displayed one case but removed all 4 rows.")

    # ------------------------------------------------------------------
    # Sampling: the window is chosen on the full table then intersected
    # with the mask; unchecked rows remain in the output.
    # ------------------------------------------------------------------
    sampled_schema = pa.DataFrameSchema(
        columns={
            "amount": pa.Column(
                float,
                pa.Check(
                    lambda s: s > 0,
                    name="positive_when_settled",
                    where=lambda table: table["settled"],
                ),
            ),
            "settled": pa.Column(bool),
        }
    )
    passing = pd.DataFrame(
        {"settled": [True] * 6, "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    result = sampled_schema.validate(
        passing, sample=3, random_state=7, lazy=True
    )
    assert len(result) == 6
    print("Sampling 3 of 6 rows left all 6 rows in the returned table.")

    # ------------------------------------------------------------------
    # A broken condition is a validation error, never an implicit pass.
    # ------------------------------------------------------------------
    broken_schema = pa.DataFrameSchema(
        checks=pa.Check(
            lambda table: table["amount"] > 0,
            name="with_broken_condition",
            where=lambda table: 1 / 0,
        )
    )
    try:
        broken_schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as err:
        print(
            "\nBroken where reported as:",
            err.schema_errors[0].reason_code.name,
        )


if __name__ == "__main__":
    main()
