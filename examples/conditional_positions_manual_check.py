"""Hand-computed selection / failure / retention positions, then verification.

This script deliberately uses a small table with repeated index labels, a
nullable-boolean ``where`` mask (NA = not selected) and duplicate business
values, so positional attribution cannot be replaced by label matching.

Run with::

    python examples/conditional_positions_manual_check.py
"""

import pandas as pd

import pandera.pandas as pa


# ---------------------------------------------------------------------------
# Hand-computed reference
# ---------------------------------------------------------------------------
# physical position | index label | settled (mask) | amount | expected
# -------------------+-------------+----------------+--------+----------------
#         0          |     "x"     |      True      |   1    | selected, pass
#         1          |     "y"     |      False     |  -5    | not selected
#         2          |     "y"     |      <NA>      |  -9    | not selected (NA)
#         3          |     "z"     |      True      |  -3    | selected, FAIL
#         4          |     "w"     |      True      |   4    | selected, pass
#
# selected positions : [0, 3, 4]
# failing positions  : [3]
# retained positions : [0, 1, 2, 4]
EXPECTED_SELECTED = [0, 3, 4]
EXPECTED_FAILURES = [3]
EXPECTED_RETAINED = [0, 1, 2, 4]


def main() -> None:
    frame = pd.DataFrame(
        {
            # nullable boolean dtype; the NA row means "not selected".
            "settled": pd.array([True, False, None, True, True],
                                dtype="boolean"),
            "amount": [1, -5, -9, -3, 4],
        },
        index=["x", "y", "y", "z", "w"],
    )

    where_calls = {"n": 0}
    check_calls = {"n": 0}

    def condition(table):
        where_calls["n"] += 1
        return table["settled"]

    def check_fn(series):
        check_calls["n"] += 1
        return series > 0

    check = pa.Check(
        check_fn,
        name="positive_when_settled",
        where=condition,
    )
    schema = pa.DataFrameSchema(
        columns={
            "amount": pa.Column(int, check),
            "settled": pa.Column(pd.BooleanDtype(), nullable=True),
        },
        drop_invalid_rows=True,
    )

    # diagnostics first
    report_schema = pa.DataFrameSchema(
        columns={
            "amount": pa.Column(int, check),
            "settled": pa.Column(pd.BooleanDtype(), nullable=True),
        }
    )
    try:
        report_schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as err:
        observed_failures = (
            err.failure_cases["row_position"].dropna().astype(int).tolist()
        )
    else:  # pragma: no cover - defensive
        observed_failures = []

    assert observed_failures == EXPECTED_FAILURES, observed_failures

    cleaned = schema.validate(frame, lazy=True)
    # retained amount values are the full frame minus failing position 3.
    assert cleaned["amount"].tolist() == [
        frame["amount"].iloc[p] for p in EXPECTED_RETAINED
    ]

    # the where callable runs once per validate; two validates -> two calls.
    # The check function receives only selected rows (3 physical rows).
    print(f"where evaluations across two validates: {where_calls['n']}")
    assert where_calls["n"] == 2
    print(f"check function invocations: {check_calls['n']}")
    assert check_calls["n"] == 2  # one vectorized call per validate

    print("hand-computed selected positions:", EXPECTED_SELECTED)
    print("observed failing positions      :", observed_failures)
    print("hand-computed retained positions:", EXPECTED_RETAINED)
    print("cleaned frame:")
    print(cleaned)
    print("\nAll hand-computed positions match the actual results.")


if __name__ == "__main__":
    main()
