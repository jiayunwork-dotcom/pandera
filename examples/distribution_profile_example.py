"""Distribution profiles: API, deterministic error bounds, and quality gates.

This example shows how to:

1. Build a compact distribution profile from reference batches.
2. Query deterministic bounds on the empirical CDF.
3. Compare a complete follow-up batch against the reference profile.
4. Wire the profile into a schema check and a cross-batch quality gate, and
   persist/reload it without keeping the reference data around.

Run with::

    python examples/distribution_profile_example.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import pandera.pandas as pa
from pandera.errors import (
    DistributionCheckError,
    DistributionUncertainError,
    SchemaErrors,
)

EPSILON = 0.01
TOLERANCE = 0.05


def reference_batches(seed: int = 0, batches: int = 5, size: int = 20_000):
    """Yield reference batches lazily (single pass, no retention)."""
    rng = np.random.default_rng(seed)
    for _ in range(batches):
        yield pd.Series(rng.normal(loc=100.0, scale=15.0, size=size))


def demonstrate_api() -> pa.NumericProfile:
    """Build a profile and inspect its deterministic bounds."""
    profile = pa.NumericProfile.from_batches(
        reference_batches(), epsilon=EPSILON
    )
    print(f"profile: {profile}")
    for threshold in (70.0, 100.0, 130.0):
        lower, upper = profile.cdf_bounds(threshold)
        print(
            f"  F({threshold}) in [{lower:.4f}, {upper:.4f}] "
            f"(width {upper - lower:.4f} <= {2 * EPSILON})"
        )
    return profile


def demonstrate_compare(profile: pa.NumericProfile) -> None:
    """Compare follow-up batches at three confidence levels."""
    rng = np.random.default_rng(1)
    same = pd.Series(rng.normal(loc=100.0, scale=15.0, size=20_000))
    drifted = pd.Series(rng.normal(loc=115.0, scale=15.0, size=20_000))

    for label, batch in (("same", same), ("drifted", drifted)):
        result = profile.compare(batch, tolerance=TOLERANCE)
        print(f"  compare({label}) -> {json.dumps(result)}")


def quality_gate(profile: pa.NumericProfile):
    """A schema-level gate that distinguishes pass, fail and uncertain."""
    check = pa.Check.distribution_within(profile, tolerance=TOLERANCE)
    return pa.DataFrameSchema({"measurement": pa.Column("float64", check)})


def demonstrate_gate(profile: pa.NumericProfile) -> None:
    schema = quality_gate(profile)
    rng = np.random.default_rng(2)
    same = pd.DataFrame({"measurement": rng.normal(100.0, 15.0, size=5_000)})
    drifted = pd.DataFrame(
        {"measurement": rng.normal(140.0, 15.0, size=5_000)}
    )

    print(f"  same batch passes: {len(schema.validate(same))} rows")

    for label, frame in (("drifted", drifted), ("drifted-lazy", drifted)):
        try:
            schema.validate(frame, lazy=label.endswith("lazy"))
            print(f"  {label}: unexpectedly passed")
        except SchemaErrors as error:
            first = error.schema_errors[0]
            print(
                f"  {label}: {type(first).__name__} "
                f"distance in [{first.distance_lower:.3f}, "
                f"{first.distance_upper:.3f}] vs tolerance {first.tolerance}"
            )
        except (DistributionCheckError, DistributionUncertainError) as error:
            print(f"  {label}: {type(error).__name__} {error.status}")


def demonstrate_inference_and_persistence(profile: pa.NumericProfile) -> None:
    """Infer a schema, save it, and reload it in a fresh schema object."""
    frame = pd.DataFrame(
        {
            "measurement": np.random.default_rng(3).normal(
                100.0, 15.0, size=20_000
            ),
            "other": np.arange(20_000),  # unchanged min/max inference
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "schema.json"
        schema = pa.infer_schema(
            frame,
            distribution={
                "columns": ["measurement"],
                "epsilon": EPSILON,
                "tolerance": TOLERANCE,
            },
        )
        schema.to_json(path)
        reloaded = pa.DataFrameSchema.from_json(path)
        check = reloaded.columns["measurement"].checks[0]
        print(f"  reloaded check: {check.name}")
        print(f"  reference n: {check.statistics['profile']['n']}")
        # The reference data is no longer needed: the check still works.
        print(f"  reloaded validates: {len(reloaded.validate(frame))} rows")


def main() -> None:
    print("== API and deterministic bounds ==")
    profile = demonstrate_api()
    print("== compare ==")
    demonstrate_compare(profile)
    print("== schema quality gate ==")
    demonstrate_gate(profile)
    print("== inference and persistence ==")
    demonstrate_inference_and_persistence(profile)


if __name__ == "__main__":
    main()
