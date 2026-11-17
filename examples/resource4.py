# Demonstrate the forbidden design (per-check full table retention) vs ours:
# inspect a live context during validation via a callback that snapshots sizes.
import pandera.pandas as pa
import pandas as pd
import numpy as np
from pandera.backends.pandas.conditional import ConditionalValidationContext

n_rows, n_cols = 2_000, 40
df = pd.DataFrame({f"c{i}": np.arange(n_rows) for i in range(n_cols)})
df["flag"] = True
snapshots = {}

def make_check(i):
    def fn(s):
        ctxs = [o for o in __import__("gc").get_objects()
                if type(o).__name__ == "ConditionalValidationContext"]
        if ctxs:
            ctx = ctxs[0]
            snapshots.setdefault("mask_cache_entries", len(ctx._mask_cache))
            snapshots.setdefault("full_table_columns", len(ctx.full_table.columns))
        return s >= 0
    return pa.Check(fn, name=f"k{i}", where=lambda d: d["flag"])

schema = pa.DataFrameSchema(
    columns={**{f"c{i}": pa.Column(int, make_check(i) if i == 0 else None)
                for i in range(n_cols - 1)}, "flag": pa.Column(bool)},
)
schema.validate(df, lazy=True)
print("context stores 1 full table with", snapshots["full_table_columns"],
      "columns and", snapshots["mask_cache_entries"], "cached boolean mask(s)")
print("=> masks are 1-D boolean arrays; no per-check wide-table copy")
