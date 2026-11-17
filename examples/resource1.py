import tracemalloc
import pandera.pandas as pa
import pandas as pd
import numpy as np

# wide table, many checks; measure peak incremental memory and verify no
# per-check full-table retention after validate returns.
n_rows, n_cols = 20_000, 50
df = pd.DataFrame({f"c{i}": np.arange(n_rows) for i in range(n_cols)})
df["flag"] = np.arange(n_rows) % 2 == 0

checks = [
    pa.Check(lambda s: s >= 0, name=f"c{i}", where=lambda d: d["flag"])
    for i in range(20)
]
schema = pa.DataFrameSchema(
    columns={f"c{i}": pa.Column(int, checks if i == 0 else None) for i in range(n_cols - 1)} | {"flag": pa.Column(bool)},
)

tracemalloc.start()
snap1 = tracemalloc.take_snapshot()
try:
    schema.validate(df, lazy=True)
except pa.errors.SchemaErrors:
    pass
snap2 = tracemalloc.take_snapshot()
stats = snap2.compare_to(snap1, "lineno")
leak = sum(s.size_diff for s in stats)
print(f"net retained bytes after validate: {leak:,}")

# context object should not be reachable from the schema afterwards
import gc
gc.collect()
objs = [o for o in gc.get_objects() if type(o).__name__ == "ConditionalValidationContext"]
print("live ConditionalValidationContext objects:", len(objs))

# mask cache size is one boolean array per conditional check, O(n_rows * n_checks)
# which is explicitly allowed; no full wide table per check.
from pandera.backends.pandas.conditional import ConditionalValidationContext
ctx = ConditionalValidationContext(df)
c = checks[0]
m = ctx.evaluate_where(c)
print("mask bytes:", m.memory_usage(deep=True), "for", n_rows, "rows;",
      "full table bytes approx:", df.memory_usage(deep=True).sum())
assert ctx.where_call_count(c) == 1
ctx.evaluate_where(c)
assert ctx.where_call_count(c) == 1
print("single evaluation per validate context OK")
