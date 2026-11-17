import pandera.pandas as pa
import pandas as pd
import numpy as np
import gc, tracemalloc

n_rows, n_cols = 5_000, 30
df = pd.DataFrame({f"c{i}": np.arange(n_rows) for i in range(n_cols)})
df["flag"] = np.arange(n_rows) % 2 == 0
schema = pa.DataFrameSchema(
    columns={
        "c0": pa.Column(int, [pa.Check(lambda s: s >= 0, name=f"k{i}",
                                      where=lambda d: d["flag"]) for i in range(10)]),
        **{f"c{i}": pa.Column(int) for i in range(1, n_cols - 1)},
        "flag": pa.Column(bool),
    },
)

def run_many(n):
    for _ in range(n):
        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors:
            pass

gc.collect()
tracemalloc.start()
run_many(1)
snap1 = tracemalloc.take_snapshot()
run_many(20)
gc.collect()
snap2 = tracemalloc.take_snapshot()
growth = sum(s.size_diff for s in snap2.compare_to(snap1, "lineno") if s.size_diff > 0)
print(f"growth after 20 more validates: {growth:,} bytes")
objs = [o for o in gc.get_objects() if type(o).__name__ == "ConditionalValidationContext"]
print("live contexts:", len(objs))
