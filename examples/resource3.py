import pandera.pandas as pa
import pandas as pd
import numpy as np
import gc

n_rows = 50_000
df = pd.DataFrame({"flag": [True] * n_rows, "v": np.arange(n_rows)})
schema = pa.DataFrameSchema(
    columns={"v": pa.Column(int, [pa.Check(lambda s: s >= 0, name=f"k{i}",
                                          where=lambda d: d["flag"]) for i in range(10)]),
             "flag": pa.Column(bool)},
    drop_invalid_rows=True,
)
# all rows pass; contexts must be fully released each time
import sys
def count_ctx():
    return sum(1 for o in gc.get_objects() if type(o).__name__ == "ConditionalValidationContext")
gc.collect()
print("before:", count_ctx())
for k in range(5):
    schema.validate(df, lazy=True)
    gc.collect()
    print(f"after run {k+1}:", count_ctx())
