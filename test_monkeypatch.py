import sys
import numpy.core._methods as np_methods
original_amax = np_methods._amax

def patched_amax(a, axis=None, out=None, keepdims=False, initial=np_methods._NoValue, where=True):
    if initial is np_methods._NoValue:
        return np_methods.umr_maximum(a, axis, None, out, keepdims, None, where)
    return np_methods.umr_maximum(a, axis, None, out, keepdims, initial, where)

np_methods._amax = patched_amax

import pandas as pd
df = pd.DataFrame({"num": [1,2,3,4], "target": [0,1,0,1]})
n_rows = len(df)
max_rows = 2
sample_df = (
    df.groupby("target", group_keys=False)
    .apply(
        lambda x: x.sample(
            n=min(len(x), max(1, int(max_rows * len(x) / n_rows))),
            random_state=42,
        )
    )
    .reset_index(drop=True)
)
print(sample_df)
