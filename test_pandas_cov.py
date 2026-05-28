import pandas as pd
import numpy as np

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
