# Basic usage

The module blueprints contains the main functionality of the library. The main
entry point is the `Blueprint` class. It already includes needed preprocessing
(including some convenience functionality like feature type detection)
and model hyperparameter tuning.

```sh
from bluecast.blueprints.cast import BlueCast

automl = BlueCast(
        class_problem="binary",
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)

# from version 0.95 also predict_proba is directly available (also for BlueCastCV)
y_probs = automl.predict_proba(df_val)
```

BlueCast has simple utilities to save and load your pipeline:

```sh
from bluecast.general_utils.general_utils import save_to_production, load_for_production

# save pipeline including tracker
save_to_production(automl, "/kaggle/working/", "bluecast_cv_pipeline")

# in production or for further experiments this can be loaded again
automl = load_for_production("/kaggle/working/", "bluecast_cv_pipeline")
```

Since version 0.80 BlueCast offers regression as well:

```sh
from bluecast.blueprints.cast_regression import BlueCastRegression

automl = BlueCast(
        class_problem="regression",
    )

automl.fit(df_train, target_col="target")
y_hat = automl.predict(df_val)
```

During classification taks the target labels might arrive as strings. BlueCast
will return these as numeric representations by default at the monment.
However the `predict` method accepts a parameter `return_original_labels` to
return the original labels.

```python
from bluecast.blueprints.cast_regression import BlueCastRegression

automl = BlueCast(
        class_problem="binary",
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val, return_original_labels=True)
```

The default behaviour might change in a future release.
