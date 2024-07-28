# Error analysis

Error analysis helps understanding in which subsegments of the data
an ml model performs well and in which of them it does not. BlueCast
provides utility for error analysis. The training configuration
has to be prepared for this however.

## What is needed for error analysis

First of all we import the required modules:

```python
from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.config.training_config import TrainingConfig
from bluecast.evaluation.error_analysis import ErrorAnalyserClassification, ErrorAnalyserClassificationCV
```

Next we need to instantiate a BlueCast instance:

```python
automl = BlueCastCV(class_problem="binary") # also multiclass is possible)
```

Next we must set a path to store out of fold datasts during training:

```python
out_of_fold_dataset_store_path = "/your_oof_path/"

# overwrite default settings
automl.conf_training.out_of_fold_dataset_store_path = out_of_fold_dataset_store_path
```

This can also be achieved with:

```python
train_config = TrainingConfig()
train_config.out_of_fold_dataset_store_path = "/your_oof_path/"

automl = BlueCastCV(
    class_problem="binary", # also multiclass is possible
    conf_training=train_config,
)
```

This works with all BlueCast instances.

The last step to enable error analysis is to train the pipeline. Here we need
to use the `fit_eval` method instead of the `fit` method.

```python
automl.fit_eval(train.copy(), target_col=target)
```

## Calling error analysis

After the training pipeline finished we can use the error analysis:

```python
error_analyser = ErrorAnalyserClassification(automl)
analysis_result = error_analyser.analyse_segment_errors()
```

Now we receive a Polars DataFrame showing the mean absolute prediction
errors of all subsegments in the dataset, which can be used for further analysis.

![Error analysis example](error_analysis_table.png)

Error analysis is avilaable for all regression models as well.
