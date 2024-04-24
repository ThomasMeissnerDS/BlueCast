# Model explainability (XAI)

As BlueCast aims to support data scientist for real world applications,
it offers various insights around the model itself.

<!-- toc -->

* [Model explainability (XAI)](#model-explainability-xai)
  * [Hyperparameter tuning](#hyperparameter-tuning)
  * [Model performance](#model-performance)
  * [Shap values](#shap-values)
  * [Plotting decision trees](#plotting-decision-trees)
      * [Accessing the trained models](#accessing-the-trained-models)
          * [BlueCast classes](#bluecast-classes)
          * [BlueCastCV classes](#bluecastcv-classes)

<!-- tocstop -->

## Hyperparameter tuning

After hyperparameter tuning the progress of the loss metric is shown.
This is useful to estimate if less tuning rounds would be sufficient
in future runs or if even more tuning rounds are required.

![Optuna loss progress example](docs/source/optuna_loss_progress.png)

Additionally, the most important hyperparameters are visualised. This can
be used to decide which parameters to include in an extra GridSearch layer.

![Optuna most important parameters example](docs/source/optuna_most_important_parameters.png)

## Model performance

When building a model, knowing the model performance is crucial to make model
and business decisions. Therefore all BlueCast classes offer a `fit_eval` function,
that tunes and trains a model like the `fit` method, and evaluates the model
on unseen data. At the end a confusion matrix and a list of metrics are shown:

![Model performance metrics](docs/source/model_performance.png)

For CV classes the mean performance on all unseen dataset and the standard
deviation of those will be shown as well.

![Model performance CV metrics](docs/source/model_performance_cv.png)

## SHAP values

While understanding the tuning process is helpful, we also want to understand
how the model behaves and which features are particularly important. Thus
BlueCast makes use of SHAP values. First it will show the global feature importance:

![SHAP global feature importance](docs/source/shap_global_importance.png)

Furthermore it will also show a sample on row level and plot waterfall plots of
SHAP values for each target clas of defined sample indices. User can change the
training config and adjust the `shap_waterfall_indices` parameter to pass a list
of indices instead.

![SHAP waterfall chart](docs/source/shap_waterfall_chart.png)

The third visualization is the dependence plot, which shows feature interactions
and their impact on SHAP values on a certain feature. By default BlueCast shows
dependence plots for the top 5 features for each class. The number of features
can adjusted via customization of the `show_dependence_plots_of_top_n_features`
param in the training config.

![SHAP dependence plot](docs/source/shap_dependence_plot.png)

The SHAP functions are part of all BlueCast classes and are called when using
the `fit` method.

This toolkit can also be used after model training for new data or for an external
model.

```python
from bluecast.evaluation.shap_values import (
    shap_dependence_plots,
    shap_explanations,
    shap_waterfall_plot,
)

shap_waterfall_indices = [2, 35, 999]
class_problem = "binary" # or 'multiclass' or 'regression'
show_dependence_plots_of_top_n_features = 5

shap_values, explainer = shap_explanations(automl.ml_model.model, x_test)

shap_waterfall_plot(
  explainer, shap_waterfall_indices, class_problem
)
shap_dependence_plots(
  shap_values,
  x_test,
  conf_training.show_dependence_plots_of_top_n_features,
)
```

## Plotting decision trees

How to plot decision trees will depend on the exact implementation. As BlueCast
uses Xgboost by default, we illustrate the default way.

### Accessing the trained models

Regardless of the model architecture used we need to know where BlueCast stores
the trained model. The exact path differs between BlueCast(Regression) and
Bluecast(Regression)CV classes as CV classes are a wrapper around BlueCast classes.

#### BlueCast classes

The BlueCast classes store their model in an `ml_model` attribute where the trained
model is a `model` attribute.

```python
import matplotlib.pyplot as plt
from xgboost import plot_tree

plot_tree(automl.ml_model.model)
fig = plt.gcf()
fig.set_size_inches(150, 80)
plt.show()
```

![Xgboost tree](docs/source/xgboost_tree.png)

#### BlueCastCV classes

BlueCast have two characteristics to pay attention two:

* they are a wrapper around BlueCast classes
* they usually store multiple BlueCast instances at once

The BlueCast instances are stored inside a list attribute `bluecast_models`:

```python
import matplotlib.pyplot as plt
from xgboost import plot_tree

for model in automl.bluecast_models:
    plot_tree(model.ml_model.model)
    fig = plt.gcf()
    fig.set_size_inches(150, 80)
    plt.show()
```

This way we loop through all stored ml models and see the decision trees for each
of them. This can reveal significant differences in the trees.

![Xgboost tree](docs/source/xgboost_multi_tree.png)
