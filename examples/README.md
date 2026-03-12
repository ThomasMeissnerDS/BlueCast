# BlueCast Examples

Self-contained example scripts that demonstrate BlueCast's feature set using
synthetic data. No external datasets required.

| Script | Topics |
|--------|--------|
| [00_full_showcase.py](00_full_showcase.py) | **End-to-end walkthrough of all features** |
| [01_quick_start.py](01_quick_start.py) | Binary classification, multiclass, regression, `fit_eval` |
| [02_cross_validation_and_ensembles.py](02_cross_validation_and_ensembles.py) | `BlueCastCV`, mean blending, stacking, hill climbing |
| [03_conformal_prediction.py](03_conformal_prediction.py) | Uncertainty quantification, group-conditional intervals |
| [04_linear_models.py](04_linear_models.py) | Logistic/Ridge/Lasso regression, preprocessing config |
| [05_unified_interface.py](05_unified_interface.py) | `BlueCastAuto` single entry point for all problem types |
| [06_advanced_customization.py](06_advanced_customization.py) | Custom preprocessing, XGBoost backend, drift monitoring, experiment tracking, save/load |
| [07_fairness.py](07_fairness.py) | Fairness auditing, demographic parity, equalized odds, conformal fairness |
| [08_eda.py](08_eda.py) | Univariate/bivariate plots, PCA, t-SNE, correlations, data quality, leakage detection |
| [09_bluecast_ai.py](09_bluecast_ai.py) | Multi-agent LLM-powered AutoML (requires API key) |

## Running

```bash
# Run any example
poetry run python examples/01_quick_start.py
```

All examples use fast training settings for quick execution. Increase
`hyperparameter_tuning_rounds` and `hypertuning_cv_folds` in the
`TrainingConfig` for better model quality.
