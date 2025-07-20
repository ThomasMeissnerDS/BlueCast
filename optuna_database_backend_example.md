# Optuna Database Backend for BlueCast

This feature allows you to store Optuna hyperparameter tuning progress in a persistent SQLite database, enabling you to resume tuning if it fails or gets interrupted.

## Quick Start

```python
from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig

# Configure training with database backend
train_config = TrainingConfig()
train_config.optuna_db_backend_path = "/path/to/optuna_study.db"  # Set database path
train_config.hyperparameter_tuning_rounds = 200
train_config.autotune_model = True

# Create BlueCast instance
automl = BlueCast(
    class_problem="binary",
    conf_training=train_config,
)

# Start training - will create/use the database
automl.fit(df, target_col="target")
```

## How it Works

### Database Storage
- When `optuna_db_backend_path` is set to a string path, BlueCast uses SQLite to store all trial data
- If the database file doesn't exist, it will be created automatically
- If the database exists, BlueCast will resume from the existing study

### Sampler State Preservation
- The sampler state is automatically saved as a pickle file alongside the database
- This ensures reproducible results when resuming studies
- File location: `{database_path}_sampler.pkl`

### Study Names
- Each model type and random seed gets a unique study name:
  - XGBoost: `xgboost_tuning_seed_{random_state}`
  - CatBoost: `catboost_tuning_seed_{random_state}`
  - Grid Search: `xgboost_grid_search` or `catboost_grid_search`

## Benefits

1. **Fault Tolerance**: Resume training if interrupted
2. **Progress Tracking**: View historical trial data
3. **Reproducibility**: Consistent results across runs
4. **Analysis**: Examine optimization history

## Example: Resuming Failed Training

```python
# First attempt (may fail)
train_config = TrainingConfig()
train_config.optuna_db_backend_path = "my_study.db"
train_config.hyperparameter_tuning_rounds = 1000

automl = BlueCast("binary", conf_training=train_config)
try:
    automl.fit(df, "target")
except Exception as e:
    print(f"Training failed: {e}")

# Resume from where it left off
automl2 = BlueCast("binary", conf_training=train_config)
automl2.fit(df, "target")  # Will resume from existing database
```

## Viewing Study Results

```python
import optuna

# Load existing study
study = optuna.load_study(
    study_name="xgboost_tuning_seed_33",
    storage="sqlite:///my_study.db"
)

# View trials dataframe
df_trials = study.trials_dataframe()
print(df_trials[["number", "value", "state"]])

# Get best parameters
print("Best params:", study.best_params)
print("Best value:", study.best_value)
```

## Configuration Options

```python
train_config = TrainingConfig()

# Enable database backend
train_config.optuna_db_backend_path = "/path/to/study.db"

# Or disable (default behavior)
train_config.optuna_db_backend_path = None  # In-memory storage
```

## Notes

- Database backend works with both XGBoost and CatBoost models
- Works with both classification and regression tasks
- Compatible with cross-validation and grid search fine-tuning
- No performance impact when using database storage
- Database files are portable and can be shared between machines 