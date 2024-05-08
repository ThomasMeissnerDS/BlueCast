# Customizing configurations and objects

<!-- toc -->

* [Customizing configurations and objects](#customizing-configurations-and-objects)
  * [Custom training configuration](#custom-training-configuration)
  * [Custom preprocessing](#custom-preprocessing)
  * [Custom feature selection](#custom-feature-selection)
  * [Custom ML model](#custom-ml-model)

<!-- tocstop -->

## Custom training configuration

BlueCast allows easy customization. Users can adjust the
configuration and just pass it to the `BlueCast` class. Here is an example:

```sh
from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig

# Create a custom tuning config and adjust hyperparameter search space
xgboost_param_config = XgboostTuneParamsConfig()
xgboost_param_config.steps_max = 100
xgboost_param_config.max_leaves_max = 16
# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.hyperparameter_tuning_rounds = 10
train_config.autotune_model = False # we want to run just normal training, no hyperparameter tuning
# We could even just overwrite the final Xgboost params using the XgboostFinalParamConfig class

# Pass the custom configs to the BlueCast class
automl = BlueCast(
        class_problem="binary",
        conf_training=train_config,
        conf_xgboost=xgboost_param_config,
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

## Custom preprocessing

The `BlueCast` class also allows for custom preprocessing. This is done by
an abstract class that can be inherited and passed into the `BlueCast` class.
BlueCast provides two entry points to inject custom preprocessing. The
attribute `custom_preprocessor` is called right after the train_test_split.
The attribute `custom_last_mile_computation` will be called before the model
training or prediction starts (when only numerical features are present anymore)
and allows users to execute last computations (i.e. sub sampling or final calculations).

```sh
from bluecast.blueprints.cast import BlueCast
from bluecast.preprocessing.custom import CustomPreprocessing

# Create a custom tuning config and adjust hyperparameter search space
xgboost_param_config = XgboostTuneParamsConfig()
xgboost_param_config.steps_max = 100
xgboost_param_config.max_leaves_max = 16
# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.hyperparameter_tuning_rounds = 10
train_config.autotune_model = False # we want to run just normal training, no hyperparameter tuning
# We could even just overwrite the final Xgboost params using the XgboostFinalParamConfig class

class MyCustomPreprocessing(CustomPreprocessing):
    def __init__(self):
        self.trained_patterns = {}

    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        num_columns = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'], axis=1).columns
        cat_df = df[['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ']].copy()

        zscores = Zscores()
        zscores.fit_all(df, ['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'])
        df = zscores.transform_all(df, ['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'])
        self.trained_patterns["zscores"] = zscores

        imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
        num_columns = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'], axis=1).columns
        imp_mean.fit(df.loc[:, num_columns])
        df = imp_mean.transform(df.loc[:, num_columns])
        self.trained_patterns["imputation"] = imp_mean

        df = pd.DataFrame(df, columns=num_columns).merge(cat_df, left_index=True, right_index=True, how="left")

        df = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha'], axis=1)

        return df, target

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        predicton_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        num_columns = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'], axis=1).columns
        cat_df = df[['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ']].copy()

        df = self.trained_patterns["zscores"].transform_all(df, ['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'])

        imp_mean = self.trained_patterns["imputation"]
        num_columns = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha', 'EJ'], axis=1).columns
        df.loc[:, num_columns] = df.loc[:, num_columns].replace([np.inf, -np.inf], np.nan)
        df = imp_mean.transform(df.loc[:, num_columns])

        df = pd.DataFrame(df, columns=num_columns).merge(cat_df, left_index=True, right_index=True, how="left")

        df = df.drop(['Beta', 'Gamma', 'Delta', 'Alpha'], axis=1)

        return df, target

# add custom last mile computation
class MyCustomLastMilePreprocessing(CustomPreprocessing):
    def custom_function(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df / 2
        df["custom_col"] = 5
        return df

    # Please note: The base class enforces that the fit_transform method is implemented
    def fit_transform(
        self, df: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = self.custom_function(df)
        df = df.head(1000)
        target = target.head(1000)
        return df, target

    # Please note: The base class enforces that the fit_transform method is implemented
    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        predicton_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = self.custom_function(df)
        if not predicton_mode and isinstance(target, pd.Series):
            df = df.head(100)
            target = target.head(100)
        return df, targe

custom_last_mile_computation = MyCustomLastMilePreprocessing()
custom_preprocessor = MyCustomPreprocessing()

# Pass the custom configs to the BlueCast class
automl = BlueCast(
        class_problem="binary",
        conf_training=train_config,
        conf_xgboost=xgboost_param_config,
        custom_preprocessor=custom_preprocessor, # this takes place right after test_train_split
        custom_last_mile_computation=custom_last_mile_computation, # last step before model training/prediction
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

## Custom feature selection

BlueCast offers automated feature selection. On default the feature
selection is disabled, but BlueCast raises a warning to inform the
user about this option. The behaviour can be controlled via the
`TrainingConfig`.

```sh
from bluecast.blueprints.cast import BlueCast
from bluecast.preprocessing.custom import CustomPreprocessing
from bluecast.config.training_config import TrainingConfig

# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.hyperparameter_tuning_rounds = 10
train_config.autotune_model = False # we want to run just normal training, no hyperparameter tuning
train_config.enable_feature_selection = True

# Pass the custom configs to the BlueCast class
automl = BlueCast(
        class_problem="binary",
        conf_training=train_config,
    )

automl.fit(df_train, target_col="target")
y_probs, y_classes = automl.predict(df_val)
```

Also this step can be customized. The following example shows how to:

```sh
from bluecast.config.training_config import TrainingConfig
from bluecast.preprocessing.custom import CustomPreprocessing
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from typing import Optional, Tuple


# Create a custom training config and adjust general training parameters
train_config = TrainingConfig()
train_config.enable_feature_selection = True

# add custom feature selection
class RFECVSelector(CustomPreprocessing):
    def __init__(self, random_state: int = 0):
        super().__init__()
        self.selected_features = None
        self.random_state = random_state
        self.selection_strategy: RFECV = RFECV(
            estimator=xgb.XGBClassifier(),
            step=1,
            cv=StratifiedKFold(5, random_state=random_state, shuffle=True),
            min_features_to_select=1,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=2,
        )

    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        self.selection_strategy.fit(df, target)
        self.selected_features = self.selection_strategy.support_
        df = df.loc[:, self.selected_features]
        return df, target

    def transform(self,
                  df: pd.DataFrame,
                  target: Optional[pd.Series] = None,
                  predicton_mode: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = df.loc[:, self.selected_features]
        return df, target

custom_feature_selector = RFECVSelector()

# Create an instance of the BlueCast class with the custom model
bluecast = BlueCast(
    class_problem="binary",
    conf_feature_selection=custom_feat_sel,
    conf_training=train_config,
    custom_feature_selector=custom_feature_selector,

# Create some sample data for testing
x_train = pd.DataFrame(
    {"feature1": [i for i in range(10)], "feature2": [i for i in range(10)]}
)
y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
x_test = pd.DataFrame(
    {"feature1": [i for i in range(10)], "feature2": [i for i in range(10)]}

x_train["target"] = y_trai
# Fit the BlueCast model using the custom model
bluecast.fit(x_train, "target"
# Predict on the test data using the custom model
predicted_probas, predicted_classes = bluecast.predict(x_test)
```

## Custom ML model

For some users it might just be convenient to use the BlueCast class to
enjoy convenience features (details see below), but use a custom ML model.
This is possible by passing a custom model to the BlueCast class. The needed properties
are defined via the BaseClassMlModel class. Here is an example:

```sh
from bluecast.ml_modelling.base_classes import (
    BaseClassMlModel,
    PredictedClasses,  # just for linting checks
    PredictedProbas,  # just for linting checks
)

class CustomModel(BaseClassMlModel):
    def __init__(self):
        self.model = None

    def autotune(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):

        eval_dataset = Pool(x_test, y_test, cat_features=["DMAR", "LD_INDL", "RF_CESAR", "SEX"])

        alpha = 0.05
        quantile_levels = [alpha, 1 - alpha]
        quantile_str = str(quantile_levels).replace('[','').replace(']','')

        def objective(trial):
            # this part is taken from: https://www.kaggle.com/code/syerramilli/catboost-multi-quantile-regression
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 2000, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                'depth': trial.suggest_int('depth', 2, 10, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 1e6, log=True),
                'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.1, 1),
                'subsample': trial.suggest_float("subsample", 0.3, 1)
            }
            model = CatBoostRegressor(
                loss_function=f'MultiQuantile:alpha={quantile_str}',
                thread_count= 4,
                cat_features=["DMAR", "LD_INDL", "RF_CESAR", "SEX"],
                bootstrap_type =  "Bernoulli",
                sampling_frequency= 'PerTree',
                **param
            )

            # train model
            model.fit(x_train, y_train, verbose=0)

            # get predictions
            preds = model.predict(x_test)

            # get perfomance metrics
            MWIS, coverage = MWIS_metric.score(
                y_test, preds[:, 0], preds[:, 1], alpha=0.1
            )

            if coverage < 0.9:
                raise optuna.exceptions.TrialPruned()

            return MWIS

        sampler = optuna.samplers.TPESampler(
                multivariate=True, seed=1000
            )
        study = optuna.create_study(
            direction="minimize", sampler=sampler, study_name=f"catboost"
        )
        study.optimize(
            objective,
            n_trials=500,
            timeout=60 * 60 * 2,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        best_parameters = study.best_trial.params
        self.model = CatBoostRegressor(
                loss_function=f'MultiQuantile:alpha={quantile_str}',
                thread_count= 4,
                cat_features=["DMAR", "LD_INDL", "RF_CESAR", "SEX"],
                bootstrap_type =  "Bernoulli",
                sampling_frequency= 'PerTree',
                **best_parameters
            ).fit(
            x_train,
            y_train,
            eval_set=eval_dataset,
            use_best_model=True,
            early_stopping_rounds=20,
            plot=True,
        )


    def fit(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        self.autotune(x_train, x_test, y_train, y_test)

    def predict(self, df: pd.DataFrame) -> Tuple[PredictedProbas, PredictedClasses]:
        # predict Catboost classifier
        preds = self.model.predict(df)

        return preds


train_config = TrainingConfig()
train_config.global_random_state = i
train_config.calculate_shap_values = False
train_config.train_size = 0.8
train_config.enable_feature_selection = False
train_config.cat_encoding_via_ml_algorithm = True # use catboosts own cat encoding

catboost_model = CustomModel()

automl = BlueCastRegression(  # BlueCastCVRegression is not possible here, because the quantile regression predictions have an incompatible shape
        class_problem="regression",
        conf_training=train_config,
        ml_model=catboost_model,
        )
automl.fit(train.copy(), target_col=target) # fit_eval is not possible here, because the predictions have an incompatible shape
ml_models.append(automl)
```

Please note that custom ML models require user defined hyperparameter tuning. Pre-defined
configurations are not available for custom models.
Also note that the calculation of SHAP values only works with tree based models by
default. For other model architectures disable SHAP values in the TrainingConfig
via:

`train_config.calculate_shap_values = False`

Just instantiate a new instance of the TrainingConfig, update the param as above
and pass the config as an argument to the BlueCast instance during instantiation.
Feature importance can be added in the custom model definition.
