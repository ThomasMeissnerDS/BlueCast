from blueprints.cast import BlueCast
from config.training_config import TrainingConfig, XgboostTuneParamsConfig
from tests.make_data.create_data import create_synthetic_dataframe


def test_shap_explanations():
    """Test that tests the BlueCast class"""
    df_train = create_synthetic_dataframe(200, random_state=20)
    df_val = create_synthetic_dataframe(100, random_state=21)
    xgboost_param_config = XgboostTuneParamsConfig()
    xgboost_param_config.steps_max = 100
    xgboost_param_config.num_leaves_max = 16
    train_config = TrainingConfig()
    train_config.hyperparameter_tuning_rounds = 10

    automl = BlueCast(
        class_problem="binary",
        target_column="target",
        conf_training=train_config,
        conf_xgboost=xgboost_param_config,
    )
    eval_dict = automl.fit_eval(
        df_train, df_val.drop("target", axis=1), df_val["target"], target_col="target"
    )
    print(eval_dict)
    assert isinstance(eval_dict, dict)
