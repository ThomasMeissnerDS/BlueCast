from unittest import mock

import numpy as np

from bluecast.blueprints.cast import BlueCast
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig
from bluecast.evaluation.eval_metrics import matthews_corrcoef
from bluecast.evaluation.shap_values import shap_explanations
from bluecast.tests.make_data.create_data import create_synthetic_dataframe


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


def test_shap_explanations_else(mock_model, test_data):
    explainer = "other"
    with mock.patch(
        "your_module.shap.KernelExplainer"
    ) as mock_kernel_explainer, mock.patch(
        "your_module.shap.summary_plot"
    ) as mock_summary_plot, mock.patch(
        "matplotlib.pyplot.show"
    ) as mock_show:
        mock_kernel_explainer.return_value.shap_values.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )

        shap_values = shap_explanations(mock_model, test_data, explainer)

        mock_kernel_explainer.assert_called_once_with(mock_model.predict, test_data)
        mock_summary_plot.assert_called_once_with(
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), test_data, show=True
        )
        mock_show.assert_called_once()

        assert shap_values is not None


def test_eval_classifier_except():
    y_true = np.array([0, 0, 0, 0])
    y_classes = np.array([1, 1, 1, 1])
    result = matthews_corrcoef(y_true, y_classes)
    assert result == 0
