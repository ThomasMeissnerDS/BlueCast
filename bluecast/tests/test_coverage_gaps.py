"""Targeted tests to cover previously uncovered code paths."""

import pandas as pd

from bluecast.config.training_config import (
    CatboostFinalParamConfig,
    CatboostRegressionFinalParamConfig,
    CatboostTuneParamsConfig,
    CatboostTuneParamsRegressionConfig,
    TrainingConfig,
    XgboostFinalParamConfig,
    XgboostRegressionFinalParamConfig,
    XgboostTuneParamsConfig,
    XgboostTuneParamsRegressionConfig,
)
from bluecast.preprocessing.train_test_split import train_test_split_time


class TestConfigRepr:
    def test_training_config_repr(self):
        tc = TrainingConfig()
        r = repr(tc)
        assert "TrainingConfig(" in r
        assert "global_random_state=33" in r

    def test_xgboost_tune_params_repr(self):
        cfg = XgboostTuneParamsConfig()
        r = repr(cfg)
        assert "XgboostTuneParamsConfig(" in r
        assert "max_depth_min" in r

    def test_xgboost_tune_params_regression_repr(self):
        cfg = XgboostTuneParamsRegressionConfig()
        r = repr(cfg)
        assert "XgboostTuneParamsRegressionConfig(" in r

    def test_xgboost_final_param_repr(self):
        cfg = XgboostFinalParamConfig()
        r = repr(cfg)
        assert "XgboostFinalParamConfig(" in r
        assert "params=" in r

    def test_xgboost_regression_final_param_repr(self):
        cfg = XgboostRegressionFinalParamConfig()
        r = repr(cfg)
        assert "XgboostRegressionFinalParamConfig(" in r

    def test_catboost_tune_params_repr(self):
        cfg = CatboostTuneParamsConfig()
        r = repr(cfg)
        assert "CatboostTuneParamsConfig(" in r

    def test_catboost_tune_params_regression_repr(self):
        cfg = CatboostTuneParamsRegressionConfig()
        r = repr(cfg)
        assert "CatboostTuneParamsRegressionConfig(" in r

    def test_catboost_final_param_repr(self):
        cfg = CatboostFinalParamConfig()
        r = repr(cfg)
        assert "CatboostFinalParamConfig(" in r
        assert "params=" in r

    def test_catboost_regression_final_param_repr(self):
        cfg = CatboostRegressionFinalParamConfig()
        r = repr(cfg)
        assert "CatboostRegressionFinalParamConfig(" in r


class TestShapWaterfallIndicesElseBranch:
    def test_shap_waterfall_indices_with_value(self):
        tc = TrainingConfig(shap_waterfall_indices=[0, 5, 99])
        assert tc.shap_waterfall_indices == [0, 5, 99]

    def test_shap_waterfall_indices_default(self):
        tc = TrainingConfig()
        assert tc.shap_waterfall_indices == []


class TestTrainTestSplitTimeFString:
    def test_time_split_with_column(self):
        df = pd.DataFrame(
            {"a": range(50), "b": range(50), "order": range(50), "target": range(50)}
        )
        x_train, x_test, y_train, y_test = train_test_split_time(
            df, "target", "order", 0.8
        )
        assert len(x_train) == 40
        assert len(x_test) == 10

    def test_time_split_without_column(self):
        df = pd.DataFrame({"a": range(50), "b": range(50), "target": range(50)})
        x_train, x_test, y_train, y_test = train_test_split_time(df, "target", "", 0.8)
        assert len(x_train) == 40
        assert len(x_test) == 10
