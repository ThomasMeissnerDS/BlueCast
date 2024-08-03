from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.config.training_config import TrainingConfig, XgboostTuneParamsConfig, XgboostTuneParamsRegressionConfig
from bluecast.experimentation.tracking import ExperimentTracker
from bluecast.general_utils.general_utils import save_to_production, load_for_production
from bluecast.preprocessing.feature_creation import AddRowLevelAggFeatures
from bluecast.preprocessing.feature_types import FeatureTypeDetector
from bluecast.evaluation.eval_metrics import ClassificationEvalWrapper
from bluecast.monitoring.data_monitoring import DataDrift
from bluecast.preprocessing.feature_types import FeatureTypeDetector

import polars as pl

import numpy as np, pandas as pd
import os
import json
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

import gc
import re
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = 'svg'


def competition_pipeline():
    print("Starting competition pipeline.")

    target = "class"
    class_problem = "binary"
    debug = False

    folder = "/home/thomas/Schreibtisch/Data Science/Preprocessing lib test/automl_competition"
    train = pd.read_csv(f"{folder}/train.csv")
    #train_original = pd.read_csv("original.csv")
    test = pd.read_csv(f"{folder}/test.csv")
    submission = pd.read_csv(f"{folder}/sample_submission.csv")

    train = train.drop("id", axis=1)
    test = test.drop("id", axis=1)

    # Model training

    train_config = TrainingConfig()
    if debug:
        train_config.autotune_model = False
        train_config.calculate_shap_values = False
    else:
        train_config.hypertuning_cv_folds = 10
        train_config.hyperparameter_tuning_rounds = 25
        train_config.hyperparameter_tuning_max_runtime_secs = 60 * 60 * 5
        train_config.enable_grid_search_fine_tuning = False
        train_config.calculate_shap_values = False
        train_config.show_detailed_tuning_logs = True
        train_config.train_size = 0.9
        # train_config.sample_data_during_tuning_alpha = True
        train_config.bluecast_cv_train_n_model = (5, 2)
        train_config.infrequent_categories_threshold = 10
        # train_config.cat_encoding_via_ml_algorithm = True
        train_config.out_of_fold_dataset_store_path = "/home/thomas/Schreibtisch/Data Science/Preprocessing lib test/automl_competition/"


    automl = BlueCastCV(
        class_problem=class_problem,
        conf_training=train_config,
        # single_fold_eval_metric_func=cew
    )

    automl.conf_xgboost.max_bin_max = 2500

    automl.fit_eval(train, target_col=target)

    y_probs, y_classes = automl.predict(test)

    reverse_mapping = {
        value: key for key, value in automl.bluecast_models[0].target_label_encoder.target_label_mapping.items()
    }

    submission[target] = y_classes.astype(int)

    submission = submission.replace(reverse_mapping).copy().to_csv('automl_grandprix_bluecast_xgboost_10fold_submission.csv', index=False)
    print(submission)

if __name__ == "__main__":
    competition_pipeline()
