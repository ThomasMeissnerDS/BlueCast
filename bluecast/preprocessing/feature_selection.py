import logging
import operator
import warnings
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from bluecast.preprocessing.custom import CustomPreprocessing


class BoostaRootaWrapper(CustomPreprocessing):
    def __init__(
        self, class_problem: Literal["binary", "multiclass", "regression"], random_state
    ):
        super().__init__()
        self.class_problem = class_problem
        self.selected_features: List[Optional[str]] = []
        self.random_state = random_state

        if self.class_problem not in ["binary", "multiclass", "regression"]:
            raise ValueError(
                "Class_problem must be one of 'binary', 'multiclass', or 'regression'."
            )

    def fit_transform(
        self, df: pd.DataFrame, targets: pd.Series
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if self.class_problem == "binary":
            model = xgb.XGBClassifier(
                tree_method="approx",
                max_bin=255,
                n_estimators=100,
                random_state=self.random_state,
                importance_type="total_cover",
                max_depth=10,
            )
            br = BoostARoota(clf=model)
        elif self.class_problem == "multiclass":
            br = BoostARoota(metric="mlogloss")
        else:
            model = xgb.XGBRegressor(
                tree_method="approx",
                max_bin=255,
                n_estimators=100,
                random_state=self.random_state,
                importance_type="total_cover",
                max_depth=10,
            )
            br = BoostARoota(clf=model)

        logging.info("Start feature selection as defined in FeatureSelectionConfig.")
        br.fit(df, targets)
        self.selected_features = br.keep_vars_
        logging.info(
            f"{len(self.selected_features)} from {len(df.columns)} features selected."
        )
        logging.info("Selected features are {self.selected_features}).")
        return df.loc[:, self.selected_features], targets

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        predicton_mode: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        logging.info("Apply feature selection.")
        df = df.loc[:, self.selected_features]
        return df, target


class BoostARoota(object):

    def __init__(
        self,
        metric=None,
        clf=None,
        cutoff=200,
        iters=10,
        max_rounds=100,
        delta=0.1,
        silent=True,
    ):
        self.metric = metric
        self.clf = clf
        self.cutoff = cutoff
        self.iters = iters
        self.max_rounds = max_rounds
        self.delta = delta
        self.silent = silent
        self.keep_vars_ = None

        # Throw errors if the inputted parameters don't meet the necessary criteria
        if (metric is None) and (clf is None):
            raise ValueError("you must enter one of metric or clf as arguments")
        if cutoff <= 0:
            raise ValueError(
                "cutoff should be greater than 0. You entered" + str(cutoff)
            )
        if iters <= 0:
            raise ValueError("iters should be greater than 0. You entered" + str(iters))
        if (delta <= 0) | (delta > 1):
            raise ValueError("delta should be between 0 and 1, was " + str(delta))

        # Issue warnings for parameters to still let it run
        if (metric is not None) and (clf is not None):
            warnings.warn(
                "You entered values for metric and clf, defaulting to clf and ignoring metric",
                stacklevel=2,
            )
        if delta < 0.02:
            warnings.warn(
                "WARNING: Setting a delta below 0.02 may not converge on a solution.",
                stacklevel=2,
            )
        if max_rounds < 1:
            warnings.warn(
                "WARNING: Setting max_rounds below 1 will automatically be set to 1.",
                stacklevel=2,
            )

    def fit(self, x, y):
        self.keep_vars_ = _BoostARoota(
            x,
            y,
            metric=self.metric,
            clf=self.clf,
            cutoff=self.cutoff,
            iters=self.iters,
            max_rounds=self.max_rounds,
            delta=self.delta,
            silent=self.silent,
        )
        return self

    def transform(self, x):
        if self.keep_vars_ is None:
            raise ValueError("You need to fit the model first")
        return x[self.keep_vars_]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)


def _create_shadow(x_train):
    """
    Take all X variables, creating copies and randomly shuffling them
    :param x_train: the dataframe to create shadow features on
    :return: dataframe 2x width and the names of the shadows for removing later
    """
    x_shadow = x_train.copy()
    for c in x_shadow.columns:
        np.random.shuffle(x_shadow[c].values)
    # rename the shadow
    shadow_names = ["ShadowVar" + str(i + 1) for i in range(x_train.shape[1])]
    x_shadow.columns = shadow_names
    # Combine to make one new dataframe
    new_x = pd.concat([x_train, x_shadow], axis=1)
    return new_x, shadow_names


def _reduce_vars_xgb(x, y, metric, this_round, cutoff, n_iterations, delta, silent):
    """
    Function to run through each
    :param x: Input dataframe - X
    :param y: Target variable
    :param metric: Metric to optimize in XGBoost
    :param this_round: Round so it can be printed to screen
    :return: tuple - stopping criteria and the variables to keep
    """
    # Set up the parameters for running the model in XGBoost - split is on multi log loss
    if metric == "mlogloss":
        param = {
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "num_class": len(np.unique(y)),
            "silent": 1,
        }
    else:
        param = {"eval_metric": metric, "silent": 1}
    for i in range(1, n_iterations + 1):
        # Create the shadow variables and run the model to obtain importances
        new_x, shadow_names = _create_shadow(x)
        dtrain = xgb.DMatrix(new_x, label=y, enable_categorical=True)
        bst = xgb.train(param, dtrain, verbose_eval=False)
        if i == 1:
            df = pd.DataFrame({"feature": new_x.columns})
            pass

        importance = bst.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df2 = pd.DataFrame(importance, columns=["feature", "fscore" + str(i)])
        df2["fscore" + str(i)] = df2["fscore" + str(i)] / df2["fscore" + str(i)].sum()
        df = pd.merge(df, df2, on="feature", how="outer")
        if not silent:
            print("Round: ", this_round, " iteration: ", i)

    df_numeric = df.select_dtypes(include=[np.number])  # Only select numeric columns
    df["Mean"] = df_numeric.mean(axis=1)
    # Split them back out
    real_vars = df[~df["feature"].isin(shadow_names)]
    shadow_vars = df[df["feature"].isin(shadow_names)]

    # Get mean value from the shadows
    mean_shadow = shadow_vars["Mean"].mean() / cutoff
    real_vars = real_vars[(real_vars.Mean > mean_shadow)]

    # Check for the stopping criteria
    # Basically looking to make sure we are removing at least 10% of the variables, or we should stop
    if (len(real_vars["feature"]) / len(x.columns)) > (1 - delta):
        criteria = True
    else:
        criteria = False

    return criteria, real_vars["feature"]


def _reduce_vars_sklearn(x, y, clf, this_round, cutoff, n_iterations, delta, silent):
    """
    Function to run through each
    :param x: Input dataframe - X
    :param y: Target variable
    :param clf: the fully specified classifier passed in by user
    :param this_round: Round so it can be printed to screen
    :return: tuple - stopping criteria and the variables to keep
    """
    # Set up the parameters for running the model in XGBoost - split is on multi log loss

    for i in range(1, n_iterations + 1):
        # Create the shadow variables and run the model to obtain importances
        new_x, shadow_names = _create_shadow(x)
        clf = clf.fit(new_x, np.ravel(y))

        if i == 1:
            df = pd.DataFrame({"feature": new_x.columns})
            df2 = df.copy()
            pass

        try:
            importance = clf.feature_importances_
            df2["fscore" + str(i)] = importance
        except ValueError:
            print(
                "this clf doesn't have the feature_importances_ method.  Only Sklearn tree based methods allowed"
            )

        # importance = sorted(importance.items(), key=operator.itemgetter(1))

        # df2 = pd.DataFrame(importance, columns=['feature', 'fscore'+str(i)])
        df2["fscore" + str(i)] = df2["fscore" + str(i)] / df2["fscore" + str(i)].sum()
        df = pd.merge(df, df2, on="feature", how="outer", suffixes=("", "_y"))
        df.drop(df.filter(regex="_y$").columns, axis=1, inplace=True)
        if not silent:
            print("Round: ", this_round, " iteration: ", i)

    df["Mean"] = df.drop("feature", axis=1).mean(axis=1)
    # Split them back out
    real_vars = df[~df["feature"].isin(shadow_names)]
    shadow_vars = df[df["feature"].isin(shadow_names)]

    # Get mean value from the shadows
    mean_shadow = shadow_vars["Mean"].mean() / cutoff
    real_vars = real_vars[(real_vars.Mean > mean_shadow)]

    # Check for the stopping criteria
    # Basically looking to make sure we are removing at least 10% of the variables, or we should stop
    if (len(real_vars["feature"]) / len(x.columns)) > (1 - delta):
        criteria = True
    else:
        criteria = False

    return criteria, real_vars["feature"]


# Main function exposed to run the algorithm
def _BoostARoota(x, y, metric, clf, cutoff, iters, max_rounds, delta, silent):
    """
    Function loops through, waiting for the stopping criteria to change
    :param x: X dataframe One Hot Encoded
    :param y: Labels for the target variable
    :param metric: The metric to optimize in XGBoost
    :return: names of the variables to keep
    """

    new_x = x.copy()
    # Run through loop until "crit" changes
    i = 0
    while True:
        # Inside this loop we reduce the dataset on each iteration exiting with keep_vars
        i += 1
        if clf is None:
            crit, keep_vars = _reduce_vars_xgb(
                new_x,
                y,
                metric=metric,
                this_round=i,
                cutoff=cutoff,
                n_iterations=iters,
                delta=delta,
                silent=silent,
            )
        else:
            crit, keep_vars = _reduce_vars_sklearn(
                new_x,
                y,
                clf=clf,
                this_round=i,
                cutoff=cutoff,
                n_iterations=iters,
                delta=delta,
                silent=silent,
            )

        if crit | (i >= max_rounds):
            break  # exit and use keep_vars as final variables
        else:
            new_x = new_x[keep_vars].copy()
    if not silent:
        print("BoostARoota ran successfully! Algorithm went through ", i, " rounds.")
    return keep_vars
