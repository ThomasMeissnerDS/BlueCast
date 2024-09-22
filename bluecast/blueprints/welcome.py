from typing import Union

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression


def print_answer_separation():
    print("-----------\n")


def print_bluecast_dragon_ascii():
    print(
        r"""
                           __====-_  _-====__
                _--^^^#####//      \\#####^^^--_
             _-^##########// (    ) \\##########^-_
            -############//  |\^^/|  \\############-
          _/############//   (@::@)   \\############\_
         /#############((     \\//     ))#############\
        -###############\\    (oo)    //###############-
       -#################\\  / "  \  //#################-
      -###################\\/  (..)  \//###################-
     _#/|##########/\######(   "   )######/\##########|\#_
     |/ |#/\#/\#/\/  \#/\##\  (   )  /\#/  \/\#/\#/\#| \|
     `  |/  V  | ||  | \//\\\ (____) ///\\//  | |  V   \|
        `   `  `-' `-'  `-'  `----'  `-'  `-'  `-'   `

    BBBBB   L        U     U  EEEEE   CCCCC    AAAAA   SSSSS  TTTTTT
    B    B  L        U     U  E       C        A   A   S        TT
    BBBBB   L        U     U  EEEEE   C        AAAAA   SSSSS    TT
    B    B  L        U     U  E       C        A   A       S    TT
    BBBBB   LLLLLL    UUUUU   EEEEE   CCCCC    A   A   SSSSS    TT

    """
    )


def ask_for_name():
    print_bluecast_dragon_ascii()

    name = input(
        """
        \n
        Welcome to BlueCast.\n

        What is your name?\n
        """
    )
    return name


def questionnaire_intro(name):
    print(
        f"""
        Hello {name}!\n\n
        The following questionnaire will\n
        help finding the most suitable configuration for you.\n
        The output of this function will return a preconfigured\n
        BlueCast instance. So please make sure to store the\n
        output of this function into a variable.\n
        I.e.: automl = welcome_to_bluecast()\n"""
    )


def ask_for_task():
    print_answer_separation()
    task = input(
        """
        🚀 What type of problem are you trying to solve today?\n
        BlueCast can predict:\n
          - binary labels (i.e. 0 or 1, "churn" or "no churn")\n
          - multiclass labels (i.e. [0, 1, 2, 3] or ["A", "B", "C", "D"])\n
          - regression of continous targets (i.e. height, salary, lifetime value like [1.2, 3.5, 8.])\n

        Please enter binary, multiclass or regression:
        """
    )
    return task


def task_questionnaire_loop():
    task = None
    while not task:
        answer = str(ask_for_task()).lower()
        if "binary" in answer:
            task = "binary"
        elif "multiclass" in answer:
            task = "multiclass"
        elif "regression" in answer:
            task = "regression"
        else:
            please_reply_with_an_correct_task()
            continue
    return task


def ask_for_debug_mode():
    print_answer_separation()
    debug = input(
        """
        Do you want to tune hyperparameters or start with a fast instance without\n
        hyperparameter tuning to just test your code end-to-end?\n
        Type "y" for hyperparameter tuning or "n" for using default hyperparameters.\n
        """
    )
    return debug


def debug_questionnaire_loop():
    debug = None
    while not debug:
        answer = str(ask_for_debug_mode()).lower()
        if "y" in answer:
            debug = "y"
        elif "n" in answer:
            debug = "n"
        else:
            please_reply_with_y_or_n()
            continue
    return debug


def ask_for_multimodel_mode():
    print_answer_separation()
    n_models = input(
        """
        Do you prefer explainability or raw performance?\n
        For explainability we will train one model only. For\n
        raw performance we can train multiple models. However \n
        explainability will be provided for every single model,\n
        not the ensemble altogether. Please type in the number of\n
         models we shall train:\n
        """
    )
    return n_models


def multimodel_questionnaire_loop():
    n_models = None
    while not n_models:
        try:
            answer = int(ask_for_multimodel_mode())
        except Exception:
            answer = None
        if isinstance(answer, int):
            n_models = answer
        else:
            please_reply_with_an_int()
            continue
    return n_models


def ask_for_shap_values():
    print_answer_separation()
    n_models = input(
        """
        It can be very helpful to understand feature importance.\n
        However calculating SHAP values can take a while (especially\n
        with big data and on CPU). This would also be required to show\n
        SHAP values for new predictions. Do you want BlueCast to calculate\n
        SHAP values and show a plot with global feature importance? (y/n)\n
        """
    )
    return n_models


def shap_value_questionnaire_loop():
    calc_shap = None
    while not calc_shap:
        answer = ask_for_shap_values()
        if "y" in answer:
            calc_shap = "y"
        elif "n" in answer:
            calc_shap = "n"
        else:
            please_reply_with_y_or_n()
            continue
    return calc_shap


def ask_for_cross_validation():
    print_answer_separation()
    n_folds = input(
        """
        How robust shall the hyperparameter tuning\n
        of each model shall be? With 1 fold the training\n
        will be much faster, but the model might overfit.\n
        With more folds (i.e. 5) the training time increases,\n
        but the model will perform better on unseen data.\n
        If the training dataset is big (i.e. 300k rows), 1 fold\n
        might still be good enough (depending on the business\n
        case). How many folds shall be used to find optimal\n
        hyperparameters? Type in a number without decimals between\n
        1 and n (usually this is 5 or 10 for cross validation):\n
        """
    )
    return n_folds


def cross_validation_questionnaire_loop():
    n_folds = None
    while not n_folds:
        try:
            answer = int(ask_for_cross_validation())
        except Exception:
            answer = None
        if isinstance(answer, int):
            n_folds = answer
        else:
            please_reply_with_an_int()
            continue
    return n_folds


def ask_for_oof_data_storage():
    print_answer_separation()
    output_path = input(
        """
        For automl pipeline training BlueCast instances\n
        offer two options:\n
        - 'fit' (usually used for final training)\n
        - 'fit_eval' (for prototyping)\n

        The 'fit_eval' method will train the model and\n
        evaluate against unseen data. For single model\n
        instances (BlueCast, BlueCastRegression) the user\n
        needs to provide the unseen data. BlueCast does not\n
        store the unseen data by default. However you can\n
        provide a path to store the unseen data. This can\n
        be used for error analysis afterwards.\n\n

        If you want to store out-of-fold data, please provide\n
        a valid path. Otherwise type 'None'. (If something else\n
        than 'None' is provided, which is not a valid path,\n
        the pipeline will break).\n
        """
    )
    return output_path


def oof_data_storage_questionnaire_loop():
    oof_path = None
    while not oof_path:
        answer = ask_for_oof_data_storage()

        if "None" in answer:
            oof_path = "None"
        elif "/" in answer:
            oof_path = answer
        else:
            please_reply_with_full_path_or_none()
            continue

    if oof_path == "None":
        oof_path = None
    return oof_path


# INPUT CORRECTION QUESTIONS


def please_reply_with_y_or_n():
    print("\n")
    print(
        """
    Incompatible input found.\n
    Please answer with either 'y' or 'n'\n
    """
    )


def please_reply_with_full_path_or_none():
    print("\n")
    print(
        """
    Incompatible input found.\n
    Please answer with 'None' or a full path:\n
    """
    )


def please_reply_with_an_int():
    print("\n")
    print(
        """
    Incompatible input found.\n
    Please answer with an integer (i.e.: 5):\n
    """
    )


def please_reply_with_an_correct_task():
    print("\n")
    print(
        """
    Incompatible input found.\n
    Please answer with one of the following options: "binary", "multiclass" or "regression"\n
    """
    )


# CLASS CREATION FUNCS


def instantiate_bluecast_instance(
    task, n_models: int
) -> Union[BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression]:
    if task in ["binary", "multiclass"] and n_models == 1:
        automl_instance_bc = BlueCast(class_problem=task)
        print(
            f"""
        Instantiated BlueCast instance with:\n
        automl_instance = BlueCast(class_problem={task})\n
        """
        )
        return automl_instance_bc
    elif task in ["binary", "multiclass"] and n_models > 1:
        automl_instance_bcv = BlueCastCV(class_problem=task)
        print(
            f"""
        Instantiated BlueCast instance with:\n
        automl_instance = BlueCastCV(class_problem={task})\n
        """
        )
        return automl_instance_bcv
    elif task in ["regression"] and n_models == 1:
        automl_instance_bcr = BlueCastRegression(class_problem=task)
        print(
            f"""
        Instantiated BlueCast instance with:\n
        automl_instance = BlueCastRegression(class_problem={task})\n
        """
        )
        return automl_instance_bcr
    elif task in ["regression"] and n_models > 1:
        automl_instance_bcrv = BlueCastCVRegression(class_problem=task)
        print(
            f"""
        Instantiated BlueCast instance with:\n
        automl_instance = BlueCastCVRegression(class_problem={task})\n
        """
        )
        return automl_instance_bcrv
    else:
        raise ValueError(
            "No suitable configuration found. Please raise a GitHub issue."
        )


def update_calc_shap_flag(
    automl_instance: Union[
        BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression
    ],
    calc_shap,
) -> Union[BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression]:
    automl_instance.conf_training.calculate_shap_values = calc_shap
    print(
        f"""
    Updating automl instance with:
    automl_instance.conf_training.calculate_shap_values = {calc_shap}
    """
    )
    return automl_instance


def update_debug_flag(
    automl_instance: Union[
        BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression
    ],
    debug,
) -> Union[BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression]:
    map_answer_to_bool = {"y": True, "n": False}
    automl_instance.conf_training.autotune_model = map_answer_to_bool.get(debug, True)
    print(
        f"""
    Updating automl instance with:
    automl_instance.conf_training.autotune_model = {map_answer_to_bool.get(debug)}
    """
    )
    return automl_instance


def update_hyperparam_folds(
    automl_instance: Union[
        BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression
    ],
    n_folds,
) -> Union[BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression]:
    automl_instance.conf_training.hypertuning_cv_folds = n_folds
    print(
        f"""
    Updating automl instance with:
    automl_instance.conf_training.hypertuning_cv_folds = {n_folds}
    """
    )
    return automl_instance


def update_oof_storage_path(
    automl_instance: Union[
        BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression
    ],
    oof_storage_path,
) -> Union[BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression]:
    automl_instance.conf_training.out_of_fold_dataset_store_path = oof_storage_path
    print(
        f"""
    Updating automl instance with:
    automl_instance.conf_training.out_of_fold_dataset_store_path = {oof_storage_path}
    """
    )
    return automl_instance


def welcome_to_bluecast() -> (
    Union[BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression]
):
    """
    A welcome function to help configuring BlueCast instances.

    After a series of questions the function return a pre-configured
    automl instance.
    """
    name = ask_for_name()
    print_answer_separation()

    questionnaire_intro(name)

    task = task_questionnaire_loop()
    print_answer_separation()

    n_models = multimodel_questionnaire_loop()
    print_answer_separation()

    automl_instance = instantiate_bluecast_instance(task, n_models)

    calc_shap = shap_value_questionnaire_loop()
    automl_instance = update_calc_shap_flag(automl_instance, calc_shap)
    print_answer_separation()

    debug = debug_questionnaire_loop()
    automl_instance = update_debug_flag(automl_instance, debug)
    print_answer_separation()

    n_folds = cross_validation_questionnaire_loop()
    automl_instance = update_hyperparam_folds(automl_instance, n_folds)
    print_answer_separation()

    oof_path = oof_data_storage_questionnaire_loop()
    automl_instance = update_oof_storage_path(automl_instance, oof_path)
    print_answer_separation()

    return automl_instance
