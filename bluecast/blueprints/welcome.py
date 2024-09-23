from typing import Union

import ipywidgets as widgets
from IPython.display import clear_output, display

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression

automl_instance = None
output = widgets.Output()


def welcome_to_bluecast() -> None:

    def on_submit_clicked(b):
        global automl_instance
        with output:
            clear_output(wait=True)
            print("Processing your inputs...")

            # Process inputs
            task_type = task.value
            debug_mode_value = debug_mode.value
            n_models_value = n_models.value
            shap_values_value = shap_values.value
            n_folds_value = n_folds.value
            oof_storage_path_value = oof_storage_path.value

            # Create the BlueCast instance
            automl_instance = instantiate_bluecast_instance(task_type, n_models_value)

            # Update instance properties
            print("Configure BlueCast instance")
            update_calc_shap_flag(automl_instance, shap_values_value)
            update_debug_flag(automl_instance, debug_mode_value)
            update_hyperparam_folds(automl_instance, n_folds_value)
            update_oof_storage_path(automl_instance, oof_storage_path_value)
            print("Finished configuration of BlueCast instance")

    def instantiate_bluecast_instance(
        task, n_models: int
    ) -> Union[BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression]:
        if task in ["binary", "multiclass"] and n_models == 1:
            return BlueCast(class_problem=task)
        elif task in ["binary", "multiclass"] and n_models > 1:
            return BlueCastCV(class_problem=task)
        elif task == "regression" and n_models == 1:
            return BlueCastRegression(class_problem=task)
        elif task == "regression" and n_models > 1:
            return BlueCastCVRegression(class_problem=task)
        else:
            raise ValueError(
                "No suitable configuration found. Please raise a GitHub issue."
            )

    def update_calc_shap_flag(automl_instance, calc_shap):
        automl_instance.conf_training.calculate_shap_values = calc_shap

    def update_debug_flag(automl_instance, debug):
        automl_instance.conf_training.autotune_model = debug

    def update_hyperparam_folds(automl_instance, n_folds):
        automl_instance.conf_training.hypertuning_cv_folds = n_folds

    def update_oof_storage_path(automl_instance, oof_storage_path):
        automl_instance.conf_training.out_of_fold_dataset_store_path = oof_storage_path

    # Create widgets with tooltips using tooltip
    task = widgets.Dropdown(
        options=["binary", "multiclass", "regression"],
        description="Task Type:",
        tooltip="Select the type of problem you're solving: binary classification (i.e. True/False), multiclass classification (i.e. 'safe', 'risky', 'dangerous'), or regression (continous targets).",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )

    debug_mode = widgets.ToggleButtons(
        options=[("Yes", True), ("No", False)],
        description="Hyperparameter Tuning:",
        tooltip="Enable tuning for model optimization or choose default parameters for faster results. Default parameters can still be overwritten.",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )

    n_models = widgets.IntSlider(
        value=1,
        min=1,
        max=10,
        description="Number of Models:",
        tooltip="Specify how many models to train. More models give better performance but reduce explainability.",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )

    shap_values = widgets.ToggleButtons(
        options=[("Yes", True), ("No", False)],
        description="Calculate SHAP Values:",
        tooltip="Choose whether to calculate SHAP values for feature importance analysis. This will increase total runtime, but add explainability.",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )

    n_folds = widgets.IntSlider(
        value=5,
        min=1,
        max=10,
        description="Cross-validation Folds:",
        tooltip="Specify the number of folds for cross-validation. 1 is a simple train-test split. Otherwise 5 or 10 are usual choices. 1 fold will train faster, but might perform worse on unseen data.",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )

    oof_storage_path = widgets.Text(
        value="None",
        description="OOF Data Storage Path:",
        tooltip="Specify a path to store out-of-fold data for analysis. If the path is invalid, the pipeline will break. However it will allow error analysis after training. Only applibcable with 'fit_eval' method.",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="500px"),
    )

    submit_button = widgets.Button(
        description="Submit",
        button_style="success",
        layout=widgets.Layout(width="200px"),
    )

    # Link the submit button to the on_submit_clicked function
    submit_button.on_click(on_submit_clicked)

    # Display the widgets
    display(
        task,
        debug_mode,
        n_models,
        shap_values,
        n_folds,
        oof_storage_path,
        submit_button,
        output,
    )
