from typing import Union

import ipywidgets as widgets
from IPython.display import clear_output, display

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression


class WelcomeToBlueCast:
    def __init__(self):
        self.automl_instance = None
        self.output = widgets.Output()

    def automl_configurator(self) -> None:
        # Create widgets with tooltips
        task = self._create_task_widget()
        debug_mode = self._create_debug_mode_widget()
        n_models = self._create_n_models_widget()
        shap_values = self._create_shap_values_widget()
        n_folds = self._create_n_folds_widget()
        oof_storage_path = self._create_oof_storage_path_widget()
        submit_button = self._create_submit_button()

        # Link the submit button to the on_submit_clicked function
        submit_button.on_click(
            lambda b: self.on_submit_clicked(
                task, debug_mode, n_models, shap_values, n_folds, oof_storage_path
            )
        )

        # Display the widgets
        display(
            task,
            debug_mode,
            n_models,
            shap_values,
            n_folds,
            oof_storage_path,
            submit_button,
            self.output,
        )

    def _create_task_widget(self):
        return widgets.Dropdown(
            options=["binary", "multiclass", "regression"],
            description="Task Type:",
            tooltip="Select the type of problem you're solving: binary classification (i.e. True/False), multiclass classification (i.e. 'safe', 'risky', 'dangerous'), or regression (continous targets).",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

    def _create_debug_mode_widget(self):
        return widgets.ToggleButtons(
            options=[("Yes", True), ("No", False)],
            description="Hyperparameter Tuning:",
            tooltip="Enable tuning for model optimization or choose default parameters for faster results. Default parameters can still be overwritten. Set this to False for fast debugging tests.",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

    def _create_n_models_widget(self):
        return widgets.IntSlider(
            value=1,
            min=1,
            max=10,
            description="Number of Models:",
            tooltip="Specify how many models to train. More models will improve performance, but make explainability more difficult. Also runtime increases linearly.",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

    def _create_shap_values_widget(self):
        return widgets.ToggleButtons(
            options=[("Yes", True), ("No", False)],
            description="Calculate SHAP Values:",
            tooltip="Choose whether to calculate SHAP values for feature importance analysis. This will increase total runtime, but add explainability.",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

    def _create_n_folds_widget(self):
        return widgets.IntSlider(
            value=5,
            min=1,
            max=10,
            description="Cross-validation Folds:",
            tooltip="Specify the number of folds for cross-validation. 1 is a simple train-test split. Otherwise 5 or 10 are usual choices. 1 fold will train faster, but might perform worse on unseen data.",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

    def _create_oof_storage_path_widget(self):
        return widgets.Text(
            value="None",
            description="OOF Data Storage Path:",
            tooltip="Specify a path to store out-of-fold data for analysis.",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

    def _create_submit_button(self):
        return widgets.Button(
            description="Submit",
            button_style="success",
            layout=widgets.Layout(width="200px"),
        )

    def on_submit_clicked(
        self, task, debug_mode, n_models, shap_values, n_folds, oof_storage_path
    ):
        with self.output:
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
            self.automl_instance = self.instantiate_bluecast_instance(
                task_type, n_models_value
            )

            # Update instance properties
            print("Configure BlueCast instance")
            self.update_calc_shap_flag(shap_values_value)
            self.update_debug_flag(debug_mode_value)
            self.update_hyperparam_folds(n_folds_value)
            self.update_oof_storage_path(oof_storage_path_value)
            print("Finished configuration of BlueCast instance")

    def instantiate_bluecast_instance(
        self, task, n_models: int
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

    def update_calc_shap_flag(self, calc_shap):
        self.automl_instance.conf_training.calculate_shap_values = calc_shap

    def update_debug_flag(self, debug):
        self.automl_instance.conf_training.autotune_model = debug

    def update_hyperparam_folds(self, n_folds):
        self.automl_instance.conf_training.hypertuning_cv_folds = n_folds

    def update_oof_storage_path(self, oof_storage_path):
        self.automl_instance.conf_training.out_of_fold_dataset_store_path = (
            oof_storage_path
        )
