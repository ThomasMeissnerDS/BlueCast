from typing import Union

import ipywidgets as widgets
from IPython.display import clear_output, display

from bluecast.blueprints.cast import BlueCast
from bluecast.blueprints.cast_cv import BlueCastCV
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.blueprints.cast_regression import BlueCastRegression
from bluecast.config.training_config import TrainingConfig


class WelcomeToBlueCast:
    def __init__(self):
        self.automl_instance = None
        self.output = widgets.Output()

    def automl_configurator(self) -> None:
        message = """
        Welcome to BlueCast!\n
        This configurator will help find the right configuration for each user in a non-programmatic way.
        Make sure to store the instantiated WelcomeToBlueCast instance into a variable
        to be able to retrieve the pre-configured automl instance after pressing submit.

        # Sample usage:
        welcome = WelcomeToBlueCast()
        welcome.automl_configurator()
        automl = welcome.automl_instance
        automl.fit(df_train, target_col="target")
        y_hat = automl.predict(df_val)\n
        """
        print(message)

        # Create widgets with tooltips
        task = self._create_task_widget()
        debug_mode = self._create_debug_mode_widget()
        n_models = self._create_n_models_widget()
        shap_values = self._create_shap_values_widget()
        n_folds = self._create_n_folds_widget()
        oof_storage_path = self._create_oof_storage_path_widget()

        # New widgets for hyperparameter tuning options
        hyperparameter_tuning_rounds = (
            self._create_hyperparameter_tuning_rounds_widget()
        )
        hyperparameter_tuning_max_runtime_secs = (
            self._create_hyperparameter_tuning_max_runtime_secs_widget()
        )
        plot_hyperparameter_tuning_overview = (
            self._create_plot_hyperparameter_tuning_overview_widget()
        )
        show_detailed_tuning_logs = self._create_show_detailed_tuning_logs_widget()

        submit_button = self._create_submit_button()

        # Link the submit button to the on_submit_clicked function
        submit_button.on_click(
            lambda b: self.on_submit_clicked(
                task,
                debug_mode,
                n_models,
                shap_values,
                n_folds,
                oof_storage_path,
                hyperparameter_tuning_rounds,
                hyperparameter_tuning_max_runtime_secs,
                plot_hyperparameter_tuning_overview,
                show_detailed_tuning_logs,
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
            hyperparameter_tuning_rounds,
            hyperparameter_tuning_max_runtime_secs,
            plot_hyperparameter_tuning_overview,
            show_detailed_tuning_logs,
            submit_button,
            self.output,
        )

    # Widget creation functions
    def _create_task_widget(self):
        return widgets.Dropdown(
            options=["binary", "multiclass", "regression"],
            description="Task Type:",
            tooltip="Select the type of problem you're solving: binary classification, multiclass classification, or regression.",
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

    def _create_hyperparameter_tuning_rounds_widget(self):
        return widgets.IntSlider(
            value=200,
            min=1,
            max=500,
            description="Tuning Rounds:",
            tooltip="Specify the number of hyperparameter tuning rounds. The more rounds the longer the tuning process. The tuning wll stop earlier when 'Max Tuning Runtime (secs)' has been reached.",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

    def _create_hyperparameter_tuning_max_runtime_secs_widget(self):
        return widgets.IntSlider(
            value=3600,
            min=60,
            max=14400,
            description="Max Tuning Runtime (secs):",
            tooltip="Specify the maximum runtime in seconds for hyperparameter tuning. (i.e. 3600 is one hour)",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

    def _create_plot_hyperparameter_tuning_overview_widget(self):
        return widgets.ToggleButtons(
            options=[("Yes", True), ("No", False)],
            description="Plot Tuning Overview:",
            tooltip="Specify whether to plot the hyperparameter tuning overview. This will create charts showing which hyperparameters were most important and the evolution of losses during the tuning..",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

    def _create_show_detailed_tuning_logs_widget(self):
        return widgets.ToggleButtons(
            options=[("Yes", True), ("No", False)],
            description="Show Detailed Tuning Logs:",
            tooltip="Specify whether to show detailed tuning logs during hyperparameter tuning. This will print every single tested hyperparameter set and the evaluation result.",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="500px"),
        )

    def _create_submit_button(self):
        return widgets.Button(
            description="Submit",
            button_style="success",
            layout=widgets.Layout(width="200px"),
        )

    # Function handling form submission
    def on_submit_clicked(
        self,
        task,
        debug_mode,
        n_models,
        shap_values,
        n_folds,
        oof_storage_path,
        hyperparameter_tuning_rounds,
        hyperparameter_tuning_max_runtime_secs,
        plot_hyperparameter_tuning_overview,
        show_detailed_tuning_logs,
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
            hyperparameter_tuning_rounds_value = hyperparameter_tuning_rounds.value
            hyperparameter_tuning_max_runtime_secs_value = (
                hyperparameter_tuning_max_runtime_secs.value
            )
            plot_tuning_overview_value = plot_hyperparameter_tuning_overview.value
            show_detailed_logs_value = show_detailed_tuning_logs.value

            # Create the BlueCast instance
            self.automl_instance = self.instantiate_bluecast_instance(
                task_type,
                n_models_value,
                n_folds_value,
                oof_storage_path_value,
                debug_mode_value,
                hyperparameter_tuning_rounds_value,
                hyperparameter_tuning_max_runtime_secs_value,
                plot_tuning_overview_value,
                show_detailed_logs_value,
            )

            # Update instance properties
            print("Configure BlueCast instance")
            self.update_calc_shap_flag(shap_values_value)
            self.update_debug_flag(debug_mode_value)
            self.update_hyperparam_folds(n_folds_value)
            self.update_oof_storage_path(oof_storage_path_value)
            self.update_hyperparameter_tuning_rounds(hyperparameter_tuning_rounds_value)
            self.update_hyperparameter_tuning_max_runtime_secs(
                hyperparameter_tuning_max_runtime_secs_value
            )
            self.update_plot_hyperparameter_tuning_overview(plot_tuning_overview_value)
            self.update_show_detailed_tuning_logs(show_detailed_logs_value)
            print("Finished configuration of BlueCast instance")

    # New update methods for hyperparameter tuning parameters
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

    def update_hyperparameter_tuning_rounds(self, rounds):
        self.automl_instance.conf_training.hyperparameter_tuning_rounds = rounds

    def update_hyperparameter_tuning_max_runtime_secs(self, runtime_secs):
        self.automl_instance.conf_training.hyperparameter_tuning_max_runtime_secs = (
            runtime_secs
        )

    def update_plot_hyperparameter_tuning_overview(self, plot_tuning_overview):
        self.automl_instance.conf_training.plot_hyperparameter_tuning_overview = (
            plot_tuning_overview
        )

    def update_show_detailed_tuning_logs(self, show_logs):
        self.automl_instance.conf_training.show_detailed_tuning_logs = show_logs

    # Function to instantiate BlueCast based on the task type
    def instantiate_bluecast_instance(
        self,
        task,
        n_models: int,
        n_folds: int,
        oof_storage_path: str,
        autotune_model: bool,
        hyperparameter_tuning_rounds: int,
        hyperparameter_tuning_max_runtime_secs: int,
        plot_hyperparameter_tuning_overview: bool,
        show_detailed_tuning_logs: bool,
    ) -> Union[BlueCast, BlueCastCV, BlueCastRegression, BlueCastCVRegression]:
        # Prepare the configuration for BlueCast
        conf_training = TrainingConfig(
            autotune_model=autotune_model,
            hypertuning_cv_folds=n_folds,
            out_of_fold_dataset_store_path=oof_storage_path,
            hyperparameter_tuning_rounds=hyperparameter_tuning_rounds,
            hyperparameter_tuning_max_runtime_secs=hyperparameter_tuning_max_runtime_secs,
            plot_hyperparameter_tuning_overview=plot_hyperparameter_tuning_overview,
            show_detailed_tuning_logs=show_detailed_tuning_logs,
        )

        if task in ["binary", "multiclass"]:
            if n_models == 1:
                automl_instance_bc = BlueCast(
                    class_problem=task, conf_training=conf_training
                )
                print(
                    f"Instantiated BlueCast instance with:\n automl_instance = BlueCast(class_problem={task})\n"
                )
                return automl_instance_bc
            else:
                automl_instance_bcc = BlueCastCV(
                    class_problem=task, conf_training=conf_training
                )
                print(
                    f"Instantiated BlueCast instance with:\n automl_instance = BlueCastCV(class_problem={task})\n"
                )
                return automl_instance_bcc
        elif task == "regression":
            if n_models == 1:
                automl_instance_bcr = BlueCastRegression(
                    class_problem=task, conf_training=conf_training
                )
                print(
                    f"Instantiated BlueCast instance with:\n automl_instance = BlueCastRegression(class_problem={task})\n"
                )
                return automl_instance_bcr
            else:
                automl_instance_bcrc = BlueCastCVRegression(
                    class_problem=task, conf_training=conf_training
                )
                print(
                    f"Instantiated BlueCast instance with:\n automl_instance = BlueCastCVRegression(class_problem={task})\n"
                )
                return automl_instance_bcrc
        else:
            raise ValueError(
                "No suitable configuration found. Please raise a GitHub issue."
            )
