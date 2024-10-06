from unittest.mock import MagicMock, patch

import ipywidgets as widgets
import pytest

from bluecast.blueprints.welcome import WelcomeToBlueCast


@pytest.fixture
def welcome():
    """Fixture to initialize WelcomeToBlueCast class."""
    return WelcomeToBlueCast()


def test_widget_creation(welcome):
    """Test if widgets are created properly."""
    assert isinstance(welcome._create_task_widget(), widgets.Dropdown)
    assert isinstance(welcome._create_debug_mode_widget(), widgets.ToggleButtons)
    assert isinstance(welcome._create_n_models_widget(), widgets.IntSlider)
    assert isinstance(welcome._create_shap_values_widget(), widgets.ToggleButtons)
    assert isinstance(welcome._create_n_folds_widget(), widgets.IntSlider)
    assert isinstance(welcome._create_oof_storage_path_widget(), widgets.Text)
    assert isinstance(
        welcome._create_hyperparameter_tuning_rounds_widget(), widgets.IntSlider
    )
    assert isinstance(
        welcome._create_hyperparameter_tuning_max_runtime_secs_widget(),
        widgets.IntSlider,
    )
    assert isinstance(
        welcome._create_plot_hyperparameter_tuning_overview_widget(),
        widgets.ToggleButtons,
    )
    assert isinstance(
        welcome._create_show_detailed_tuning_logs_widget(), widgets.ToggleButtons
    )
    assert isinstance(welcome._create_submit_button(), widgets.Button)


@patch("bluecast.blueprints.welcome.clear_output")
def test_on_submit_clicked(mock_clear_output, welcome):
    """Test the submission handling logic."""
    task = MagicMock(value="binary")
    debug_mode = MagicMock(value=True)
    n_models = MagicMock(value=2)
    shap_values = MagicMock(value=True)
    n_folds = MagicMock(value=5)
    oof_storage_path = MagicMock(value="/path/to/oof")
    hyperparameter_tuning_rounds = MagicMock(value=100)
    hyperparameter_tuning_max_runtime_secs = MagicMock(value=3600)
    plot_hyperparameter_tuning_overview = MagicMock(value=True)
    show_detailed_tuning_logs = MagicMock(value=True)

    # Mock the instantiation of BlueCast to avoid creating real objects
    with patch.object(
        welcome, "instantiate_bluecast_instance", return_value=MagicMock()
    ) as mock_instantiate:
        welcome.on_submit_clicked(
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

        # Ensure clear_output is called
        mock_clear_output.assert_called_once()

        # Ensure instantiate_bluecast_instance is called with correct parameters
        mock_instantiate.assert_called_once_with(
            "binary", 2, 5, "/path/to/oof", True, 100, 3600, True, True
        )


def test_bluecast_instance_instantiation(welcome):
    """Test if BlueCast instances are correctly instantiated based on task type and number of models."""
    with patch("bluecast.blueprints.welcome.BlueCast") as MockBlueCast, patch(
        "bluecast.blueprints.welcome.BlueCastCV"
    ) as MockBlueCastCV, patch(
        "bluecast.blueprints.welcome.BlueCastRegression"
    ) as MockBlueCastRegression, patch(
        "bluecast.blueprints.welcome.BlueCastCVRegression"
    ) as MockBlueCastCVRegression:

        # Binary classification, single model
        automl_instance = welcome.instantiate_bluecast_instance(
            "binary", 1, 5, "/path/to/oof", True, 100, 3600, True, True
        )
        MockBlueCast.assert_called_once()
        assert automl_instance == MockBlueCast.return_value

        # Binary classification, multiple models
        automl_instance = welcome.instantiate_bluecast_instance(
            "binary", 2, 5, "/path/to/oof", True, 100, 3600, True, True
        )
        MockBlueCastCV.assert_called_once()
        assert automl_instance == MockBlueCastCV.return_value

        # Regression, single model
        automl_instance = welcome.instantiate_bluecast_instance(
            "regression", 1, 5, "/path/to/oof", True, 100, 3600, True, True
        )
        MockBlueCastRegression.assert_called_once()
        assert automl_instance == MockBlueCastRegression.return_value

        # Regression, multiple models
        automl_instance = welcome.instantiate_bluecast_instance(
            "regression", 2, 5, "/path/to/oof", True, 100, 3600, True, True
        )
        MockBlueCastCVRegression.assert_called_once()
        assert automl_instance == MockBlueCastCVRegression.return_value


@patch("bluecast.blueprints.welcome.display")
def test_automl_configurator(mock_display, welcome):
    """
    Test the automl_configurator method to ensure that widgets are created and displayed.
    Also, check that the submit button triggers the correct function.
    """
    # Mock the on_submit_clicked method to verify it is called when the button is clicked
    with patch.object(welcome, "on_submit_clicked") as mock_on_submit:
        # Call the configurator method, which sets up the UI and event handlers
        welcome.automl_configurator()

        # Ensure widgets are displayed (mock_display should be called once)
        assert mock_display.called

        # Find the submit button from the displayed widgets
        submit_button = None
        for widget in mock_display.call_args[0]:
            if isinstance(widget, widgets.Button):
                submit_button = widget
                break

        assert (
            submit_button is not None
        ), "Submit button not found in displayed widgets."

        # Simulate a click event on the submit button
        submit_button.click()

        # Ensure that the on_submit_clicked function was called after button click
        mock_on_submit.assert_called_once()


@patch("bluecast.blueprints.welcome.clear_output")
def test_on_submit_called_via_configurator(mock_clear_output, welcome):
    """
    Test if on_submit_clicked is triggered after button click in the automl_configurator method.
    """
    task = MagicMock(value="binary")
    debug_mode = MagicMock(value=True)
    n_models = MagicMock(value=2)
    shap_values = MagicMock(value=True)
    n_folds = MagicMock(value=5)
    oof_storage_path = MagicMock(value="/path/to/oof")
    hyperparameter_tuning_rounds = MagicMock(value=100)
    hyperparameter_tuning_max_runtime_secs = MagicMock(value=3600)
    plot_hyperparameter_tuning_overview = MagicMock(value=True)
    show_detailed_tuning_logs = MagicMock(value=True)

    # Mock the instantiation of BlueCast to avoid creating real objects
    with patch.object(
        welcome, "instantiate_bluecast_instance", return_value=MagicMock()
    ) as mock_instantiate:
        welcome.on_submit_clicked(
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

        # Ensure clear_output is called
        mock_clear_output.assert_called_once()

        # Ensure instantiate_bluecast_instance is called with correct parameters
        mock_instantiate.assert_called_once_with(
            "binary", 2, 5, "/path/to/oof", True, 100, 3600, True, True
        )
