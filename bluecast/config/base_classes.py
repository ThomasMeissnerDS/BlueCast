import inspect
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Dict, Literal, Union, get_args, get_origin, get_type_hints

import pandas as pd

from bluecast.config.training_config import TrainingConfig


class BaseClassExperimentTracker(ABC):
    """Base class for the experiment tracker.

    Enforces the implementation of the add_results and retrieve_results_as_df methods.
    """

    @abstractmethod
    def add_results(
        self,
        experiment_id: int,
        score_category: Literal["simple_train_test_score", "cv_score", "oof_score"],
        training_config: TrainingConfig,
        model_parameters: Dict[Any, Any],
        eval_scores: Union[float, int, None],
        metric_used: str,
        metric_higher_is_better: bool,
    ) -> None:
        """
        Add results to the ExperimentTracker class.
        """
        pass

    @abstractmethod
    def retrieve_results_as_df(self) -> pd.DataFrame:
        """
        Retrieve results from the ExperimentTracker class
        """
        pass


def check_types_init(init_method):
    @wraps(init_method)
    def wrapper(self, *args, **kwargs):
        sig = inspect.signature(init_method)
        type_hints = get_type_hints(init_method)

        bound_arguments = sig.bind(self, *args, **kwargs)
        bound_arguments.apply_defaults()

        for name, value in bound_arguments.arguments.items():
            if name == "self":
                continue

            expected_type = type_hints.get(name)
            if expected_type is None:
                continue

            # A small helper function to handle Union/Optional:
            if not _matches_type(value, expected_type):
                raise TypeError(
                    f"Argument '{name}' must be of type '{expected_type}', "
                    f"but got value '{value}' of type '{type(value)}'."
                )

        return init_method(self, *args, **kwargs)

    return wrapper


def _matches_type(value, expected_type) -> bool:
    """Return True if 'value' matches the 'expected_type' annotation."""
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is None:
        # expected_type is a regular (non-parameterized) type like int or float
        return isinstance(value, expected_type)
    elif origin is Union:
        # e.g. Union[str, int]
        return any(_matches_type(value, t) for t in args)
    else:
        # fallback to a direct isinstance check
        return isinstance(value, expected_type)
