"""Module to create Pydantic like validations."""

import inspect
from functools import wraps
from typing import Union, get_args, get_origin, get_type_hints


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
    origin = get_origin(
        expected_type
    )  # Extract the origin of the type (e.g., list for List[int])
    args = get_args(
        expected_type
    )  # Extract the type parameters (e.g., int for List[int])

    if origin is None:
        # If there's no origin, it's a non-parameterized type like int or str
        return isinstance(value, expected_type)
    elif origin is Union:
        # Handle Union types (e.g., Union[str, int])
        return any(_matches_type(value, t) for t in args)
    elif origin in {list, set, tuple}:
        # Handle parameterized collections like List[int], Set[str], Tuple[int, int]
        if not isinstance(value, origin):
            return False
        if args:
            if origin is tuple:
                # Special handling for tuples
                if len(args) == 2 and args[1] is Ellipsis:
                    # e.g., Tuple[int, ...] (variable-length tuple)
                    return all(_matches_type(v, args[0]) for v in value)
                if len(value) != len(args):
                    # Fixed-length tuple: lengths must match
                    return False
                return all(_matches_type(v, t) for v, t in zip(value, args))
            # For lists and sets, check all elements
            return all(_matches_type(v, args[0]) for v in value)
        return True
    elif origin is dict:
        # Handle parameterized dictionaries like Dict[str, int]
        if not isinstance(value, dict):
            return False
        if args:
            key_type, value_type = args
            return all(
                _matches_type(k, key_type) and _matches_type(v, value_type)
                for k, v in value.items()
            )
        return True
    else:
        # For other parameterized or unsupported types, return False
        return False
