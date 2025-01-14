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
