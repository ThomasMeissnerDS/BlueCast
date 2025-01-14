from typing import Dict, Optional, Tuple

import pytest

from bluecast.config.config_validations import check_types_init


# Sample class to test the decorator
class TestClass:
    @check_types_init
    def __init__(
        self,
        a: int,
        b: str,
        c: list[int],
        d: Optional[dict[str, float]] = None,
        e: Optional[Tuple[str]] = None,
        f: Optional[Dict[str, float]] = None,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f


@pytest.fixture
def valid_instance():
    return TestClass(
        1,
        "hello",
        [1, 2, 3],
        {"key1": 1.0, "key2": 2.5},
        ("a", "b"),
        {"key1": 1.0, "key2": 2.5},
    )
