import pytest
from bluecast.config.config_validations import check_types_init

# Sample class to test the decorator
class TestClass:
    @check_types_init
    def __init__(self, a: int, b: str, c: list[int], d: dict[str, float] = None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

@pytest.fixture

def valid_instance():
    return TestClass(1, "hello", [1, 2, 3], {"key1": 1.0, "key2": 2.5})
