from typing import Dict, List, Optional, Set, Tuple, Union

import pytest

from bluecast.config.config_validations import check_types_init, _matches_type


# Sample class to test the decorator
class SampleClass:
    @check_types_init
    def __init__(
        self,
        a: int,
        b: str,
        c: list[int],
        d: Optional[dict[str, float]] = None,
        e: Optional[Tuple[str, str]] = None,  # Fixed to match two-element tuple
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
    return SampleClass(
        1,
        "hello",
        [1, 2, 3],
        {"key1": 1.0, "key2": 2.5},
        ("a", "b"),
        {"key1": 1.0, "key2": 2.5},
    )


class TestMatchesType:
    """Test the _matches_type function with various type scenarios."""

    def test_basic_types(self):
        """Test basic non-parameterized types."""
        assert _matches_type(5, int) is True
        assert _matches_type("hello", str) is True
        assert _matches_type(3.14, float) is True
        assert _matches_type(True, bool) is True
        
        # Wrong types
        assert _matches_type("hello", int) is False
        assert _matches_type(5, str) is False
        assert _matches_type(3.14, int) is False

    def test_union_types(self):
        """Test Union type validation."""
        union_type = Union[str, int]
        
        assert _matches_type("hello", union_type) is True
        assert _matches_type(42, union_type) is True
        assert _matches_type(3.14, union_type) is False
        assert _matches_type([], union_type) is False

    def test_optional_types(self):
        """Test Optional type validation (which is Union[T, None])."""
        optional_int = Optional[int]
        
        assert _matches_type(42, optional_int) is True
        assert _matches_type(None, optional_int) is True
        assert _matches_type("hello", optional_int) is False

    def test_list_validation(self):
        """Test List type validation."""
        # Valid lists
        assert _matches_type([1, 2, 3], List[int]) is True
        assert _matches_type(["a", "b", "c"], List[str]) is True
        assert _matches_type([], List[int]) is True  # Empty list is valid
        
        # Invalid element types
        assert _matches_type([1, "2", 3], List[int]) is False
        assert _matches_type(["a", 2, "c"], List[str]) is False
        
        # Not a list
        assert _matches_type((1, 2, 3), List[int]) is False
        assert _matches_type({1, 2, 3}, List[int]) is False
        assert _matches_type("hello", List[str]) is False

    def test_set_validation(self):
        """Test Set type validation."""
        # Valid sets
        assert _matches_type({1, 2, 3}, Set[int]) is True
        assert _matches_type({"a", "b", "c"}, Set[str]) is True
        assert _matches_type(set(), Set[int]) is True  # Empty set is valid
        
        # Invalid element types
        assert _matches_type({1, "2", 3}, Set[int]) is False
        assert _matches_type({"a", 2, "c"}, Set[str]) is False
        
        # Not a set
        assert _matches_type([1, 2, 3], Set[int]) is False
        assert _matches_type((1, 2, 3), Set[int]) is False

    def test_tuple_fixed_length_validation(self):
        """Test fixed-length Tuple type validation."""
        # Valid fixed-length tuples
        assert _matches_type((1, "hello"), Tuple[int, str]) is True
        assert _matches_type((1, 2, 3), Tuple[int, int, int]) is True
        # Test empty tuple with basic tuple type
        assert _matches_type((), tuple) is True
        
        # Wrong length
        assert _matches_type((1,), Tuple[int, str]) is False
        assert _matches_type((1, "hello", 3), Tuple[int, str]) is False
        assert _matches_type((1, 2), Tuple[int, int, int]) is False
        
        # Wrong types in correct positions
        assert _matches_type(("hello", 1), Tuple[int, str]) is False
        assert _matches_type((1, 2, "3"), Tuple[int, int, int]) is False
        
        # Not a tuple
        assert _matches_type([1, "hello"], Tuple[int, str]) is False

    def test_tuple_variable_length_validation(self):
        """Test variable-length Tuple type validation (Tuple[T, ...])."""
        # Valid variable-length tuples
        assert _matches_type((1, 2, 3, 4), Tuple[int, ...]) is True
        assert _matches_type((1,), Tuple[int, ...]) is True
        assert _matches_type((), Tuple[int, ...]) is True  # Empty tuple
        assert _matches_type(("a", "b", "c"), Tuple[str, ...]) is True
        
        # Invalid element types
        assert _matches_type((1, "2", 3), Tuple[int, ...]) is False
        assert _matches_type(("a", 2, "c"), Tuple[str, ...]) is False
        
        # Not a tuple
        assert _matches_type([1, 2, 3], Tuple[int, ...]) is False

    def test_dict_validation(self):
        """Test Dict type validation."""
        # Valid dictionaries
        assert _matches_type({"a": 1, "b": 2}, Dict[str, int]) is True
        assert _matches_type({1: "a", 2: "b"}, Dict[int, str]) is True
        assert _matches_type({}, Dict[str, int]) is True  # Empty dict is valid
        
        # Invalid key types
        assert _matches_type({1: 1, "b": 2}, Dict[str, int]) is False
        assert _matches_type({"a": 1, 2: 2}, Dict[str, int]) is False
        
        # Invalid value types
        assert _matches_type({"a": 1, "b": "2"}, Dict[str, int]) is False
        assert _matches_type({"a": "1", "b": 2}, Dict[str, int]) is False
        
        # Not a dictionary
        assert _matches_type([("a", 1), ("b", 2)], Dict[str, int]) is False
        assert _matches_type("hello", Dict[str, int]) is False

    def test_nested_collections(self):
        """Test nested collection types."""
        # List of lists
        assert _matches_type([[1, 2], [3, 4]], List[List[int]]) is True
        assert _matches_type([[1, "2"], [3, 4]], List[List[int]]) is False
        
        # Dict with list values
        assert _matches_type({"a": [1, 2], "b": [3, 4]}, Dict[str, List[int]]) is True
        assert _matches_type({"a": [1, "2"], "b": [3, 4]}, Dict[str, List[int]]) is False
        
        # Tuple of different types
        assert _matches_type(([1, 2], {"a": 1}), Tuple[List[int], Dict[str, int]]) is True
        assert _matches_type(([1, "2"], {"a": 1}), Tuple[List[int], Dict[str, int]]) is False

    def test_unparameterized_collections(self):
        """Test collections without type parameters."""
        # Should return True for any content when no type args
        assert _matches_type([1, "2", 3.14], list) is True
        assert _matches_type({1, "2", 3.14}, set) is True
        assert _matches_type((1, "2", 3.14), tuple) is True
        assert _matches_type({"a": 1, 2: "b"}, dict) is True

    def test_unsupported_origins(self):
        """Test behavior with unsupported generic origins."""
        # This tests the final else clause in _matches_type
        # We need to create a mock type with an origin that's not handled
        from typing import Callable
        
        # Callable is an example of a parameterized type not explicitly handled
        assert _matches_type(lambda x: x, Callable[[int], int]) is False


class TestCheckTypesInit:
    """Test the check_types_init decorator with various scenarios."""

    def test_valid_initialization(self, valid_instance):
        """Test that valid initialization works correctly."""
        assert valid_instance.a == 1
        assert valid_instance.b == "hello"
        assert valid_instance.c == [1, 2, 3]

    def test_invalid_basic_type(self):
        """Test that invalid basic types raise TypeError."""
        with pytest.raises(TypeError, match="Argument 'a' must be of type"):
            SampleClass("not_an_int", "hello", [1, 2, 3])
        
        with pytest.raises(TypeError, match="Argument 'b' must be of type"):
            SampleClass(1, 123, [1, 2, 3])

    def test_invalid_list_elements(self):
        """Test that invalid list element types raise TypeError."""
        with pytest.raises(TypeError, match="Argument 'c' must be of type"):
            SampleClass(1, "hello", [1, "not_an_int", 3])
        
        with pytest.raises(TypeError, match="Argument 'c' must be of type"):
            SampleClass(1, "hello", "not_a_list")

    def test_invalid_optional_dict(self):
        """Test that invalid optional dict types raise TypeError."""
        with pytest.raises(TypeError, match="Argument 'd' must be of type"):
            SampleClass(1, "hello", [1, 2, 3], {"key": "not_a_float"})
        
        with pytest.raises(TypeError, match="Argument 'd' must be of type"):
            SampleClass(1, "hello", [1, 2, 3], {123: 1.0})  # Wrong key type

    def test_invalid_tuple(self):
        """Test that invalid tuple types raise TypeError."""
        with pytest.raises(TypeError, match="Argument 'e' must be of type"):
            SampleClass(1, "hello", [1, 2, 3], None, (1, 2))  # Wrong element type
        
        with pytest.raises(TypeError, match="Argument 'e' must be of type"):
            SampleClass(1, "hello", [1, 2, 3], None, ("a", "b", "c"))  # Wrong length

    def test_none_values_for_optional(self):
        """Test that None values work for Optional parameters."""
        instance = SampleClass(1, "hello", [1, 2, 3], None, None, None)
        assert instance.d is None
        assert instance.e is None
        assert instance.f is None


class TestComplexScenarios:
    """Test complex scenarios for comprehensive coverage."""

    class ComplexTestClass:
        @check_types_init
        def __init__(
            self,
            list_param: List[str],
            set_param: Set[int],
            fixed_tuple: Tuple[int, str, float],
            var_tuple: Tuple[int, ...],
            dict_param: Dict[str, List[int]],
            union_param: Union[str, List[int]],
        ):
            self.list_param = list_param
            self.set_param = set_param
            self.fixed_tuple = fixed_tuple
            self.var_tuple = var_tuple
            self.dict_param = dict_param
            self.union_param = union_param

    def test_complex_valid_types(self):
        """Test complex valid type combinations."""
        instance = self.ComplexTestClass(
            list_param=["a", "b", "c"],
            set_param={1, 2, 3},
            fixed_tuple=(1, "hello", 3.14),
            var_tuple=(1, 2, 3, 4, 5),
            dict_param={"key1": [1, 2], "key2": [3, 4]},
            union_param="string_value"
        )
        assert instance.list_param == ["a", "b", "c"]
        
        # Test union with list
        instance2 = self.ComplexTestClass(
            list_param=["a", "b"],
            set_param={1, 2},
            fixed_tuple=(1, "hello", 3.14),
            var_tuple=(1,),
            dict_param={"key": [1]},
            union_param=[1, 2, 3]  # List variant of union
        )
        assert instance2.union_param == [1, 2, 3]

    def test_complex_invalid_types(self):
        """Test complex invalid type combinations."""
        # Invalid list elements
        with pytest.raises(TypeError):
            self.ComplexTestClass(
                list_param=["a", 1, "c"],  # Mixed types
                set_param={1, 2, 3},
                fixed_tuple=(1, "hello", 3.14),
                var_tuple=(1, 2),
                dict_param={"key": [1]},
                union_param="test"
            )
        
        # Invalid set elements
        with pytest.raises(TypeError):
            self.ComplexTestClass(
                list_param=["a", "b"],
                set_param={1, "2", 3},  # Mixed types
                fixed_tuple=(1, "hello", 3.14),
                var_tuple=(1, 2),
                dict_param={"key": [1]},
                union_param="test"
            )
        
        # Invalid fixed tuple length
        with pytest.raises(TypeError):
            self.ComplexTestClass(
                list_param=["a", "b"],
                set_param={1, 2, 3},
                fixed_tuple=(1, "hello"),  # Too short
                var_tuple=(1, 2),
                dict_param={"key": [1]},
                union_param="test"
            )
        
        # Invalid variable tuple elements
        with pytest.raises(TypeError):
            self.ComplexTestClass(
                list_param=["a", "b"],
                set_param={1, 2, 3},
                fixed_tuple=(1, "hello", 3.14),
                var_tuple=(1, "not_int", 3),  # Mixed types
                dict_param={"key": [1]},
                union_param="test"
            )
        
        # Invalid dict value types
        with pytest.raises(TypeError):
            self.ComplexTestClass(
                list_param=["a", "b"],
                set_param={1, 2, 3},
                fixed_tuple=(1, "hello", 3.14),
                var_tuple=(1, 2),
                dict_param={"key": ["not", "int", "list"]},  # Wrong value type
                union_param="test"
            )
        
        # Invalid union type
        with pytest.raises(TypeError):
            self.ComplexTestClass(
                list_param=["a", "b"],
                set_param={1, 2, 3},
                fixed_tuple=(1, "hello", 3.14),
                var_tuple=(1, 2),
                dict_param={"key": [1]},
                union_param=3.14  # Neither str nor List[int]
            )
