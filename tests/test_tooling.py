"""Tests for Tool class and @tool decorator."""
import pytest
from lib.tooling import Tool, tool


def sample_func(query: str, limit: int = 5) -> str:
    """Search for games matching the query."""
    return f"Results for {query}"


class TestTool:

    def test_tool_name_defaults_to_function_name(self):
        t = Tool(sample_func)
        assert t.name == "sample_func"

    def test_tool_custom_name(self):
        t = Tool(sample_func, name="my_search")
        assert t.name == "my_search"

    def test_tool_description_from_docstring(self):
        t = Tool(sample_func)
        assert "Search for games" in t.description

    def test_tool_custom_description(self):
        t = Tool(sample_func, description="Custom description")
        assert t.description == "Custom description"

    def test_tool_parameters_extracted(self):
        t = Tool(sample_func)
        param_names = [p["name"] for p in t.parameters]
        assert "query" in param_names
        assert "limit" in param_names

    def test_tool_required_parameter(self):
        t = Tool(sample_func)
        params = {p["name"]: p for p in t.parameters}
        assert params["query"]["required"] is True
        assert params["limit"]["required"] is False

    def test_tool_dict_format(self):
        t = Tool(sample_func)
        d = t.dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "sample_func"
        assert "parameters" in d["function"]
        assert "properties" in d["function"]["parameters"]

    def test_tool_dict_has_required_list(self):
        t = Tool(sample_func)
        d = t.dict()
        required = d["function"]["parameters"]["required"]
        assert "query" in required
        assert "limit" not in required

    def test_tool_callable(self):
        t = Tool(sample_func)
        result = t(query="mario")
        assert "mario" in result

    def test_tool_from_func(self):
        t = Tool.from_func(sample_func)
        assert t.name == "sample_func"

    def test_tool_repr(self):
        t = Tool(sample_func)
        assert "sample_func" in repr(t)


class TestToolDecorator:

    def test_tool_decorator_creates_tool(self):
        @tool
        def my_func(q: str) -> str:
            """A test tool."""
            return q
        assert isinstance(my_func, Tool)

    def test_tool_decorator_with_name(self):
        @tool(name="custom_name")
        def my_func(q: str) -> str:
            """A test tool."""
            return q
        assert my_func.name == "custom_name"

    def test_tool_decorator_callable(self):
        @tool
        def add_one(n: int) -> int:
            """Adds one."""
            return n + 1
        result = add_one(n=5)
        assert result == 6

    def test_tool_infers_string_type(self):
        @tool
        def func(name: str) -> str:
            """Test."""
            return name
        d = func.dict()
        props = d["function"]["parameters"]["properties"]
        assert props["name"]["type"] == "string"

    def test_tool_infers_integer_type(self):
        @tool
        def func(count: int) -> str:
            """Test."""
            return str(count)
        d = func.dict()
        props = d["function"]["parameters"]["properties"]
        assert props["count"]["type"] == "integer"
