import unittest
from typing import Any

from mlx_batch_server.chat.mlx.tools.glm45_tools_parser import GLM45ToolParser


class TestGLM45ToolCallTypeConversion(unittest.TestCase):
    """Test that GLM-4.5 tool call arguments have proper JSON types based on schema."""

    def _parse_tool_call_with_schema(
        self, model_output: str, tools: list[dict[str, Any]]
    ):
        """Helper to parse GLM-4.5 model output with tools schema.

        Args:
            model_output: Raw model output containing tool call
            tools: List of tool definitions in OpenAI format

        Returns:
            Parsed tool call result
        """
        parser = GLM45ToolParser()
        parser.set_tools_schema(tools)
        return parser.parse_tools(model_output)

    def test_integer_parameter_returns_int_type(self):
        """Test that integer parameters are converted to Python int."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_volume",
                    "parameters": {
                        "type": "object",
                        "properties": {"level": {"type": "integer"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>set_volume
<arg_key>level</arg_key>
<arg_value>75</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].arguments["level"], 75)
        self.assertIsInstance(result[0].arguments["level"], int)

    def test_negative_integer_parameter(self):
        """Test that negative integers are handled correctly."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_temperature",
                    "parameters": {
                        "type": "object",
                        "properties": {"celsius": {"type": "integer"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>set_temperature
<arg_key>celsius</arg_key>
<arg_value>-10</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        self.assertEqual(result[0].arguments["celsius"], -10)
        self.assertIsInstance(result[0].arguments["celsius"], int)

    def test_boolean_true_parameter(self):
        """Test that boolean true is converted to Python True."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "toggle_feature",
                    "parameters": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>toggle_feature
<arg_key>enabled</arg_key>
<arg_value>true</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        self.assertEqual(result[0].arguments["enabled"], True)
        self.assertIsInstance(result[0].arguments["enabled"], bool)

    def test_boolean_false_parameter(self):
        """Test that boolean false is converted to Python False."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "toggle_feature",
                    "parameters": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>toggle_feature
<arg_key>enabled</arg_key>
<arg_value>false</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        self.assertEqual(result[0].arguments["enabled"], False)
        self.assertIsInstance(result[0].arguments["enabled"], bool)

    def test_null_parameter_becomes_none(self):
        """Test that null values are converted to Python None."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "update_record",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>update_record
<arg_key>value</arg_key>
<arg_value>null</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        self.assertIsNone(result[0].arguments["value"])

    def test_number_parameter_returns_float(self):
        """Test that number parameters are converted to Python float."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_price",
                    "parameters": {
                        "type": "object",
                        "properties": {"amount": {"type": "number"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>set_price
<arg_key>amount</arg_key>
<arg_value>19.99</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        self.assertEqual(result[0].arguments["amount"], 19.99)
        self.assertIsInstance(result[0].arguments["amount"], float)

    def test_array_parameter_returns_list(self):
        """Test that array parameters are parsed from JSON to Python list."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process_items",
                    "parameters": {
                        "type": "object",
                        "properties": {"items": {"type": "array"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>process_items
<arg_key>items</arg_key>
<arg_value>["apple", "banana", "cherry"]</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        self.assertEqual(result[0].arguments["items"], ["apple", "banana", "cherry"])
        self.assertIsInstance(result[0].arguments["items"], list)

    def test_object_parameter_returns_dict(self):
        """Test that object parameters are parsed from JSON to Python dict."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "update_config",
                    "parameters": {
                        "type": "object",
                        "properties": {"settings": {"type": "object"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>update_config
<arg_key>settings</arg_key>
<arg_value>{"theme": "dark", "fontSize": 14}</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        self.assertEqual(
            result[0].arguments["settings"], {"theme": "dark", "fontSize": 14}
        )
        self.assertIsInstance(result[0].arguments["settings"], dict)

    def test_mixed_parameter_types(self):
        """Test function with multiple parameter types."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "quantity": {"type": "integer"},
                            "price": {"type": "number"},
                            "express": {"type": "boolean"},
                            "items": {"type": "array"},
                            "notes": {"type": "string"},
                        },
                    },
                },
            }
        ]

        model_output = """<tool_call>create_order
<arg_key>quantity</arg_key>
<arg_value>5</arg_value>
<arg_key>price</arg_key>
<arg_value>29.99</arg_value>
<arg_key>express</arg_key>
<arg_value>true</arg_value>
<arg_key>items</arg_key>
<arg_value>["widget", "gadget"]</arg_value>
<arg_key>notes</arg_key>
<arg_value>Handle with care</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        args = result[0].arguments

        self.assertEqual(args["quantity"], 5)
        self.assertIsInstance(args["quantity"], int)

        self.assertEqual(args["price"], 29.99)
        self.assertIsInstance(args["price"], float)

        self.assertEqual(args["express"], True)
        self.assertIsInstance(args["express"], bool)

        self.assertEqual(args["items"], ["widget", "gadget"])
        self.assertIsInstance(args["items"], list)

        self.assertEqual(args["notes"], "Handle with care")
        self.assertIsInstance(args["notes"], str)

    def test_without_schema_strings_are_preserved(self):
        """Test that without schema, all values remain as strings."""
        parser = GLM45ToolParser()
        # Don't set schema

        model_output = """<tool_call>some_function
<arg_key>number</arg_key>
<arg_value>42</arg_value>
<arg_key>flag</arg_key>
<arg_value>true</arg_value>
</tool_call>"""

        result = parser.parse_tools(model_output)

        self.assertIsNotNone(result)
        args = result[0].arguments

        # Without schema, values should remain as strings
        self.assertEqual(args["number"], "42")
        self.assertIsInstance(args["number"], str)

        self.assertEqual(args["flag"], "true")
        self.assertIsInstance(args["flag"], str)

    def test_invalid_integer_stays_as_string(self):
        """Test that invalid integer values stay as strings."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_count",
                    "parameters": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>set_count
<arg_key>count</arg_key>
<arg_value>not_a_number</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        # Should stay as string since conversion failed
        self.assertEqual(result[0].arguments["count"], "not_a_number")
        self.assertIsInstance(result[0].arguments["count"], str)

    def test_invalid_json_array_stays_as_string(self):
        """Test that invalid JSON array values stay as strings."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process_list",
                    "parameters": {
                        "type": "object",
                        "properties": {"items": {"type": "array"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>process_list
<arg_key>items</arg_key>
<arg_value>[invalid json</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        # Should stay as string since JSON parsing failed
        self.assertEqual(result[0].arguments["items"], "[invalid json")
        self.assertIsInstance(result[0].arguments["items"], str)

    def test_case_insensitive_boolean(self):
        """Test that boolean parsing is case-insensitive."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "toggle",
                    "parameters": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                    },
                },
            }
        ]

        for true_value in ["true", "True", "TRUE"]:
            model_output = f"""<tool_call>toggle
<arg_key>enabled</arg_key>
<arg_value>{true_value}</arg_value>
</tool_call>"""

            result = self._parse_tool_call_with_schema(model_output, tools)
            self.assertIsNotNone(result)
            self.assertEqual(result[0].arguments["enabled"], True)

        for false_value in ["false", "False", "FALSE"]:
            model_output = f"""<tool_call>toggle
<arg_key>enabled</arg_key>
<arg_value>{false_value}</arg_value>
</tool_call>"""

            result = self._parse_tool_call_with_schema(model_output, tools)
            self.assertIsNotNone(result)
            self.assertEqual(result[0].arguments["enabled"], False)

    def test_case_insensitive_null(self):
        """Test that null parsing is case-insensitive."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "clear",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                },
            }
        ]

        for null_value in ["null", "Null", "NULL"]:
            model_output = f"""<tool_call>clear
<arg_key>value</arg_key>
<arg_value>{null_value}</arg_value>
</tool_call>"""

            result = self._parse_tool_call_with_schema(model_output, tools)
            self.assertIsNotNone(result)
            self.assertIsNone(result[0].arguments["value"])

    def test_unknown_parameter_stays_as_string(self):
        """Test that parameters not in schema stay as strings."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "do_something",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "known": {"type": "integer"},
                        },
                    },
                },
            }
        ]

        model_output = """<tool_call>do_something
<arg_key>known</arg_key>
<arg_value>42</arg_value>
<arg_key>unknown</arg_key>
<arg_value>some value</arg_value>
</tool_call>"""

        result = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsNotNone(result)
        args = result[0].arguments

        # Known parameter should be converted
        self.assertEqual(args["known"], 42)
        self.assertIsInstance(args["known"], int)

        # Unknown parameter should stay as string
        self.assertEqual(args["unknown"], "some value")
        self.assertIsInstance(args["unknown"], str)


if __name__ == "__main__":
    unittest.main()
