"""Behavioral tests for Qwen3 tool call type conversion.

These tests verify that tool call arguments are returned with proper JSON types
matching the OpenAI API specification. When a tool schema specifies a parameter
as integer, boolean, etc., the parsed arguments should contain those types,
not strings.

The tests are written from the client's perspective - they describe what the
client (e.g., Zed editor) expects to receive, not how the implementation works.
"""

import json
import unittest


class TestToolCallTypeConversion(unittest.TestCase):
    """Test that tool call arguments have proper JSON types matching OpenAI API."""

    def _parse_tool_call_with_schema(self, model_output: str, tools: list[dict]):
        """Parse model output with tools schema and return arguments dict.

        This helper simulates the full flow from model output to client response.
        """
        from mlx_batch_server.chat.mlx.tools.qwen3_moe_tools_parser import (
            Qwen3MoeToolParser,
        )
        from mlx_batch_server.chat.openai.openai_adapter import _convert_tool_calls

        parser = Qwen3MoeToolParser()

        # Pass schema to parser if it supports it
        if hasattr(parser, "set_tools_schema"):
            parser.set_tools_schema(tools)

        # Parse model output
        core_tool_calls = parser.parse_tools(model_output)
        if not core_tool_calls:
            return None

        # Convert to OpenAI format (this is what client receives)
        openai_tool_calls = _convert_tool_calls(core_tool_calls)

        # Parse the JSON arguments string (what client would do)
        return json.loads(openai_tool_calls[0].function.arguments)

    # =========================================================================
    # Integer parameter tests
    # =========================================================================

    def test_integer_parameter_returns_int_type(self):
        """When schema specifies integer, client should receive an int, not string."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                        },
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=read_file>
<parameter=path>example.py</parameter>
<parameter=start_line>10</parameter>
<parameter=end_line>20</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertEqual(args["path"], "example.py")
        self.assertIsInstance(args["start_line"], int)
        self.assertEqual(args["start_line"], 10)
        self.assertIsInstance(args["end_line"], int)
        self.assertEqual(args["end_line"], 20)

    def test_negative_integer_parameter(self):
        """Negative integers should be parsed correctly."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "seek",
                    "parameters": {
                        "type": "object",
                        "properties": {"offset": {"type": "integer"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=seek>
<parameter=offset>-5</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsInstance(args["offset"], int)
        self.assertEqual(args["offset"], -5)

    # =========================================================================
    # Boolean parameter tests
    # =========================================================================

    def test_boolean_true_parameter(self):
        """When schema specifies boolean, 'true' should become Python True."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "case_sensitive": {"type": "boolean"},
                        },
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=search>
<parameter=query>def main</parameter>
<parameter=case_sensitive>true</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsInstance(args["case_sensitive"], bool)
        self.assertTrue(args["case_sensitive"])

    def test_boolean_false_parameter(self):
        """When schema specifies boolean, 'false' should become Python False."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "regex": {"type": "boolean"},
                        },
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=search>
<parameter=regex>false</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsInstance(args["regex"], bool)
        self.assertFalse(args["regex"])

    # =========================================================================
    # Null parameter tests
    # =========================================================================

    def test_null_parameter_becomes_none(self):
        """When model outputs 'null', client should receive JSON null."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "encoding": {"type": "string"},
                        },
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=write_file>
<parameter=path>example.py</parameter>
<parameter=encoding>null</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertEqual(args["path"], "example.py")
        self.assertIsNone(args["encoding"])

    # =========================================================================
    # Number (float) parameter tests
    # =========================================================================

    def test_number_parameter_returns_float(self):
        """When schema specifies number, client should receive a float."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_temperature",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number"},
                        },
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=set_temperature>
<parameter=value>0.7</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsInstance(args["value"], float)
        self.assertEqual(args["value"], 0.7)

    # =========================================================================
    # Array parameter tests
    # =========================================================================

    def test_array_parameter_returns_list(self):
        """When schema specifies array, JSON array should be parsed to list."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "batch_delete",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "files": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=batch_delete>
<parameter=files>["file1.py", "file2.py", "file3.py"]</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsInstance(args["files"], list)
        self.assertEqual(args["files"], ["file1.py", "file2.py", "file3.py"])

    # =========================================================================
    # Object parameter tests
    # =========================================================================

    def test_object_parameter_returns_dict(self):
        """When schema specifies object, JSON object should be parsed to dict."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "configure",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "settings": {"type": "object"},
                        },
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=configure>
<parameter=settings>{"theme": "dark", "fontSize": 14}</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsInstance(args["settings"], dict)
        self.assertEqual(args["settings"], {"theme": "dark", "fontSize": 14})

    # =========================================================================
    # Mixed types in single call
    # =========================================================================

    def test_mixed_parameter_types(self):
        """A single tool call can have parameters of different types."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "complex_operation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "count": {"type": "integer"},
                            "ratio": {"type": "number"},
                            "enabled": {"type": "boolean"},
                            "tags": {"type": "array"},
                            "metadata": {"type": "object"},
                        },
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=complex_operation>
<parameter=name>test</parameter>
<parameter=count>42</parameter>
<parameter=ratio>3.14</parameter>
<parameter=enabled>true</parameter>
<parameter=tags>["a", "b"]</parameter>
<parameter=metadata>{"key": "value"}</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsInstance(args["name"], str)
        self.assertEqual(args["name"], "test")

        self.assertIsInstance(args["count"], int)
        self.assertEqual(args["count"], 42)

        self.assertIsInstance(args["ratio"], float)
        self.assertEqual(args["ratio"], 3.14)

        self.assertIsInstance(args["enabled"], bool)
        self.assertTrue(args["enabled"])

        self.assertIsInstance(args["tags"], list)
        self.assertEqual(args["tags"], ["a", "b"])

        self.assertIsInstance(args["metadata"], dict)
        self.assertEqual(args["metadata"], {"key": "value"})

    # =========================================================================
    # Fallback behavior (no schema or unknown function)
    # =========================================================================

    def test_without_schema_strings_are_preserved(self):
        """Without schema, string values should be preserved as-is."""
        tools = []  # No schema provided

        model_output = """<tool_call>
<function=unknown_function>
<parameter=value>hello world</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsInstance(args["value"], str)
        self.assertEqual(args["value"], "hello world")

    def test_unknown_function_preserves_strings(self):
        """If function not in schema, preserve string values."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "other_function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        model_output = """<tool_call>
<function=unknown_function>
<parameter=count>10</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        # Without schema info, we can't know the type, so string is acceptable
        # But if the value looks like a number, converting it is also acceptable
        self.assertIn(args["count"], [10, "10"])

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_integer_with_leading_zeros(self):
        """Leading zeros in integers should be handled correctly."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_mode",
                    "parameters": {
                        "type": "object",
                        "properties": {"mode": {"type": "integer"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=set_mode>
<parameter=mode>007</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsInstance(args["mode"], int)
        self.assertEqual(args["mode"], 7)

    def test_whitespace_around_values(self):
        """Whitespace around values should be trimmed before type conversion."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "parameters": {
                        "type": "object",
                        "properties": {"num": {"type": "integer"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=test>
<parameter=num>
  42
</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        self.assertIsInstance(args["num"], int)
        self.assertEqual(args["num"], 42)

    def test_invalid_integer_stays_as_string(self):
        """If integer conversion fails, value should remain as string."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "parameters": {
                        "type": "object",
                        "properties": {"num": {"type": "integer"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=test>
<parameter=num>not_a_number</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        # Should gracefully fall back to string
        self.assertEqual(args["num"], "not_a_number")

    def test_invalid_json_array_stays_as_string(self):
        """If JSON array parsing fails, value should remain as string."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "parameters": {
                        "type": "object",
                        "properties": {"items": {"type": "array"}},
                    },
                },
            }
        ]

        model_output = """<tool_call>
<function=test>
<parameter=items>[invalid json</parameter>
</function>
</tool_call>"""

        args = self._parse_tool_call_with_schema(model_output, tools)

        # Should gracefully fall back to string
        self.assertEqual(args["items"], "[invalid json")


if __name__ == "__main__":
    unittest.main()
