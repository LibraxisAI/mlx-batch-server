import unittest

from mlx_batch_server.chat.mlx.tools.glm45_tools_parser import GLM45ToolParser


class TestGLM45ToolParser(unittest.TestCase):
    def setUp(self):
        self.tools_parser = GLM45ToolParser()

    def test_glm45_decode_single_tool_call(self):
        # Test single tool call in GLM-4.5 format
        text = """<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>San Francisco</arg_value>
<arg_key>unit</arg_key>
<arg_value>fahrenheit</arg_value>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "get_weather")
        self.assertEqual(
            tool_call.arguments, {"location": "San Francisco", "unit": "fahrenheit"}
        )

    def test_glm45_decode_multiple_tool_calls(self):
        # Test multiple tool calls
        text = """<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>San Francisco</arg_value>
<arg_key>unit</arg_key>
<arg_value>fahrenheit</arg_value>
</tool_call>
<tool_call>send_email
<arg_key>to</arg_key>
<arg_value>john@example.com</arg_value>
<arg_key>subject</arg_key>
<arg_value>Meeting Tomorrow</arg_value>
<arg_key>body</arg_key>
<arg_value>Hi John, just confirming our meeting scheduled for tomorrow. Best regards!</arg_value>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

        # Check first tool call
        tool_call1 = result[0]
        self.assertEqual(tool_call1.name, "get_weather")
        self.assertEqual(
            tool_call1.arguments, {"location": "San Francisco", "unit": "fahrenheit"}
        )

        # Check second tool call
        tool_call2 = result[1]
        self.assertEqual(tool_call2.name, "send_email")
        self.assertEqual(
            tool_call2.arguments,
            {
                "to": "john@example.com",
                "subject": "Meeting Tomorrow",
                "body": "Hi John, just confirming our meeting scheduled for tomorrow. Best regards!",
            },
        )

    def test_glm45_decode_tool_call_with_whitespace(self):
        # Test tool call with extra whitespace
        text = """  <tool_call>  get_weather
  <arg_key>  location  </arg_key>
  <arg_value>  San Francisco  </arg_value>
  <arg_key>  unit  </arg_key>
  <arg_value>  fahrenheit  </arg_value>
  </tool_call>  """
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "get_weather")
        self.assertEqual(
            tool_call.arguments, {"location": "San Francisco", "unit": "fahrenheit"}
        )

    def test_glm45_decode_tool_call_no_parameters(self):
        # Test tool call without parameters
        text = """<tool_call>list_files
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "list_files")
        self.assertEqual(tool_call.arguments, {})

    def test_glm45_decode_invalid_tool_call_no_function(self):
        # Test invalid tool call format (missing function name)
        text = """<tool_call>
<arg_key>location</arg_key>
<arg_value>San Francisco</arg_value>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        # Should return None because function name couldn't be extracted
        self.assertIsNone(result)

    def test_glm45_decode_malformed_xml(self):
        # Test malformed XML (missing closing arg_value tag)
        text = """<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>San Francisco</arg_value>
<arg_key>unit</arg_key>
<arg_value>fahrenheit
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        # Should still extract what it can
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "get_weather")
        # Only the complete arg_key/arg_value pair should be extracted
        self.assertEqual(result[0].arguments, {"location": "San Francisco"})

    def test_glm45_decode_non_tool_call(self):
        # Test regular text without tool call
        text = "This is a regular message without any tool calls."
        result = self.tools_parser.parse_tools(text)

        self.assertIsNone(result)  # Should return None for non-tool call text

    def test_glm45_decode_mixed_content(self):
        # Test tool call mixed with regular text
        text = """Here's the weather information:

<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>San Francisco</arg_value>
<arg_key>unit</arg_key>
<arg_value>fahrenheit</arg_value>
</tool_call>

Let me check that for you."""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "get_weather")
        self.assertEqual(
            tool_call.arguments, {"location": "San Francisco", "unit": "fahrenheit"}
        )

    def test_glm45_strict_mode(self):
        # Test strict mode behavior
        self.tools_parser.strict_mode = True

        # Valid format should work
        text = """<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>San Francisco</arg_value>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)
        self.assertIsNotNone(result)

        # Mixed content should not work in strict mode
        text_mixed = """Some text before
<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>San Francisco</arg_value>
</tool_call>
Some text after"""
        result_mixed = self.tools_parser.parse_tools(text_mixed)
        self.assertIsNone(result_mixed)

    def test_glm45_decode_multiline_parameter(self):
        # Test parameter with multiline content
        text = """<tool_call>send_email
<arg_key>body</arg_key>
<arg_value>Hello,

This is a multiline email body.
It contains multiple paragraphs.

Best regards,
Assistant</arg_value>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        tool_call = result[0]
        self.assertEqual(tool_call.name, "send_email")
        expected_body = """Hello,

This is a multiline email body.
It contains multiple paragraphs.

Best regards,
Assistant"""
        self.assertEqual(tool_call.arguments["body"], expected_body)

    def test_glm45_decode_function_with_dots(self):
        # Test function name with dots (like browser.search)
        text = """<tool_call>browser.search
<arg_key>query</arg_key>
<arg_value>python tutorials</arg_value>
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "browser.search")
        self.assertEqual(result[0].arguments, {"query": "python tutorials"})

    def test_glm45_decode_empty_input(self):
        # Test empty input
        result = self.tools_parser.parse_tools("")
        self.assertIsNone(result)

        result = self.tools_parser.parse_tools(None)
        self.assertIsNone(result)

    def test_glm45_decode_incomplete_tool_call(self):
        # Test incomplete tool call (no closing tag)
        text = """<tool_call>get_weather
<arg_key>location</arg_key>
<arg_value>San Francisco</arg_value>"""
        result = self.tools_parser.parse_tools(text)

        # Should return None because tool_call block is incomplete
        self.assertIsNone(result)

    def test_glm45_tool_call_id_generation(self):
        # Test that tool calls get unique IDs
        text = """<tool_call>func1
</tool_call>
<tool_call>func2
</tool_call>"""
        result = self.tools_parser.parse_tools(text)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

        # IDs should be different
        self.assertNotEqual(result[0].id, result[1].id)
        # IDs should follow the expected format
        self.assertTrue(result[0].id.startswith("call_"))
        self.assertTrue(result[1].id.startswith("call_"))


if __name__ == "__main__":
    unittest.main()
