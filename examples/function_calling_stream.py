import json

from openai import OpenAI

# Configure client to use local server
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Point to ollama server
    # base_url="http://localhost:8100/v1",  # Point to mlx batch server
    api_key="not-needed",  # API key is not required for local server
)

# Define functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_recipe",
            "description": "Generate a recipe based on the user's input",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the recipe.",
                    },
                    "ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ingredients required for the recipe.",
                    },
                    "instructions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Step-by-step instructions for the recipe.",
                    },
                },
                "required": ["title", "ingredients", "instructions"],
                "additionalProperties": False,
            },
        },
    }
]

response_stream = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert cook who can help turn any user input into a delicious recipe."
                "As soon as the user tells you what they want, use the generate_recipe tool to create a detailed recipe for them."
            ),
        },
        {
            "role": "user",
            "content": "I want to make pancakes for 4.",
        },
    ],
    tools=tools,
    stream=True,
)

function_arguments = ""
function_name = ""
is_collecting_function_args = False

for part in response_stream:
    delta = part.choices[0].delta
    finish_reason = part.choices[0].finish_reason

    # Process assistant content
    if "content" in delta:
        print("Assistant:", delta.content)

    if delta.tool_calls:
        is_collecting_function_args = True
        tool_call = delta.tool_calls[0]

        if tool_call.function.name:
            function_name = tool_call.function.name
            print(f"Function name: '{function_name}'")

        # Process function arguments delta
        if tool_call.function.arguments:
            function_arguments += tool_call.function.arguments
            print(f"Arguments: {function_arguments}")

    # Process tool call with complete arguments
    if finish_reason == "tool_calls" and is_collecting_function_args:
        print(f"Function call '{function_name}' is complete.")
        args = json.loads(function_arguments)
        print("Complete function arguments:")
        print(json.dumps(args, indent=2))

        # Reset for the next potential function call
        function_arguments = ""
        function_name = ""
        is_collecting_function_args = False
