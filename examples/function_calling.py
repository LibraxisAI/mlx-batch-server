from openai import OpenAI

# Configure client to use local server
client = OpenAI(
    # base_url="http://localhost:11434/v1",  # Point to ollama server
    base_url="http://localhost:8100/v1",  # Point to mlx batch server
    api_key="not-needed",  # API key is not required for local server
)


def chat(model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

    return client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto"
    )


if __name__ == "__main__":
    completion = chat(model="mlx-community/Llama-3.2-3B-Instruct-4bit")
    message = completion.choices[0].message
    print(f"message: {message}")
    if message.tool_calls:
        print("success")
    else:
        print("failed")
