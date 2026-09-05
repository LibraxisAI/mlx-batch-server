<div align="center">

# Anthropic API Documentation

*Complete Anthropic Claude API compatibility for local MLX inference*

[Installation](#-installation--setup) • [Quick Start](#-basic-usage) • [Messages API](#-messages-api) • [Advanced Features](#-advanced-features)

</div>

---

MLX Batch Server provides full Anthropic Claude API compatibility, enabling seamless integration with existing Anthropic SDK clients while leveraging local MLX inference on Apple Silicon.

## 🚀 Installation & Setup

```bash
pip install mlx-batch-server
mlx-batch-server  # Start the server
```

## ⚡ Basic Usage

```python
import anthropic

# Connect to local server
client = anthropic.Anthropic(
    base_url="http://localhost:10240/anthropic",
    api_key="not-needed"
)

# Simple message completion
message = client.messages.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ]
)
print(message.content[0].text)
```

## 📋 Supported Endpoints

| Endpoint | Feature | Status |
|----------|---------|--------|
| `/anthropic/v1/messages` | Messages with tools, streaming, thinking mode | ✅ |
| `/anthropic/v1/models` | Model listing with pagination support | ✅ |

**Key Features:**
- ✅ Message completions with system prompts
- ✅ Real-time streaming responses
- ✅ Advanced tool calling
- ✅ Extended thinking mode
- ✅ Multiple content blocks

### Basic Message Completion

```python
message = client.messages.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms."
        }
    ]
)
print(message.content[0].text)
```

#### Message with System Prompt

```python
message = client.messages.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    max_tokens=1000,
    system="You are a helpful assistant that explains complex topics clearly.",
    messages=[
        {
            "role": "user",
            "content": "What is machine learning?"
        }
    ]
)
```

#### Streaming Messages

```python
with client.messages.stream(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Tell me a story about AI"}
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

#### Advanced Streaming with Event Handling

```python
with client.messages.stream(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Explain the solar system"}
    ]
) as stream:
    for event in stream:
        if event.type == "message_start":
            print(f"Message started: {event.message.id}")
        elif event.type == "content_block_delta":
            if hasattr(event.delta, 'text'):
                print(event.delta.text, end="", flush=True)
        elif event.type == "message_stop":
            print("\nMessage completed")
```

### Extended Thinking Mode

Enable the model's internal reasoning process:

```python
# Select a model that supports thinking mode
thinking_model = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

message = client.messages.create(
    model=thinking_model,
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[
        {
            "role": "user",
            "content": "Prove that there are infinitely many prime numbers."
        }
    ]
)

# Process thinking and response blocks
for block in message.content:
    if block.type == "thinking":
        print(f"💭 Thinking: {block.thinking}")
    elif block.type == "text":
        print(f"📝 Response: {block.text}")
```

#### Streaming with Thinking Mode

```python
with client.messages.stream(
    model=thinking_model,
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[
        {"role": "user", "content": "What is 27 * 453? Show your work."}
    ]
) as stream:
    for event in stream:
        if event.type == "content_block_start":
            if event.content_block.type == "thinking":
                print("💭 Thinking process started...")
            elif event.content_block.type == "text":
                print("📝 Response started...")
        elif event.type == "content_block_delta":
            if hasattr(event.delta, 'thinking'):
                print(f"💭 {event.delta.thinking}", end="")
            elif hasattr(event.delta, 'text'):
                print(f"📝 {event.delta.text}", end="", flush=True)
```

### Tool Calling

#### Basic Tool Usage

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email content"}
            },
            "required": ["to", "subject", "body"]
        }
    }
]

message = client.messages.create(
    model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    max_tokens=1024,
    tools=tools,
    messages=[
        {
            "role": "user",
            "content": "Check the weather in Tokyo and send an email to john@example.com about it."
        }
    ]
)

# Process tool calls and text response
for block in message.content:
    if block.type == "text":
        print(f"📝 {block.text}")
    elif block.type == "tool_use":
        print(f"🔧 Tool: {block.name}")
        print(f"   ID: {block.id}")
        print(f"   Parameters: {block.input}")
```

#### Streaming Tool Calls

```python
def stream_with_tools(user_message):
    """Stream messages with tool call monitoring"""

    with client.messages.stream(
        model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:

        text_content = ""
        current_tool = None
        tool_input_buffer = ""

        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "text":
                    print("🤖 Assistant: ", end="", flush=True)
                elif event.content_block.type == "tool_use":
                    current_tool = {
                        "name": event.content_block.name,
                        "id": event.content_block.id
                    }
                    tool_input_buffer = ""
                    print(f"\n🔧 Calling tool: {current_tool['name']}")

            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    text_content += event.delta.text
                    print(event.delta.text, end="", flush=True)
                elif event.delta.type == "input_json_delta":
                    tool_input_buffer += event.delta.partial_json
                    print(f"   📝 Building: {tool_input_buffer}", end="\r")

            elif event.type == "content_block_stop":
                if current_tool:
                    try:
                        import json
                        parsed_input = json.loads(tool_input_buffer)
                        print(f"   ✅ Parameters: {parsed_input}")
                    except json.JSONDecodeError:
                        print(f"   ❌ Invalid JSON: {tool_input_buffer}")
                    current_tool = None
                    tool_input_buffer = ""
                else:
                    print()  # End text line

# Usage
stream_with_tools("Get weather for London and send email to team@company.com")
```

### Models API (`/anthropic/v1/models`)

#### List Available Models

```python
# Get list of available models
models = client.models.list(limit=20)

print(f"Total models: {len(models.data)}")
for model in models.data:
    print(f"• {model.id} (Created: {model.created_at})")
```

#### Paginated Results

```python
# Get models with pagination
page1 = client.models.list(limit=10)
page2 = client.models.list(limit=10, after_id=page1.data[-1].id)

print(f"Page 1: {len(page1.data)} models")
print(f"Page 2: {len(page2.data)} models")
```

## 🔧 Advanced Usage

### Advanced Configuration

```python
message = client.messages.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    max_tokens=1000,
    temperature=0.7,           # Response randomness (0-1)
    top_p=0.9,                 # Nucleus sampling
    top_k=40,                  # Top-k sampling
    stop_sequences=["END"],    # Custom stop sequences
    system="You are a helpful assistant",
    messages=[
        {"role": "user", "content": "Generate creative content"}
    ]
)
```

### Tool Choice Control

```python
# Auto tool selection (default)
message = client.messages.create(
    model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    tools=tools,
    tool_choice={"type": "auto"},
    messages=[{"role": "user", "content": "What's the weather?"}]
)

# Force tool usage
message = client.messages.create(
    model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    tools=tools,
    tool_choice={"type": "any"},
    messages=[{"role": "user", "content": "Check weather somewhere"}]
)

# Disable tools
message = client.messages.create(
    model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    tools=tools,
    tool_choice={"type": "none"},
    messages=[{"role": "user", "content": "Just talk to me"}]
)

# Use specific tool
message = client.messages.create(
    model="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    tools=tools,
    tool_choice={"type": "tool", "name": "get_weather"},
    messages=[{"role": "user", "content": "I need weather information"}]
)
```

### Complex Content Blocks

```python
# Multi-part messages
message = client.messages.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this data and provide insights:"
                },
                {
                    "type": "text",
                    "text": "Sales increased by 25% in Q3, with the highest growth in the Asia-Pacific region."
                }
            ]
        }
    ]
)
```

### Metadata and Service Tier

```python
message = client.messages.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    max_tokens=1000,
    metadata={"user_id": "user123"},
    service_tier="auto",  # or "standard_only"
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

## 📡 REST API Examples

### Basic Message

```bash
curl -X POST "http://localhost:10240/anthropic/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: not-needed" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "max_tokens": 1000,
    "messages": [
      {
        "role": "user",
        "content": "Hello! How are you?"
      }
    ]
  }'
```

### Streaming Message

```bash
curl -X POST "http://localhost:10240/anthropic/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: not-needed" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "max_tokens": 1000,
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Tell me a joke"
      }
    ]
  }'
```

### Message with Tools

```bash
curl -X POST "http://localhost:10240/anthropic/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: not-needed" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    "max_tokens": 1000,
    "tools": [
      {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    ],
    "messages": [
      {
        "role": "user",
        "content": "What'\''s the weather in Tokyo?"
      }
    ]
  }'
```

### List Models

```bash
curl "http://localhost:10240/anthropic/v1/models" \
  -H "x-api-key: not-needed" \
  -H "anthropic-version: 2023-06-01"
```

### List Models with Pagination

```bash
curl "http://localhost:10240/anthropic/v1/models?limit=10&after_id=model_id" \
  -H "x-api-key: not-needed" \
  -H "anthropic-version: 2023-06-01"
```

## 🧪 Development & Testing

### Using TestClient

```python
import anthropic
from fastapi.testclient import TestClient
from mlx_batch_server.main import app

# Use TestClient for development
test_client = anthropic.Anthropic(
    base_url="http://testserver/anthropic",
    api_key="test-key",
    http_client=TestClient(app)
)

message = test_client.messages.create(
    model="mlx-community/gemma-3-1b-it-4bit-DWQ",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Error Handling

Failures use the Anthropic error envelope and carry a correlation id both in
the `request-id` response header and in the body:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "messages.0.content.0.type: Input should be 'text', 'image', ..."
  },
  "request_id": "req_2f1c8a0b4d9e7c6a5b3f1e2d"
}
```

Only documented Anthropic error types are emitted — `invalid_request_error`,
`authentication_error`, `permission_error`, `not_found_error`,
`request_too_large`, `rate_limit_error`, `timeout_error`, `api_error`,
`overloaded_error` — so `anthropic.APIError` subclasses resolve correctly. An
internal failure with no documented counterpart is reported as `api_error`
rather than as an invented type.

**Requests fail closed.** An unknown top-level field, an unsupported content
block, or a `tool_choice` naming a tool that was not declared is rejected with
`invalid_request_error`. Nothing is silently ignored: if the server answers,
it honoured what you sent.

When a failure happens after a stream has already opened, it arrives as a
terminal `error` event on the stream instead of an HTTP status:

```text
event: error  {"type":"error","error":{"type":"overloaded_error","message":"..."},"request_id":"req_..."}
```

```python
try:
    message = client.messages.create(
        model="non-existent-model",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Hello"}]
    )
except anthropic.APIError as e:
    print(f"API Error: {e} (request id: {e.request_id})")
```

## 📊 Response Structure

### Message Response

```python
# Example response structure
{
    "id": "msg_123456789",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Hello! I'\''m here to help you."
        }
    ],
    "model": "mlx-community/gemma-3-1b-it-4bit-DWQ",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 15,
        "output_tokens": 12
    }
}
```

### Streaming Events

A stream always opens with `message_start` and ends with `message_stop`. Every
content block is framed by an indexed `content_block_start` /
`content_block_stop` pair, and `message_delta` carries the stop reason together
with **cumulative** usage for the whole message.

```text
event: message_start        {"type":"message_start","message":{... ,"content":[]}}
event: content_block_start  {"type":"content_block_start","index":0,"content_block":{...}}
event: content_block_delta  {"type":"content_block_delta","index":0,"delta":{...}}
event: content_block_stop   {"type":"content_block_stop","index":0}
event: message_delta        {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{...}}
event: message_stop         {"type":"message_stop"}
```

Delta bodies are discriminated by their own `type`:

| Delta type | Block | Carries |
|---|---|---|
| `text_delta` | `text` | `text` — visible output |
| `thinking_delta` | `thinking` | `thinking` — reasoning, never duplicated into text |
| `signature_delta` | `thinking` | `signature` |
| `input_json_delta` | `tool_use` | `partial_json` — tool arguments |

Two events may appear at any point in the stream: `ping` (keep-alive, no
payload) and `error` (a terminal failure delivered inside an otherwise healthy
connection).

#### Tool call streaming

A `tool_use` block opens with an **empty** `input` object; the arguments arrive
only through `input_json_delta`. Concatenating every `partial_json` for one
block yields exactly the final arguments string — the payload is never
replayed, so a client must accumulate rather than overwrite.

```text
event: content_block_start  {"index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}
event: content_block_delta  {"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\": "}}
event: content_block_delta  {"index":1,"delta":{"type":"input_json_delta","partial_json":"\"Kielce\"}"}}
event: content_block_stop   {"index":1}
```

#### Stop reasons

`stop_reason` is mapped intentionally, not guessed. Truncation outranks tool
use: a tool call cut off mid-arguments reports `max_tokens`, not `tool_use`.

| Reason | Meaning |
|---|---|
| `end_turn` | The model finished on its own |
| `max_tokens` | Output hit the requested `max_tokens` budget |
| `stop_sequence` | A configured stop sequence matched |
| `tool_use` | The turn ended asking for a tool |
| `refusal` / `pause_turn` | Passed through when the runtime reports them |

The `max_tokens` you send is honoured as-is; this server introduces no
additional client-side token ceiling.

## 🔍 Troubleshooting

### Common Issues

**Model Not Found**
```bash
# Check available models
curl "http://localhost:10240/anthropic/v1/models"

# Pre-download models
huggingface-cli download mlx-community/gemma-3-1b-it-4bit-DWQ
```

**Streaming Issues**
```bash
# Check server logs
MLX_OMNI_LOG_LEVEL=debug mlx-batch-server

# Test with simple non-streaming request first
```

**Tool Calling Problems**
```bash
# Verify tool schema is valid JSON
# Check model supports tool calling
# Use models like: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
```

### Debug Mode

```bash
# Enable debug logging
MLX_OMNI_LOG_LEVEL=debug mlx-batch-server

# Test with curl for detailed error messages
curl -v "http://localhost:10240/anthropic/v1/messages" \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "max_tokens": 100, "messages": [{"role": "user", "content": "test"}]}'
```

## 📚 API Reference

For complete Anthropic API specifications, see:
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [Anthropic Models API](https://docs.anthropic.com/en/api/models)
- [MLX Batch Server Source Code](https://github.com/LibraxisAI/mlx-batch-server) (fork of [madroidmaq/mlx-omni-server](https://github.com/madroidmaq/mlx-omni-server))

## 🤝 Contributing

Contributions are welcome! Please see the main repository for guidelines on:
- Setting up development environment
- Running tests
- Submitting pull requests

---

**Note**: This documentation covers Anthropic API compatibility. For OpenAI API documentation, see [docs/openai-api.md](../openai-api.md).
