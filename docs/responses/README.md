# Responses API

MLX Batch Server implements OpenAI's `/v1/responses` API endpoint with full
SSE streaming support. This directory contains documentation for the
Responses API subsystem.

- `HowToUse.md` - Usage examples and curl commands
- `harmony.md` - GPT-OSS Harmony format support

## Overview

The Responses API is OpenAI's newer API format, designed for:
- Server-sent events (SSE) streaming
- Multi-turn conversation chaining via `previous_response_id`
- Structured input format with content types

## Quick Start

```bash
# Basic request
curl -X POST http://localhost:10240/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
    "stream": true
  }'
```

## Input Format

```json
{
  "model": "model-id",
  "input": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Hello!"},
        {"type": "input_image", "image_url": "https://..."}
      ]
    }
  ],
  "stream": true,
  "previous_response_id": "resp_abc123"
}
```

## SSE Events

Streaming responses emit the following event types:

| Event Type | Description |
|------------|-------------|
| `response.created` | Initial response object |
| `response.in_progress` | Status update |
| `response.output_text.delta` | Text chunk |
| `response.output_text.done` | Final text |
| `response.completed` | Response complete |

Example stream:
```
event: response.created
data: {"type":"response.created","response":{"id":"resp_xxx","status":"in_progress"}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"!"}

event: response.output_text.done
data: {"type":"response.output_text.done","text":"Hello!"}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_xxx","status":"completed"}}

data: [DONE]
```

## Conversation Chaining

Use `previous_response_id` for multi-turn conversations:

```bash
# First request
RESP_ID=$(curl -s http://localhost:10240/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"chat","input":[{"role":"user","content":[{"type":"input_text","text":"My name is Alex"}]}]}' \
  | jq -r '.id')

# Follow-up (uses conversation history)
curl http://localhost:10240/v1/responses \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"chat\",
    \"input\": [{\"role\": \"user\", \"content\": [{\"type\": \"input_text\", \"text\": \"What is my name?\"}]}],
    \"previous_response_id\": \"$RESP_ID\"
  }"
```

Refer to `HowToUse.md` for more examples and `harmony.md` for GPT-OSS model support.

---
Vibecrafted with AI Agents by VetCoders (c)2026 VetCoders
