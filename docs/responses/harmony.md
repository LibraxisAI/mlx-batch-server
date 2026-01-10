# Harmony Format Support

MLX Batch Server includes a streaming parser for OpenAI's GPT-OSS models that
use the Harmony response format.

## What is Harmony?

Harmony is OpenAI's format for GPT-OSS models (like gpt-oss-120b). It uses
special tokens to structure responses into channels:

- `<|channel|>analysis` - Reasoning/thinking channel
- `<|channel|>final` - Final answer channel
- `<|message|>` - Message content marker
- `<|call|>` - Tool call marker

## Automatic Parsing

When using GPT-OSS models, the server automatically:

1. Detects Harmony tokens in streaming output
2. Strips tokens from user-visible deltas
3. Routes content to appropriate SSE event types
4. Parses tool calls from Harmony format

## SSE Events for Harmony Models

| Event Type | Harmony Channel |
|------------|-----------------|
| `response.reasoning_text.delta` | `analysis` channel |
| `response.output_text.delta` | `final` channel |

## Example

Raw model output:
```
<|channel|>analysis<|message|>Let me think about this...
<|channel|>final<|message|>The answer is 42.
```

Streamed to client as:
```
event: response.reasoning_text.delta
data: {"type":"response.reasoning_text.delta","delta":"Let me think about this..."}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"The answer is 42."}
```

## Streaming Parser

The `HarmonyStreamingParser` handles edge cases:

- Fragmented tokens across chunk boundaries
- Channel name arriving in separate chunk from `<|channel|>`
- Tool call reconstruction
- Commentary channel filtering

## Usage

No special configuration needed - Harmony parsing is automatic when:
1. Model ID contains "gpt-oss" or is aliased to a GPT-OSS model
2. Using `/v1/responses` endpoint with streaming

```bash
curl -sS -N http://localhost:8100/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "Explain relativity"}]}],
    "stream": true
  }'
```

## Alternative: openai-harmony Package

OpenAI provides the `openai-harmony` package as a reference parser:

```bash
pip install openai-harmony
```

Our custom parser is preferred because:
- Battle-tested with MLX streaming infrastructure
- Handles fragmented token edge cases
- Integrated with SSE event generation

Both approaches are equivalent per GPT-OSS README: "use chat template OR
openai-harmony package".

## Disabling Harmony Parsing

If you need raw Harmony tokens (for debugging):

```python
# Use chat completions endpoint instead of responses
# Harmony parsing only applies to /v1/responses
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-120b","messages":[...]}'
```

---
Created by M&K (c)2026 VetCoders
