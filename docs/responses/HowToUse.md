# How to Use Responses API

## Basic Streaming Request

```bash
curl -sS -N http://localhost:10240/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "Explain quantum computing briefly."}]}],
    "stream": true
  }'
```

## Non-Streaming Request

```bash
curl http://localhost:10240/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "What is 2+2?"}]}],
    "stream": false
  }'
```

Response:
```json
{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1704067200,
  "status": "completed",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {"type": "output_text", "text": "2+2 equals 4."}
      ]
    }
  ]
}
```

## With System Prompt

```bash
curl -sS -N http://localhost:10240/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "input": [
      {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
      {"role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}
    ],
    "stream": true
  }'
```

## Python with httpx

```python
import httpx
import json

def stream_response(prompt: str, model: str = "mlx-community/Qwen3-0.6B-4bit"):
    """Stream a response from the API."""
    with httpx.stream(
        "POST",
        "http://localhost:10240/v1/responses",
        json={
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "stream": True
        },
        timeout=60.0
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                event = json.loads(line[6:])
                if event.get("type") == "response.output_text.delta":
                    print(event.get("delta", ""), end="", flush=True)
        print()

if __name__ == "__main__":
    stream_response("Tell me a joke about programming")
```

## Python with openai SDK

The openai SDK can work with our endpoint using custom base_url:

```python
from openai import OpenAI

# Note: Using chat completions endpoint, not responses
client = OpenAI(
    base_url="http://localhost:10240/v1",
    api_key="not-needed"
)

# For responses API, use httpx directly (see above)
# The openai SDK is for /v1/chat/completions compatibility
```

## JavaScript/Fetch

```javascript
async function streamResponse(prompt) {
  const response = await fetch('http://localhost:10240/v1/responses', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'mlx-community/Qwen3-0.6B-4bit',
      input: [{ role: 'user', content: [{ type: 'input_text', text: prompt }] }],
      stream: true,
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ') && line !== 'data: [DONE]') {
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === 'response.output_text.delta') {
            process.stdout.write(event.delta || '');
          }
        } catch (e) {
          // Skip non-JSON lines
        }
      }
    }
  }
}

streamResponse('What is the meaning of life?');
```

## Multi-turn Conversation

```python
import httpx
import json

def chat_turn(prompt: str, previous_id: str = None) -> tuple[str, str]:
    """Send a chat turn and return (text, response_id)."""
    payload = {
        "model": "mlx-community/Qwen3-0.6B-4bit",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "stream": False
    }
    if previous_id:
        payload["previous_response_id"] = previous_id

    response = httpx.post(
        "http://localhost:10240/v1/responses",
        json=payload,
        timeout=60.0
    ).json()

    text = ""
    for item in response.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text += content.get("text", "")

    return text, response.get("id")

# Multi-turn conversation
text, resp_id = chat_turn("My name is Alex")
print(f"Assistant: {text}")

text, resp_id = chat_turn("What is my name?", resp_id)
print(f"Assistant: {text}")  # Should mention "Alex"
```

## Error Handling

```python
import httpx

try:
    response = httpx.post(
        "http://localhost:10240/v1/responses",
        json={
            "model": "nonexistent-model",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]
        },
        timeout=60.0
    )
    response.raise_for_status()
except httpx.HTTPStatusError as e:
    print(f"HTTP Error: {e.response.status_code}")
    print(f"Detail: {e.response.json()}")
except httpx.RequestError as e:
    print(f"Request Error: {e}")
```

---
Vibecrafted. with AI Agents by VetCoders (c)2024-2026 The LibraxisAI Team
