# How to Use Batch Processing

## Configuration

Batch processing is enabled by default. Configure via environment variables:

```bash
# Enable/disable batch inference
export MLX_BATCH_ENABLE_BATCH=true

# Time window (ms) to collect requests before processing
export MLX_BATCH_BATCH_WINDOW_MS=50

# Maximum requests per batch
export MLX_BATCH_MAX_BATCH_SIZE=10

# mlx-lm BatchGenerator settings
export MLX_BATCH_BATCH_COMPLETION_SIZE=32
export MLX_BATCH_BATCH_PREFILL_SIZE=8
export MLX_BATCH_BATCH_PREFILL_STEP_SIZE=2048
```

## Health Check

```bash
curl http://localhost:8100/v1/batch/stats | jq
```

Response:
```json
{
  "enabled": true,
  "settings": {
    "batch_window_ms": 50,
    "max_batch_size": 10,
    "completion_batch_size": 32,
    "prefill_batch_size": 8
  },
  "coordinators": {}
}
```

## Streaming Example

```bash
# Single streaming request (will use batch infrastructure)
curl -sS -N http://localhost:8100/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
    "stream": true
  }'
```

## Concurrent Requests with curl

```bash
# Launch 5 concurrent requests
for i in {1..5}; do
  curl -sS -N http://localhost:8100/v1/responses \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"mlx-community/Qwen3-0.6B-4bit\",
      \"input\": [{\"role\": \"user\", \"content\": [{\"type\": \"input_text\", \"text\": \"Count from $i to 10\"}]}],
      \"stream\": true
    }" &
done
wait
```

## Python Example

```python
import asyncio
import httpx

async def send_request(client: httpx.AsyncClient, prompt: str):
    """Send streaming request and collect response."""
    async with client.stream(
        "POST",
        "http://localhost:8100/v1/responses",
        json={
            "model": "mlx-community/Qwen3-0.6B-4bit",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "stream": True
        }
    ) as response:
        text = ""
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                import json
                event = json.loads(line[6:])
                if event.get("type") == "response.output_text.delta":
                    text += event.get("delta", "")
        return text

async def test_concurrent():
    """Test concurrent requests."""
    prompts = [
        "What is 2+2?",
        "Name a color",
        "What day is today?",
        "Say hello in Spanish",
        "Count to 5"
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = [send_request(client, p) for p in prompts]
        results = await asyncio.gather(*tasks)

    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}")
        print(f"A: {result[:100]}...")
        print()

if __name__ == "__main__":
    asyncio.run(test_concurrent())
```

## Interactive Tester

Open `api-tester.html` in your browser for an interactive multi-lane
comparison tool. Features:

- Multiple concurrent lanes
- Chain length support (multi-turn conversations)
- Streaming/non-streaming toggle
- Response time metrics (TTFT, tok/s)
- Export results to JSON

## Troubleshooting

### Batch not engaging

Check that:
1. `enable_batch_inference` is `true` in config
2. Using `/v1/responses` endpoint (not `/v1/chat/completions`)
3. Multiple concurrent requests are active

### Low throughput

Try adjusting:
```bash
# Larger completion batch for more parallelism
export MLX_BATCH_BATCH_COMPLETION_SIZE=64

# Smaller batch window for faster response
export MLX_BATCH_BATCH_WINDOW_MS=25
```

### Memory issues

Reduce batch sizes:
```bash
export MLX_BATCH_BATCH_COMPLETION_SIZE=16
export MLX_BATCH_BATCH_PREFILL_SIZE=4
export MLX_BATCH_MAX_BATCH_SIZE=5
```

## Monitoring

Watch batch coordinator activity:

```bash
# In one terminal - watch stats
watch -n 1 'curl -s http://localhost:8100/v1/batch/stats | jq'

# In another - send concurrent requests
./test_concurrent.py
```

---
Created by M&K (c)2026 VetCoders
