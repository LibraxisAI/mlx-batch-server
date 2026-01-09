<div align="center">

# MLX Omni Server

*Local AI inference server optimized for Apple Silicon*

[![PyPI version](https://img.shields.io/pypi/v/mlx-omni-server.svg)](https://pypi.python.org/pypi/mlx-omni-server)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/madroidmaq/mlx-omni-server)

![MLX Omni Server Banner](docs/banner.png)

**MLX Omni Server** provides dual API compatibility with both **OpenAI** and **Anthropic APIs**, enabling seamless local inference on Apple Silicon using the MLX framework.

[Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## LibraxisAI Fork Enhancements

This fork by [LibraxisAI](https://github.com/LibraxisAI) adds production-grade features:

| Feature | Description |
|---------|-------------|
| **[Responses API](docs/responses/)** | Full OpenAI `/v1/responses` endpoint with SSE streaming |
| **[Batch Processing](docs/batch/)** | Concurrent request batching via mlx-lm BatchGenerator |
| **[Harmony Parser](docs/responses/harmony.md)** | Streaming parser for GPT-OSS models (OpenAI's open-source release) |
| **Model Management** | Load/unload endpoints for dynamic model switching |
| **Enhanced Config** | Environment-based configuration with batch settings |

### Key Additions

```
/v1/responses          - OpenAI Responses API with streaming
/v1/batch/stats        - Batch coordinator statistics
/v1/models/load        - Dynamic model loading
/v1/models/unload      - Model unloading
```

---

## Features

- Apple Silicon Optimized - Built on MLX framework for M1/M2/M3/M4 chips
- Dual API Support - Compatible with both OpenAI and Anthropic APIs
- Complete AI Suite - Chat, audio processing, image generation, embeddings
- Batch Inference - Handle 10+ concurrent requests efficiently
- Harmony Support - Native GPT-OSS model support with channel parsing
- Privacy-First - All processing happens locally on your machine
- Drop-in Replacement - Works with existing OpenAI and Anthropic SDKs

## Installation

```bash
pip install mlx-omni-server
```

Or from source (LibraxisAI fork):

```bash
git clone https://github.com/LibraxisAI/mlx-omni-server.git
cd mlx-omni-server
uv sync
```

## Quick Start

1. **Start the server:**
   ```bash
   mlx-omni-server
   ```

2. **Choose your preferred API:**

   <details>
   <summary><b>OpenAI Responses API</b> (Recommended)</summary>

   ```python
   import httpx

   response = httpx.post(
       "http://localhost:10240/v1/responses",
       json={
           "model": "mlx-community/Qwen3-0.6B-4bit",
           "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
           "stream": True
       },
       headers={"Content-Type": "application/json"}
   )

   for line in response.iter_lines():
       if line.startswith("data: "):
           print(line[6:])
   ```
   </details>

   <details>
   <summary><b>OpenAI Chat Completions API</b></summary>

   ```python
   from openai import OpenAI

   client = OpenAI(
       base_url="http://localhost:10240/v1",
       api_key="not-needed"
   )

   response = client.chat.completions.create(
       model="mlx-community/gemma-3-1b-it-4bit-DWQ",
       messages=[{"role": "user", "content": "Hello!"}]
   )
   print(response.choices[0].message.content)
   ```
   </details>

   <details>
   <summary><b>Anthropic API</b></summary>

   ```python
   import anthropic

   client = anthropic.Anthropic(
       base_url="http://localhost:10240/anthropic",
       api_key="not-needed"
   )

   message = client.messages.create(
       model="mlx-community/gemma-3-1b-it-4bit-DWQ",
       max_tokens=1000,
       messages=[{"role": "user", "content": "Hello!"}]
   )
   print(message.content[0].text)
   ```
   </details>

## API Support

### OpenAI Compatible Endpoints (`/v1/*`)

| Endpoint | Feature | Status |
|----------|---------|--------|
| `/v1/responses` | **Responses API with SSE streaming** | **NEW** |
| `/v1/chat/completions` | Chat with tools, streaming, structured output | Stable |
| `/v1/batch/stats` | Batch coordinator statistics | **NEW** |
| `/v1/models/load` | Dynamic model loading | **NEW** |
| `/v1/models/unload` | Model unloading | **NEW** |
| `/v1/audio/speech` | Text-to-Speech | Stable |
| `/v1/audio/transcriptions` | Speech-to-Text | Stable |
| `/v1/images/generations` | Image Generation | Stable |
| `/v1/embeddings` | Text Embeddings | Stable |
| `/v1/models` | Model Management | Stable |

### Anthropic Compatible Endpoints (`/anthropic/v1/*`)

| Endpoint | Feature | Status |
|----------|---------|--------|
| `/anthropic/v1/messages` | Messages with tools, streaming, thinking mode | Stable |
| `/anthropic/v1/models` | Model listing with pagination | Stable |


## Configuration

```bash
# Default (port 10240)
mlx-omni-server

# Custom options
mlx-omni-server --port 8000
MLX_OMNI_LOG_LEVEL=debug mlx-omni-server

# View all options
mlx-omni-server --help
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLX_OMNI_LOG_LEVEL` | Logging level | `info` |
| `MLX_OMNI_CORS` | CORS origins (comma-separated) | - |
| `MLX_OMNI_ENABLE_BATCH` | Enable batch inference | `true` |
| `MLX_OMNI_BATCH_WINDOW_MS` | Batch collection window | `50` |
| `MLX_OMNI_MAX_BATCH_SIZE` | Maximum batch size | `10` |

## Documentation

| Resource | Description |
|----------|-------------|
| [Responses API Guide](docs/responses/) | Full Responses API reference |
| [Batch Processing Guide](docs/batch/) | Batch inference configuration |
| [OpenAI API Guide](docs/openai-api.md) | OpenAI API reference |
| [Anthropic API Guide](docs/anthropic-api.md) | Anthropic API reference |
| [Examples](examples/) | Practical usage examples |

## Development

<details>
<summary><b>Development Setup</b></summary>

```bash
git clone https://github.com/LibraxisAI/mlx-omni-server.git
cd mlx-omni-server
make setup  # Install deps + hooks

# Start with hot-reload
make dev

# Or with custom port
make dev PORT=8100
```

**Testing:**
```bash
make test              # All tests
make test-responses    # Responses API tests
make test-fast         # Skip slow tests
```

**Code Quality:**
```bash
make lint              # Run linters
make format            # Format code
make check             # All checks (CI simulation)
```

**Model Management:**
```bash
make load MODEL=mlx-community/Qwen3-0.6B-4bit
make unload
make ps                # List loaded models
make batch-stats       # Batch coordinator stats
```
</details>

## Troubleshooting

<details>
<summary><b>Common Issues</b></summary>

**Requirements:**
- Python 3.11+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX framework installed

**Quick fixes:**
```bash
# Check requirements
python --version  # Should be 3.11+
python -c "import mlx; print(mlx.__version__)"

# Pre-download models (if needed)
huggingface-cli download mlx-community/gemma-3-1b-it-4bit-DWQ

# Enable debug logging
MLX_OMNI_LOG_LEVEL=debug mlx-omni-server
```
</details>

## Contributing

**Quick contributor setup:**
```bash
git clone https://github.com/LibraxisAI/mlx-omni-server.git
cd mlx-omni-server
make setup && make test
```

<div align="center">

---

## Acknowledgments

Built with [MLX](https://github.com/ml-explore/mlx) by Apple | [FastAPI](https://fastapi.tiangolo.com/) | [MLX-LM](https://github.com/ml-explore/mlx-lm)

Original project by [madroidmaq](https://github.com/madroidmaq/mlx-omni-server)

LibraxisAI enhancements by M&K (c)2026 VetCoders

## License

[MIT License](LICENSE) | Not affiliated with OpenAI, Anthropic, or Apple

</div>
