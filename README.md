<div align="center">

# MLX Batch Server

*High-performance local AI inference server for Apple Silicon with batch processing*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1--M4_Ultra-000000?logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![MLX](https://img.shields.io/badge/MLX-%E2%89%A50.30-FF6F00)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Responses API](https://img.shields.io/badge/Responses_API-native-6366f1)]()
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI_API-compatible-412991?logo=openai&logoColor=white)]()
[![Anthropic Compatible](https://img.shields.io/badge/Anthropic_API-compatible-D97757?logo=anthropic&logoColor=white)]()

**MLX Batch Server** is a production-grade inference server optimized for Apple Silicon, featuring concurrent batch
processing, OpenAI Responses API, and Harmony parser for GPT-OSS models.

[Features](#-features) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Configuration](#-configuration)

</div>

---

## Origin & Acknowledgments

This project is a **standalone fork** of [mlx-batch-server](https://github.com/madroidmaq/mlx-batch-server) by **[@madroidmaq](https://github.com/madroidmaq)**, whose excellent work laid the foundation for local MLX inference with OpenAI/Anthropic API compatibility.

**VetCoders** extended the original project with:

- Batch inference coordinator (10+ concurrent requests)
- Full OpenAI Responses API (`/v1/responses`)
- Streaming Harmony parser for GPT-OSS models
- Production hardening for 24/7 operation

We maintain this as a separate project due to significant architectural divergence, while continuing to contribute
improvements back to the upstream project where applicable.

---

## Features

| Feature | Description |
|---------|-------------|
| **Batch Processing** | Handle 10+ concurrent requests via mlx-lm BatchGenerator |
| **Responses API** | Full OpenAI `/v1/responses` with SSE streaming |
| **Harmony Parser** | Native GPT-OSS model support with channel parsing |
| **Dual API** | Compatible with OpenAI and Anthropic SDKs |
| **Model Management** | Dynamic load/unload endpoints |
| **Privacy-First** | All processing happens locally on your Mac |

### What's Different From Upstream

```text
├── Batch Coordinator      → Concurrent request batching (NEW)
├── /v1/responses          → OpenAI Responses API (NEW)
├── Harmony Streaming      → GPT-OSS channel parser (NEW)
├── /v1/models/load        → Dynamic model loading (NEW)
├── /v1/models/unload      → Model unloading (NEW)
└── Production Config      → Environment-based settings (NEW)
```

---

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/VetCoders/mlx-batch-server.git
cd mlx-batch-server

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

Local development uses the editable sibling dependency `../mlx-vlm-local`, so upstream-facing `mlx-vlm` fixes land in the server immediately after `uv sync`.

### Run the Server

```bash
# Default (port 10240)
mlx-batch-server

# Custom port
mlx-batch-server --port 10240

# With debug logging
MLX_BATCH_LOG_LEVEL=debug mlx-batch-server
```

### Test It

```bash
# Chat completion
curl http://localhost:10240/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Responses API (streaming)
curl http://localhost:10240/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
    "stream": true
  }'
```

### Preparing Qwen3.6-VL-30B

The local `mlx-vlm` dependency already understands `qwen3.6-vl` and `qwen3.6-vl-moe` aliases during conversion. To expose a converted `qwen3.6-vl-30b` cleanly through the server, point `MODEL_ALIASES` or an in-process runtime alias at the converted model path or repo id.

---

## API Reference

### OpenAI Compatible (`/v1/*`)

| Endpoint | Description | Status |
|----------|-------------|--------|
| `POST /v1/responses` | Responses API with SSE streaming | Stable |
| `POST /v1/chat/completions` | Chat with tools, streaming, structured output | Stable |
| `GET /v1/batch/stats` | Batch coordinator statistics | Stable |
| `POST /v1/models/load` | Dynamic model loading | Stable |
| `POST /v1/models/unload` | Model unloading | Stable |
| `POST /v1/audio/speech` | Text-to-Speech | Stable |
| `POST /v1/audio/transcriptions` | Speech-to-Text (Whisper) | Stable |
| `POST /v1/images/generations` | Image Generation | Stable |
| `POST /v1/embeddings` | Text Embeddings | Stable |
| `GET /v1/models` | List available models | Stable |

### Anthropic Compatible (`/anthropic/v1/*`)

| Endpoint | Description | Status |
|----------|-------------|--------|
| `POST /anthropic/v1/messages` | Messages with tools, streaming, thinking | Stable |
| `GET /anthropic/v1/models` | Model listing with pagination | Stable |

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLX_BATCH_LOG_LEVEL` | Logging level (`debug`, `info`, `warning`) | `info` |
| `MLX_BATCH_CORS` | CORS origins (comma-separated) | `*` |
| `MLX_BATCH_ENABLE_BATCH` | Enable batch inference | `true` |
| `MLX_BATCH_BATCH_WINDOW_MS` | Batch collection window (ms) | `50` |
| `MLX_BATCH_MAX_BATCH_SIZE` | Maximum concurrent requests | `10` |
| `MLX_BATCH_DEFAULT_MODEL` | Model to load on startup | - |

### Batch Processing

Batch processing collects incoming requests within a time window and processes them together, significantly improving
throughput on Apple Silicon:

```bash
# Tune for your workload
MLX_BATCH_BATCH_WINDOW_MS=100 \
MLX_BATCH_MAX_BATCH_SIZE=16 \
mlx-batch-server
```

**Performance (M3 Ultra, 512GB):**

- Single request: ~50 tok/s
- Batched (10 requests): ~35 tok/s per request = **350 tok/s total**

---

## Development

```bash
# Setup
make setup           # Install deps + pre-commit hooks

# Run
make dev             # Start with hot-reload
make dev PORT=10240  # Custom port

# Test
make test            # All tests
make test-responses  # Responses API tests
make test-fast       # Skip slow tests

# Quality
make lint            # Run linters
make format          # Format code
make check           # Full CI check

# Model management
make load MODEL=mlx-community/Qwen3-0.6B-4bit
make unload
make ps              # List loaded models
make batch-stats     # Coordinator stats
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [Responses API Guide](docs/responses/) | Full Responses API reference |
| [Batch Processing Guide](docs/batch/) | Batch inference configuration |
| [Harmony Parser](docs/responses/harmony.md) | GPT-OSS channel parsing |
| [OpenAI API Guide](docs/openai-api.md) | OpenAI compatibility reference |
| [Anthropic API Guide](docs/anthropic-api.md) | Anthropic compatibility reference |
| [Examples](examples/) | Practical usage examples |

---

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.11+**
- **MLX framework** (auto-installed)

---

## Contributing

```bash
git clone https://github.com/VetCoders/mlx-batch-server.git
cd mlx-batch-server
make setup && make test
```

Pull requests welcome! For major changes, please open an issue first.

---

<div align="center">

## License

[MIT License](LICENSE)

---

**Original project:** [mlx-batch-server](https://github.com/madroidmaq/mlx-batch-server) by [@madroidmaq](https://github.com/madroidmaq)

**Fork maintained by:** [VetCoders](https://github.com/VetCoders) — M&K (c)2026

Built with [MLX](https://github.com/ml-explore/mlx) by Apple • [FastAPI](https://fastapi.tiangolo.com/) • [MLX-LM](https://github.com/ml-explore/mlx-lm)

*Not affiliated with OpenAI, Anthropic, or Apple*

</div>
