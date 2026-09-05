<div align="center">

# MLX Batch Server

*Local OpenAI-compatible Responses inference owner for Apple Silicon*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1--M4_Ultra-000000?logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![MLX](https://img.shields.io/badge/MLX-%E2%89%A50.30-FF6F00)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Responses API](https://img.shields.io/badge/Responses_API-native-6366f1)]()
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI_API-compatible-412991?logo=openai&logoColor=white)]()
[![Anthropic Compatible](https://img.shields.io/badge/Anthropic_API-compatible-D97757?logo=anthropic&logoColor=white)]()

**MLX Batch Server** is the LibraxisAI inference owner. `/v1/responses` is the primary
surface. Production role `main` on port **8100** runs fused Qwen Flash (`fused_mtp_mlx`)
with native HTTP/SSE and multiplexed WebSocket, tools, vision, and MTP. Tensor execution
is honestly **row-serial** today: health reports `tensor_batch_mode=row_serial` and
`text.batch_capable=false`. The product name is historical; do not read it as true
multi-row tensor batching.

[Features](#-features) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Configuration](#-configuration)

</div>

---

## Origin & Acknowledgments

This project is a **standalone fork** of [mlx-batch-server](https://github.com/madroidmaq/mlx-batch-server) by **[@madroidmaq](https://github.com/madroidmaq)**, whose excellent work laid the foundation for local MLX inference with OpenAI/Anthropic API compatibility.

**VetCoders / LibraxisAI** extended the original project with:

- Native OpenAI `/v1/responses` as the primary product surface
- Fused Qwen Flash runtime (`fused_mtp_mlx`) with signed 8100–8102 roles
- Streaming Harmony parser for GPT-OSS models
- Legacy mlx-lm batch coordinator kept as a compatibility lane, not the Flash product

We maintain this as a separate project due to significant architectural divergence, while continuing to contribute
improvements back to the upstream project where applicable.

---

## Features

| Feature | Description |
|---------|-------------|
| **Responses API** | Native `/v1/responses` with SSE and multiplexed WebSocket |
| **Fused Flash runtime** | `fused_mtp_mlx` on signed role `main`/`8100`: text, vision, tools, MTP |
| **Honest capabilities** | Fused Flash reports row-serial tensors; no true multi-row claim |
| **Signed roles** | `8100` main, `8101` canary, `8102` vision (`legacy_mlx`) |
| **Legacy mlx-lm batch** | Compatibility lane for unsupported models, not the Flash product |
| **Harmony Parser** | Native GPT-OSS model support with channel parsing |
| **Dual API** | Compatible with OpenAI and Anthropic SDKs |
| **Model Management** | Dynamic load/unload/alias plus `/admin` |
| **Privacy-First** | All processing happens locally on your Mac |

### What's Different From Upstream

```text
├── /v1/responses          → Primary OpenAI Responses surface (HTTP/SSE/WebSocket)
├── Fused Flash backend    → fused_mtp_mlx owns Qwen Flash text/vision/tools/MTP
├── Signed role topology   → 8100-8102 fail-closed manifest, not silent rebind
├── Honest row-serial      → scheduler admits 8 / plans 4 decode rows; QSA is B=1
├── Legacy batch lane      → mlx-lm BatchGenerator for unsupported models only
├── Harmony Streaming      → GPT-OSS channel parser
├── /v1/models/load        → Dynamic model loading
└── /v1/models/unload      → Model unloading
```

---

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/LibraxisAI/mlx-batch-server.git
cd mlx-batch-server

# Core install (inference only)
uv sync
# Or
pip install -e .

# Full surface (auth + operator UI)
uv sync --extra auth --extra operator
# Or
pip install -e ".[auth,operator]"
```

| Extra      | Pulls                          | Enables |
|------------|--------------------------------|---------|
| `auth`     | `redis`, `pyjwt`               | Session auth + Redis-backed API keys/HMAC + rate limiting |
| `operator` | `click`, `jinja2`, `python-multipart`, `ruamel.yaml` | `mlx-batch-operator` CLI + htmx admin UI |

Local development uses the editable sibling dependency `../mlx-vlm-local`, so upstream-facing `mlx-vlm` fixes land in the server immediately after `uv sync`.

### Run the Server

Unbound local bind still defaults to **10240**. That is a developer bind, not the
production product. Libraxis production uses the signed role manifest:
`main`/`8100` fused Flash, `canary`/`8101`, `vision`/`8102`.

```bash
# Local unbound bind (developer default)
mlx-batch-server

# Production role main (fused Flash Responses owner)
MLX_BATCH_RUNTIME_ROLE=main mlx-batch-server --port 8100

# With debug logging
MLX_BATCH_LOG_LEVEL=debug mlx-batch-server
```

### Test It

```bash
# Production Responses owner (role main / 8100)
curl http://127.0.0.1:8100/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "Reply with exactly OK"}]}],
    "reasoning": {"effort": "none"}
  }'

# Local unbound developer bind (10240) for a small mlx-lm model
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
| `POST /v1/videos/generations` | Isolated local LTX image-to-video | Preview |
| `GET /v1/videos/capabilities` | Video adapter availability without model wake | Preview |
| `POST /v1/embeddings` | Text Embeddings | Stable |
| `GET /v1/models` | List available models | Stable |

### Anthropic Compatible (`/anthropic/v1/*`)

| Endpoint | Description | Status |
|----------|-------------|--------|
| `POST /anthropic/v1/messages` | Bounded Messages profile: text/tools, SSE, admitted rich inputs | Stable |
| `GET /anthropic/v1/models` | Model listing with pagination | Stable |

The admitted client oracle is exactly `anthropic==0.96.0`, exercised through
sync and async non-stream plus sync and async SSE clients. Raw HTTP covers the
extended request-field matrix. This local runtime accepts service tiers `auto`
and `standard_only` and reports the actual delivered tier as `standard`.

Supported rich content includes admitted image and document source forms plus
caller-supplied `search_result` blocks. Hosted web/search execution, containers,
prompt caching, citations, structured output, priority tier, enabled extended
thinking, prior-thinking continuation, and redacted-thinking continuation are
explicitly refused with HTTP 400 before streaming or inference starts. See the
[exact compatibility and refusal matrix](docs/anthropic-api.md).

Run the public admission verifier against a separately started final candidate
on a caller-selected non-production port (for example `10241`):

```bash
uv run --with 'anthropic==0.96.0' \
  python scripts/quality/verify_live_mlx_batch_api.py \
  --base-url http://127.0.0.1:10241 \
  --model "$MLX_BATCH_ADMISSION_MODEL" \
  --api-key-env MLX_BATCH_ADMISSION_KEY \
  --anthropic-api-key-env MLX_BATCH_ADMISSION_KEY \
  --public-url "$MLX_BATCH_PUBLIC_FIXTURE_URL" \
  --private-redirect-url "$MLX_BATCH_PRIVATE_REDIRECT_URL" \
  --receipt artifacts/anthropic-public-admission.json
```

Never use this admission run to replace or mutate the services on `8100` or
`10240`.

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLX_BATCH_RUNTIME_ROLE` | Signed role (`main`, `canary`, `vision`) | unbound |
| `MLX_BATCH_LOG_LEVEL` | Logging level (`debug`, `info`, `warning`) | `info` |
| `MLX_BATCH_CORS` | CORS origins (comma-separated) | `*` |
| `MLX_BATCH_ENABLE_BATCH` | Enable **legacy mlx-lm** batch coordinator (not fused Flash) | `true` |
| `MLX_BATCH_BATCH_WINDOW_MS` | Legacy mlx-lm batch collection window (ms) | `50` |
| `MLX_BATCH_MAX_BATCH_SIZE` | Legacy mlx-lm maximum concurrent requests | `10` |
| `MLX_BATCH_DEFAULT_MODEL` | Model to load on startup | - |

### Legacy mlx-lm batch lane

The mlx-lm `BatchRequestCoordinator` still exists as a **compatibility lane** for
unsupported models. Fused Flash on 8100 does **not** use it. That role admits up
to 8 requests and may plan 4 decode rows, but QSA executes `B=1` row-serial and
`/health` reports `text.batch_capable=false`.

```bash
# Tune only the legacy mlx-lm compatibility lane
MLX_BATCH_BATCH_WINDOW_MS=100 \
MLX_BATCH_MAX_BATCH_SIZE=16 \
mlx-batch-server
```

Do not quote historical "350 tok/s batched" figures as fused Flash performance.
Warm fused Flash receipts live in the 2026-09-04 acceptance artifacts (~53 tok/s
direct median decode at concurrency 1). True multi-row tensor batching remains a
future QSA/GDN cut, not a public promise.

---

## Security

The server ships **open by default** (`SECURITY_LEVEL=0`) so existing deployments keep working. Set a level to lock the surface down:

| Level | Behavior |
|-------|----------|
| `0` | Open. No auth, every request maps to a stable pseudo-owner. Default. |
| `1` | *Deprecated.* Treated internally as `2` with a warning. |
| `2` | HMAC **or** session token **or** API key (any one of them). |
| `3` | Session token only (HMAC + API key fallback disabled). |

When the level is `>0`, every protected route — including `/api/admin/models/{load,unload,alias}` — requires a credential. `/health` and `/v1/ready` stay open at all levels for load balancers.

```bash
# Static API key (simplest, single-secret deploys)
SECURITY_LEVEL=2 API_KEY=sk-mlx-… mlx-batch-server

# HMAC clients (machine-to-machine, /hmac/register issues secrets)
SECURITY_LEVEL=2 API_KEY=sk-… mlx-batch-server
curl -H "x-api-key: sk-…" -X POST http://127.0.0.1:10240/hmac/register \
     -d '{"client_id":"node-1","description":"build agent"}' \
     -H "Content-Type: application/json"

# Session-only (browser sessions via /auth/login)
SECURITY_LEVEL=3 SESSION_AUTH_ENABLED=true mlx-batch-server
```

Auxiliary auth env vars: `API_KEY_HEADER` (default `x-api-key`), `REDIS_URL` (Redis-backed sessions/API keys/rate-limit), `SESSION_AUTH_ENABLED`, `SESSION_PROVIDER`, `SESSION_TTL_HOURS`, `RATE_LIMIT_ENABLED`, `ACCESS_REGISTRATION_SECRET` (enables `/access` HTML registration page), `MLX_BATCH_HMAC_SECRETS_FILE` (XDG path by default), `HMAC_TIMESTAMP_TOLERANCE`.

The auth router family (`/auth/*`, `/hmac/*`, `/access`) is **opt-in** — it only mounts when at least one auth-related env var is configured.

---

## Operator UI

A standalone htmx admin lives in `mlx_batch_server.operator` and runs as a sibling app on port **10241**:

```bash
# Inference (port 10240)
mlx-batch-server &

# Operator UI (port 10241) — connects back to inference at 10240
mlx-batch-operator serve

# Custom inference URL / port
MLX_BATCH_OPERATOR_INFERENCE_BASE_URL=http://localhost:10240 \
    mlx-batch-operator serve --port 10241
```

Tabs: Fleet (live runtime + model summary), Sessions (recent playground sessions with delete-guard), Logs (tail + SSE follow), Lifecycle (status + restart/stop), Playground (in-browser SSE prompt with response chaining).

Auth posture inherits inference: if you start inference with `SECURITY_LEVEL=2`, the operator UI also requires that key. Override per side with:

| Variable | Effect |
|----------|--------|
| `MLX_BATCH_OPERATOR_SECURITY_LEVEL` | Force a different operator level than inference. |
| `MLX_BATCH_OPERATOR_REQUIRE_AUTH=true` | Force operator auth even when inference is open (useful behind a public proxy). |
| `MLX_BATCH_INTERNAL_API_KEY` | Key the operator forwards on the loopback playground proxy. |

The operator's `/health` and `/api/health` stay open at all levels so monitoring keeps working.

There is also a thin landing page at `http://127.0.0.1:10240/admin` on the inference port — it links to the richer operator UI on port 10241 and is gated by the same `SECURITY_LEVEL`.

---

## Readiness

Two health surfaces, used for different purposes:

| Endpoint | Purpose | Auth | Body |
|----------|---------|------|------|
| `GET /health` | Lightweight liveness for load balancers | open | `{"status":"ok", …}` |
| `GET /v1/ready` | Rich readiness — process, models loaded, batch coordinators, config, auth backends | open | `{"ready":bool, "checks":{…}}` |

`/v1/ready` returns `200` only when every check passes; otherwise `503` with the failing check called out. When `SECURITY_LEVEL>0` the readiness payload also includes an `auth_backends` block reporting Redis connectivity for sessions/API keys.

---

## HF Model Cards

Tooling for keeping the LibraxisAI Hugging Face model cards consistent. Sources of truth:

- `templates/HF_MODEL_CARD.md` — canonical card template with placeholders, fixed `## Inference tested on` section pointing here, and the canonical Vibecrafted footer.
- `scripts/rewrite_hf_model_cards.py` — full rewrite of every LibraxisAI card from the template, preserving metrics and base lineage when present.
- `scripts/backfill_hf_inference_section.py` — conservative patch that only adds `## Inference tested on` to cards that don't have it yet.
- `scripts/backfill_hf_canonical_footer.py` — conservative patch that only appends the canonical Vibecrafted footer to cards that don't have any form of it yet.

All scripts default to **dry-run** (list which cards would change without pushing). Add `--apply` to actually push, or use the `*-apply` Make targets:

```bash
# One-time auth
hf auth login

# Dry-run a full rewrite
make hf-rewrite

# Push a full rewrite (idempotent commit message: "card: full rewrite from canonical template")
make hf-rewrite-apply

# Backfill only the inference section across cards that lack it
make hf-backfill-inference         # dry-run
make hf-backfill-inference-apply   # push

# Backfill only the canonical footer across cards that lack any form of it
make hf-backfill-footer            # dry-run
make hf-backfill-footer-apply      # push

# Filters (work with all the above)
make hf-rewrite HF_LIMIT=5
make hf-backfill-inference HF_ONLY="Bielik Qwen"
```

The backfill scripts are intentionally conservative: they never delete content, never replace existing variants, and skip any card that already has the target section. Use `hf-rewrite` when you want a full normalisation pass.

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
make test-fast              # Unit/contract tests; no local model loading
make test-model-integration # Explicit model/download/optional-dependency gate

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
| [Responses API Guide](docs/responses/) | Primary `/v1/responses` surface |
| [Legacy mlx-lm batch](docs/batch/) | Compatibility coordinator; not fused Flash |
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
git clone https://github.com/LibraxisAI/mlx-batch-server.git
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
