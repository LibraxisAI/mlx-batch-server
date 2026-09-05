# Anthropic Messages compatibility

MLX Batch Server exposes Anthropic Messages at
`POST /anthropic/v1/messages`. Configure the Anthropic Python client with the
server root plus `/anthropic`:

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:10241/anthropic",
    api_key="local-key",
)
message = client.messages.create(
    model="grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit",
    max_tokens=64,
    messages=[{"role": "user", "content": "Reply briefly."}],
)
```

This is a bounded compatibility profile, not a claim of complete Anthropic API
parity. `/v1/responses` remains the primary product surface and the same
canonical inference runtime owns both protocols.

## Admitted profile

| Layer | Admitted version/profile | Evidence |
|---|---|---|
| Python SDK | exactly `anthropic==0.96.0` | synchronous and asynchronous non-stream plus synchronous and asynchronous SSE |
| Raw wire | `POST /anthropic/v1/messages`, `anthropic-version: 2023-06-01` | fields outside or newer than the generated SDK models are sent as JSON without monkey-patching the SDK |
| Capability authority | W3-AA/AB/AC/AC2 source profile resolved from the selected alias and runtime-role receipt | one preflight admission before mapping, model acquisition, or `StreamingResponse` creation |
| Local service tier | requests `auto` and `standard_only`; delivered `usage.service_tier` is always `standard` | this process has no priority or batch capacity lane |
| Acceptance receipt | `mlx-batch-server.live-api-acceptance.v1` | finalized JSON with exact required probe and matrix cell IDs |

## Supported requests

The public admission verifier covers all of these on one server instance:

| Area | Supported behavior |
|---|---|
| Messages | text content, system text, sampling controls, exact stop sequences, HTTP response and SSE lifecycle |
| SDK modes | sync non-stream, async non-stream, sync SSE, async SSE with `anthropic==0.96.0` |
| Tools | caller-defined custom tools, `tool_use`, and the W2 `tool_result` fidelity contract |
| Thinking | omitted or `{ "type": "disabled" }`; neither form may emit `thinking`, `thinking_delta`, `signature_delta`, or `redacted_thinking` |
| Service tier | requested `auto` and `standard_only`, both reporting the actual delivered tier `standard` |
| Images | direct `image` with `source.type=base64` or an admitted `url`; support is source-field-specific |
| Documents | direct `document` using admitted canonical file sources; the admission receipt uses inline `text/plain` data |
| Search results | caller-supplied `search_result` blocks are preserved in order as delimited untrusted text |
| Rich-content order | one request interleaves text, image, text, document, search result, and text; source verification proves the mapper keeps that order and live verification requires public success |

A `search_result.source` URL is provenance only. It is never interpreted as a
hosted search request, fetch instruction, or network authorization. URL and
file-id image/document forms are admitted only when the composed runtime
publishes that exact source field; support for one source form never implies
support for another.

## Explicit refusals

Every cell below is exercised with both `stream=false` and `stream=true`.
Admission requires HTTP 400 with top-level `type=error`,
`error.type=invalid_request_error`, a field-specific message, matching
`request-id` header and `request_id` body, zero SSE event bytes, and zero change
in the isolated runtime's inference-start counter.

| Matrix entry | Result |
|---|---|
| `cache_control` | refused; no prompt-cache semantic owner |
| `container` | refused; no container runtime |
| `inference_geo` | refused; local execution has no geography router |
| `output_config.format` | refused; no structured-output execution owner |
| `effort=high` | refused; no effort scheduler |
| `effort=max` | refused; no effort scheduler |
| `citations.enabled=true` | refused; citation semantics are not projected |
| `server_tool_use` | refused; no Anthropic-hosted tool execution |
| hosted tool definition such as `web_search_20250305` | refused; only caller-defined custom tools execute |
| `web_search_tool_result` | refused; no hosted-result continuation |
| `container_upload` | refused; no container file owner |
| enabled thinking budgets (minimum and larger probes) | refused; no budget enforcement or production signature owner |
| prior `thinking` continuation | refused; signatures cannot be verified and re-emitted |
| `redacted_thinking` continuation | refused; opaque reasoning cannot be restored |

Enabled extended thinking is therefore not supported. Runtime reasoning may
exist internally, but it is not enough to claim Anthropic extended-thinking
compatibility and never creates unsigned thinking blocks.

## Reproducible admission

Start the final candidate build separately on a caller-selected free port. The
example uses `10241`; do not point this admission run at production `8100` or at
the foreign developer service on `10240`. The isolated candidate must expose
the canonical `/health` payload, including
`role_runtime.runtime_stats.executor.active_requests` and `tombstones`, so the
refusal matrix can prove that no cell crossed the inference boundary.

```bash
export MLX_BATCH_ADMISSION_BASE_URL=http://127.0.0.1:10241
export MLX_BATCH_ADMISSION_MODEL=grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit
export MLX_BATCH_ADMISSION_KEY=replace-with-local-key
export MLX_BATCH_PUBLIC_FIXTURE_URL=https://fixtures.example/public.txt
export MLX_BATCH_PRIVATE_REDIRECT_URL=https://fixtures.example/redirect-to-loopback

uv run --with 'anthropic==0.96.0' \
  python scripts/quality/verify_live_mlx_batch_api.py \
  --base-url "$MLX_BATCH_ADMISSION_BASE_URL" \
  --model "$MLX_BATCH_ADMISSION_MODEL" \
  --api-key-env MLX_BATCH_ADMISSION_KEY \
  --anthropic-api-key-env MLX_BATCH_ADMISSION_KEY \
  --public-url "$MLX_BATCH_PUBLIC_FIXTURE_URL" \
  --private-redirect-url "$MLX_BATCH_PRIVATE_REDIRECT_URL" \
  --receipt artifacts/anthropic-public-admission.json
```

The command exits non-zero if the SDK version differs, a required probe or
matrix cell is missing or duplicated, a supported call fails, an unsupported
field is ignored, a streaming refusal opens SSE, request IDs diverge, an
unsupported call starts inference, or the receipt cannot be completed. The
receipt always sets `finalized: true` and redacts credentials and URL query
strings.

For source-only verification, including independent mutations of the five
admission clauses:

```bash
uv run python scripts/quality/verify_mlx_batch_api_contract.py \
  --section anthropic --expect green --no-imports
```

This profile does not claim hosted web/search/fetch execution, containers,
prompt caching, citations, structured output, priority/batch service tiers, or
full extended-thinking support.
