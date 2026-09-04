# Batch Processing Module

This directory documents the **legacy mlx-lm** batch coordinator
(`BatchRequestCoordinator` / `BatchChatGenerator`). It is a compatibility lane
for unsupported models.

It is **not** the fused Qwen Flash product. Role `main` on 8100 uses
`fused_mtp_mlx`, reports `tensor_batch_mode=row_serial` and
`text.batch_capable=false`, and does not route `/v1/responses` through
mlx-lm `BatchGenerator`.

- `HowToUse.md` - Configuration and usage examples
- `api-tester.html` - Interactive batch testing tool

When the legacy lane is enabled, requests arriving within the batch window can
be collected for unsupported mlx-lm models. Do not treat `/v1/batch/stats` as
proof of fused Flash tensor batching.

## Quick Start

```bash
# Legacy compatibility lane only (unbound local bind)
mlx-batch-server

# Check legacy coordinator stats — not fused Flash tensor batching
curl http://localhost:10240/v1/batch/stats | jq
```

## Architecture

```
Request 1 ─┐
Request 2 ─┼─→ BatchRequestCoordinator ─→ BatchChatGenerator ─→ mlx-lm BatchGenerator
Request 3 ─┘         │                          │
                     │                          │
              (collect within              (parallel
               batch_window_ms)            inference)
                     │                          │
                     ▼                          ▼
              Per-request queue ←───── Token dispatch
```

## Target Performance (legacy mlx-lm lane only)

These numbers are historical mlx-lm coordinator targets. They are **not** fused
Flash promises. Fused Flash on 8100 is row-serial; warm concurrency-1 decode
was measured near 53 tok/s in the 2026-09-04 paired acceptance run.

- 10+ concurrent streaming requests (legacy coordinator)
- Do not claim 500+ tok/s fused Flash tensor batching
- <500ms time-to-first-token per request is a legacy target, not a fused SLA

Refer to `HowToUse.md` for detailed configuration and testing instructions.

---
Vibecrafted. with AI Agents by VetCoders (c)2024-2026 The LibraxisAI Team
