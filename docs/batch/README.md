# Batch Processing Module

MLX Batch Server includes a batch processing module for efficient handling of
concurrent requests. This directory contains documentation for the batch
inference subsystem.

- `HowToUse.md` - Configuration and usage examples
- `api-tester.html` - Interactive batch testing tool

The batch module wraps mlx-lm's `BatchGenerator` to provide transparent
batching of concurrent `/v1/responses` requests. When enabled, requests
arriving within the batch window are collected and processed together,
improving throughput for concurrent workloads.

## Quick Start

```bash
# Start server with default batch settings
mlx-batch-server

# Check batch stats
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

## Target Performance

- 10+ concurrent streaming requests
- 500+ tok/s total throughput (70B model)
- <500ms time-to-first-token per request
- <150MB overhead per concurrent request

Refer to `HowToUse.md` for detailed configuration and testing instructions.

---
Vibecrafted. with AI Agents by VetCoders (c)2024-2026 The LibraxisAI Team
