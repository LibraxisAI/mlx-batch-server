# VLM Runtime Findings

## 2026-03-11
- Stopped local server on port 10240 (PID 37485) before integrating runtime changes.
- Goal contract: MLX-VLM is primary runtime for multimodal-capable models; text-only requests batch via the model language tower; multimodal requests are single-flight for now.
- 2026-03-11 21:06 CET — shared multimodal runtime refactor truth pass green.
  - `wrapper_cache` is now the single residency owner for multimodal-capable runtimes.
  - `get_vlm_backend()` reuses the resident wrapper runtime instead of loading a second VLM cache entry.
  - text generation and text batching use the resident model language tower (`text_model`).
  - multimodal requests remain intentional single-flight via per-model `vlm_execution(...)` lock.
  - request-local `ChatTemplate` instances now isolate parser/tool/thinking state per request.
  - text batch lane now explicitly falls back for tools, custom stop, custom top_p, and structured output.
  - multimodal requests with tools or media across multiple turns are rejected as unsupported contract states.
  - focused gates: `ruff check` clean, `85 passed / 5 skipped` on targeted pytest suite.
