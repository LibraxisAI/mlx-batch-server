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
- 2026-03-11 22:05 CET — live smoke on repo-id runtime `LibraxisAI/Qwen3.5-VL-122B-A10B-mlx-crk-mxfp4` confirmed the new residency contract and isolated the next failure class.
  - Server was launched in foreground on `:10240` and the model was loaded by repo id, not local path.
  - Load truth after `make load`:
    - `cache_size = 1`
    - `loaded_models_by_backend.wrapper = ["libraxisai/qwen3.5-vl-122b-a10b-mlx-crk-mxfp4"]`
    - `loaded_models_by_backend.vlm = ["libraxisai/qwen3.5-vl-122b-a10b-mlx-crk-mxfp4"]`
    - runtime contract reported by the server:
      - `product_residency = single_model`
      - text runtime = `mlx-vlm.language_model`
      - multimodal runtime = `mlx-vlm`
      - multimodal execution = `single_flight`
    - `make ps` printed one resident entry with `backend=wrapper+vlm`.
  - Request sequence exercised against the same model:
    1. `text` request — failed
    2. `image` request — completed
    3. `text` request — failed again
  - After the full `text -> image -> text` sequence:
    - `make ps` still showed exactly one resident model
    - `/health` still reported `loaded_models_count = 1`
    - no second or third hidden residency appeared
  - Failure class is therefore no longer lifecycle or residency. Shared residency held under mixed modality traffic.
  - The remaining break is text-lane compatibility on a VLM-primary runtime:
    - image lane succeeds on the shared resident model
    - text lane fails when routed through the VLM language tower
    - exact server-side error: `'LanguageModelOutput' object is not subscriptable`
  - Working hypothesis to investigate:
    - our current text generation / batching path assumes the `mlx_lm.generate.stream_generate(...)` output contract
    - the `language_model` exposed by `mlx_vlm` is not plug-compatible with that assumption
    - the incompatibility may live in generation output shape, prompt-cache assumptions, or the adapter around streaming/token extraction
- 2026-03-11 22:13 CET — VLM text-lane compatibility fix landed and the live mixed-modality contract is now green on the repo-id runtime.
  - Local seam repair implemented in `src/mlx_batch_server/chat/mlx/model_types.py`:
    - added `MLXLMCompatibleLanguageModel` around the resident VLM `language_model`
    - unwraps `LanguageModelOutput` to raw logits for `mlx_lm`
    - maps `input_embeddings` to `inputs_embeds`
    - normalizes batched cache offsets before the VLM tower sees them, so `mlx_lm` batch caches no longer trip the Qwen3.5-VL attention path
  - Adjacent batch finalization hardening implemented in `src/mlx_batch_server/batch/generator.py`:
    - `BatchGenerator.stats()` zero-duration snapshots no longer fail the request
    - server falls back to raw counters with safe `0.0` throughput when timing is zero
  - Focused regression coverage added:
    - real `mlx_lm.generate_step(...)` regression for the sequential text lane
    - real `mlx_lm.BatchGenerator._step(...)` regression for the batch text lane
    - real `mlx_lm.BatchGenerator.insert()/next()` regression for batched cache-offset normalization
    - batch stats regression proving successful streams survive `ZeroDivisionError` during stats collection
  - Focused gates after implementation:
    - `uv run pytest -q tests/chat/mlx/test_chat_generator_limits.py tests/batch/test_batch_generator.py` → `11 passed`
    - `uv run ruff check src/mlx_batch_server/chat/mlx/model_types.py src/mlx_batch_server/batch/generator.py tests/chat/mlx/test_chat_generator_limits.py tests/batch/test_batch_generator.py` → clean
    - `uv run mypy src/mlx_batch_server/chat/mlx/model_types.py src/mlx_batch_server/batch/generator.py` → clean
  - Fresh-process live smoke on `LibraxisAI/Qwen3.5-VL-122B-A10B-mlx-crk-mxfp4`:
    1. loaded model by repo id on a fresh server
    2. `text` request completed
    3. `image` request completed
    4. final `text` request on the streaming batch lane completed with tail events:
       - `response.output_text.done`
       - `response.content_part.done`
       - `response.output_item.done`
       - `response.completed`
       - `[DONE]`
  - Residency truth after the full `text -> image -> text(stream)` sequence:
    - `/health` reported `loaded_models_count = 1`
    - `loaded_models_by_backend.wrapper = ["libraxisai/qwen3.5-vl-122b-a10b-mlx-crk-mxfp4"]`
    - `loaded_models_by_backend.vlm = ["libraxisai/qwen3.5-vl-122b-a10b-mlx-crk-mxfp4"]`
    - `loaded_models_by_backend.batch = ["libraxisai/qwen3.5-vl-122b-a10b-mlx-crk-mxfp4"]`
    - runtime contract stayed:
      - `product_residency = single_model`
      - text runtime = `mlx-vlm.language_model`
      - multimodal runtime = `mlx-vlm`
      - multimodal execution = `single_flight`
