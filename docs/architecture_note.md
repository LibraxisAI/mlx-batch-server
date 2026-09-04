# Architecture Note: Universal Multimodal Runtime Contract

> **SUPERSEDED (2026-09-04 polarize).** This note describes the pre-fusion
> wrapper / VLM split-brain. It is not current product truth.
>
> Current contract: `mlx-batch-server` owns `/v1/responses` on signed roles
> 8100–8102. Role `main` (8100) is fused Qwen Flash (`fused_mtp_mlx`) with live
> MTP and honest `tensor_batch_mode=row_serial` / `text.batch_capable=false`.
> MTPLX and oMLX are bounded donors. See `AGENTS.md`.

## Current State

The codebase currently suffers from a split-brain model lifecycle, leading to hidden dual residency for multimodal models.

1.  **Text/Tools Execution Path (`src/mlx_batch_server/chat/*`, `src/mlx_batch_server/batch/*`)**:
    *   Loads models primarily through `MLXModel.load` (using `mlx_lm.load` under the hood).
    *   Caches instances centrally in `MLXWrapperCache` inside `src/mlx_batch_server/chat/mlx/wrapper_cache.py`.
    *   Supports robust text batching via `BatchRequestCoordinator` and `BatchChatGenerator`.

2.  **Image/Video Execution Path (`src/mlx_batch_server/responses/adapter.py`)**:
    *   Operates independently. When it encounters an image/video request, it calls `mlx_vlm.load()`.
    *   Caches the resulting `(model, processor)` tuple in a completely isolated dictionary: `_VLM_CACHE` (with its own lock `_VLM_LOCK`).
    *   Bypasses the batching coordinator entirely, sending requests directly to `vlm_generate` or `vlm_stream_generate`.

**The Root Cause of Harm:**
Because a multimodal model like Qwen2.5-VL or Pixtral can handle *both* text and images, a user hitting the `/v1/chat/completions` endpoint with a text-only prompt and an image prompt for the *exact same model* will trigger both execution paths. This causes the exact same weights to be loaded twice—once into `MLXWrapperCache` via `mlx_lm` and once into `_VLM_CACHE` via `mlx_vlm`, doubling the VRAM footprint and breaking observability (since unloaded models might remain in the other cache).

## Required Product/Runtime Contract

The production requirement is strict: **one model, one product surface, arbitrary request order, without duplicate residency.**

*   **Single Unified Registry:** The system must have exactly one cache/registry that owns the lifecycle of model weights, regardless of the prompt content type.
*   **Arbitrary Sequencing:** Sending [Text], [Text + Image], [Tools], [Text] must all hit the same resident model instance in VRAM.
*   **Feature Preservation:** Text batching is a non-negotiable requirement. Any text-only request hitting a VLM must still benefit from high-throughput batching, while image/video requests fall back to synchronous stream generation (while vision batching is still being built).

## Minimal Architecture

The leanest architecture that satisfies this contract without premature abstraction relies on exposing the underlying language model component of the VLM to the existing text batching engine.

1.  **Unified Model Cache (`UnifiedRegistry`)**:
    *   Deprecate `_VLM_CACHE` entirely.
    *   `MLXWrapperCache` becomes the single source of truth. When a model load is requested, the registry detects if the model is multimodal (e.g., via config `architectures`). If so, it uses `mlx_vlm.load()`; otherwise, `mlx_lm.load()`.
    *   The registry returns a unified `MLXModel` data container holding the instantiated `model` and `tokenizer/processor`.

2.  **Structural Shim for Text Batching**:
    *   The core insight: Multimodal models in MLX typically compose a vision tower and an LLM tower. The underlying text generation engine is often stored under `model.language_model` (as seen in `src/mlx_batch_server/embeddings/qwen3_vl_embedder.py`).
    *   Update the `MLXModel` data container to expose a `.language_model` property. For standard LLMs, it returns `self.model`. For VLMs, it returns `self.model.language_model`.
    *   The `BatchChatGenerator` in `src/mlx_batch_server/batch/generator.py` relies on `mlx_lm.generate.BatchGenerator(model=self.model.model, ...)`. We update this to initialize with `model=self.model.language_model`.
    *   This allows the existing text batching engine to step the VLM's text tower directly for pure-text requests.

3.  **Content-Aware Routing (`adapter.py`)**:
    *   `adapter.py` retrieves the model from the `UnifiedRegistry`.
    *   If the request contains *only* text/tools, `adapter.py` routes the payload to the `BatchRequestCoordinator`. The coordinator runs it through the VLM's extracted `.language_model` via the batch generator.
    *   If the request contains vision/video, `adapter.py` bypasses the batcher and invokes `vlm_stream_generate` using the full `MLXModel.model` reference.

## Migration Sequence

1.  **Phase 1: Consolidate the Registry (The `wrapper_cache.py` update)**
    *   Merge the loading logic. Update `MLXModel.load` to use `mlx_vlm.load()` if a VLM is detected.
    *   Update the cache to store `(model, processor)` properly wrapped in the `MLXModel` container.

2.  **Phase 2: Adapter Refactoring (`adapter.py` update)**
    *   Remove `_VLM_CACHE`, `_VLM_LOCK`, `get_loaded_vlm_models`, and `unload_vlm_model`.
    *   Update `_get_vlm_backend` to pull from the `MLXWrapperCache` instead. This immediately stops dual residency.

3.  **Phase 3: Expose LLM Component (`core_types.py` / `model_types.py` update)**
    *   Implement the `.language_model` property on the `MLXModel` wrapper to safely return the text transformer, regardless of whether it's an LLM or a VLM.
    *   Ensure the VLM `processor.tokenizer` fulfills the interface expected by `BatchChatGenerator` (e.g., exposing `.encode()`, `._eos_token_ids`).

4.  **Phase 4: Update Batcher (`batch/generator.py` update)**
    *   Change `BatchGenerator` initialization from `model=self.model.model` to `model=self.model.language_model`.

5.  **Phase 5: Route Text Requests on VLMs**
    *   Ensure `adapter.py` correctly routes text-only requests on VLM models into the batching engine, rather than forcing them down the slow synchronous VLM path just because the model ID matches a vision model.

## Risks and Tradeoffs

*   **Coupling to VLM Architecture:** Extracting `.language_model` assumes that the vision and text transformers are cleanly separated. For highly integrated architectures (like Llama 3.2 Vision's deep cross-attention layers), the text model may require dummy `pixel_values` or cross-attention state even for text-only prompts. This will cause the standard `mlx_lm` `BatchGenerator` to fail.
*   **Tokenizer vs. Processor Mismatch:** `BatchChatGenerator` expects a standard tokenizer. `mlx_vlm` returns a processor. We will need a shim to ensure the processor's tokenizer exposes identical APIs for token encode/decode and stop-word handling.
*   **Tradeoff:** By routing text-only requests to the batcher and image requests to synchronous generation on the *same model*, we risk thread starvation or lock contention on the model weights if batching operations and synchronous generation try to run concurrently. The inference lock mechanism will need to be unified.

## Acceptance Tests

*   **Test 1 (Residency):** Load `qwen2.5-vl`. Send a pure text prompt. Send an image prompt. Verify via the observability endpoint (`/v1/models/loaded`) that only ONE model instance exists in VRAM.
*   **Test 2 (Throughput):** Send 50 concurrent text-only requests to `qwen2.5-vl`. Verify they are processed by `BatchRequestCoordinator` (yielding high tokens/sec) and are not forced into the sequential `vlm_generate` queue.
*   **Test 3 (Functionality):** Send a mixed sequence of Text -> Image -> Text to the same VLM model. Verify responses are functionally correct and no memory spikes occur.

## Recommendation

Implement the **Unified Model Cache** immediately as a direct strike against the dual residency bug. Follow up by refactoring `BatchChatGenerator` to accept `model.language_model`. This preserves text batching without over-engineering VLM batching, honoring the current product constraint while setting up a clean runway for Scope A (true VLM batching) to arrive later.
