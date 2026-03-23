# VLM-Primary Text-Lane Compatibility Autopsy

## Current state

- Shared residency is working as intended.
  - `ResponsesAdapter.generate()` routes media requests to the VLM lane and text requests to the text lane in [src/mlx_batch_server/responses/adapter.py](src/mlx_batch_server/responses/adapter.py).
  - `MLXModel.load()` now loads multimodal-capable models through `mlx_vlm`, and `MLXModel.text_model` exposes `model.language_model` for text generation in [src/mlx_batch_server/chat/mlx/model_types.py](src/mlx_batch_server/chat/mlx/model_types.py).
  - `ChatGenerator.generate_stream()` and `BatchChatGenerator._get_or_create_generator()` both consume `self.model.text_model`, so sequential text and batched text are already aiming at the same shared language tower in [src/mlx_batch_server/chat/mlx/chat_generator.py](src/mlx_batch_server/chat/mlx/chat_generator.py) and [src/mlx_batch_server/batch/generator.py](src/mlx_batch_server/batch/generator.py).
- The observed `text -> image -> text` sequence on `LibraxisAI/Qwen3.5-VL-122B-A10B-mlx-crk-mxfp4` proves residency is not the bug.
  - Text fails before the image request.
  - Image succeeds on the same resident model.
  - Text fails again after the image request.
  - Therefore the image path is not poisoning cache or residency state. The text lane is already broken on first contact.

## Root cause

### Broken contract

The exact incompatibility is **generation output shape / forward return contract**.

- Our text lane passes the VLM language tower into `mlx_lm`.
  - `MLXModel.text_model` currently returns `getattr(self.model, "language_model", self.model)`.
- The `mlx_vlm` language tower for Qwen-style VLMs returns a `LanguageModelOutput` object, not a raw logits tensor.
  - In the repo-pinned `mlx-vlm-local`, `LanguageModelOutput` is a dataclass with a `.logits` field in `../mlx-vlm-local/mlx_vlm/models/base.py`.
  - The Qwen3.5 language tower returns `LanguageModelOutput(logits=out)` in `../mlx-vlm-local/mlx_vlm/models/qwen3_5/language.py`.
- `mlx_lm` text generation assumes the model forward pass returns a tensor that can be sliced directly.
  - `mlx_lm.generate.generate_step()` does:
    - `logits = model(...)`
    - `logits = logits[:, -1, :]`
  - `mlx_lm.generate.BatchGenerator._step()` does the same shape assumption.
- That is why the exact error is:
  - `'LanguageModelOutput' object is not subscriptable`

### Why image requests still work

The multimodal lane does **not** use the `mlx_lm` contract.

- Our image path calls `mlx_vlm.generate` from [src/mlx_batch_server/responses/adapter.py](src/mlx_batch_server/responses/adapter.py).
- The repo-pinned `mlx_vlm` generator explicitly unwraps `outputs.logits` when talking to `model.language_model`.
- So the image lane succeeds because `mlx_vlm` understands its own `LanguageModelOutput` contract, while the text lane hands the same tower to `mlx_lm`, which does not.

## What is not broken

- Not residency.
  - The live smoke already proved one shared runtime remains resident through mixed traffic.
- Not chat template selection.
  - The failure occurs after the forward pass result is returned from the model, not while preparing prompts.
- Not prompt cache as the primary root cause.
  - The first text request already fails.
  - A synthetic repro hits the same error with `prompt_cache=[]`.
  - The Qwen3.5 VLM language tower also exposes `make_cache()` and cache primitives built on top of `mlx_lm` cache types, so cache compatibility is not the first contract that breaks.

## Narrowest credible fix

### Proposal

Add a **small `mlx_lm`-compat adapter around the VLM language tower** and make `MLXModel.text_model` return that adapter for multimodal runtimes.

The adapter should:

- wrap `model.language_model` from the resident `mlx_vlm` runtime
- return raw logits tensors from `__call__`
  - if the wrapped model returns `LanguageModelOutput`, return `output.logits`
  - if the wrapped model already returns a tensor, pass it through unchanged
- normalize keyword compatibility
  - map `input_embeddings` from `mlx_lm` to `inputs_embeds` expected by VLM towers when needed
- preserve cache and batching capabilities
  - forward `make_cache()`
  - expose `layers`
  - expose `head_dim`
  - expose `n_kv_heads`
- remain tree-walkable / MLX-friendly
  - the safest shape is an `nn.Module` wrapper that stores the underlying tower

### Why this is the smallest strong move

- It preserves the chosen product contract:
  - one shared multimodal residency
  - text batching through the resident language tower
  - multimodal single-flight unchanged
- It keeps the repair surface at the natural seam:
  - `MLXModel.text_model`
- It avoids broad changes in:
  - `ResponsesAdapter`
  - `OpenAIAdapter`
  - `ChatGenerator`
  - `BatchChatGenerator`
  - `wrapper_cache`
- It does not require forking or rewriting `mlx_lm` batch generation.

## Why other fixes are worse

- Patching `responses/adapter.py` alone is too high-level.
  - The contract break is below the adapter layer.
- Special-casing text requests through `mlx_vlm.generate` would violate the product contract.
  - That would bypass text batching through the language tower.
- Reverting to dual residency violates the selected architecture for this research round.
- Patching `mlx_lm` directly is broader and riskier.
  - It expands the maintenance surface into upstream dependency behavior instead of fixing our local compatibility seam.

## Required tests and runtime checks

### Tests that exist but are not strong enough

Current tests only prove that the right object is selected, not that the selected object satisfies the `mlx_lm` forward contract.

- `tests/chat/mlx/test_chat_generator_limits.py`
  - currently asserts that `stream_generate()` receives `wrapper.model.text_model`
- `tests/batch/test_batch_generator.py`
  - currently asserts that `BatchGenerator` receives the language model object

These tests miss the real bug because they monkeypatch away `mlx_lm` itself.

### Tests to add after implementation

1. `MLXModel.text_model` compatibility test
   - Load or fake a multimodal runtime whose `language_model` returns `LanguageModelOutput`.
   - Assert that `model.text_model(...)` returns a logits tensor shaped like `[batch, seq, vocab]`.
   - Assert that `make_cache`, `layers`, `head_dim`, and `n_kv_heads` remain available.

2. Sequential text-lane regression
   - Use the real `mlx_lm.generate.generate_step()` against a fake VLM tower that returns `LanguageModelOutput`.
   - Verify no `TypeError` is raised once the adapter is in place.

3. Batch text-lane regression
   - Use the real `mlx_lm.generate.BatchGenerator._step()` or `BatchChatGenerator` against the same fake VLM tower.
   - Verify no `TypeError` is raised and token generation proceeds.

4. Existing routing tests should be updated, not removed
   - Keep the current assertions that text requests use the language tower.
   - Extend them to assert **compatibility**, not just identity.

### Required runtime verification

Run a real repo-id smoke on `LibraxisAI/Qwen3.5-VL-122B-A10B-mlx-crk-mxfp4` after the fix:

1. Load the model by repo id.
2. Send a text-only request.
3. Send an image request.
4. Send another text-only request.
5. Repeat at least one text request through the streaming path with batch inference enabled.
6. Confirm:
   - exactly one resident wrapper/VLM runtime remains loaded
   - image still succeeds
   - both text requests succeed
   - no `'LanguageModelOutput' object is not subscriptable` appears in logs

## Evidence gathered in this research pass

### Local repo checks

- `uv run pytest -q tests/chat/mlx/test_chat_generator_limits.py tests/batch/test_batch_generator.py`
  - result: `7 passed`
  - meaning: current tests only cover object routing, not real `mlx_lm` compatibility

### Environment confirmation

- `uv run python` resolved:
  - Python `3.12.3`
  - `mlx-lm 0.31.0`
  - `mlx-vlm 0.4.0.dev0`
- The repo model config for `LibraxisAI/Qwen3.5-VL-122B-A10B-mlx-crk-mxfp4` reports:
  - `model_type = qwen3_5_moe`
  - `vision_config_present = True`
  - `text_config_present = True`

### Synthetic repro

Using the real repo environment (`uv run python`):

- `mlx_lm.generate.generate_step(...)` with a fake model returning `LanguageModelOutput(logits=...)`
  - result: `TypeError: 'LanguageModelOutput' object is not subscriptable`
- `mlx_lm.generate.BatchGenerator._step(...)` with the same fake model
  - result: `TypeError: 'LanguageModelOutput' object is not subscriptable`

That reproduces the exact failure class without needing the 122B model in memory and isolates the broken contract to the `mlx_lm` forward-return assumption.

## Migration plan

1. Add the compatibility adapter at the `MLXModel.text_model` seam.
2. Strengthen the existing text-lane tests so they execute the real `mlx_lm` generation contract against a fake VLM tower.
3. Run the targeted unit tests.
4. Run the real `text -> image -> text` smoke on the Qwen3.5-VL repo-id model.

## Quick win

The smallest strong implementation step is:

- introduce a request-transparent `mlx_lm`-compatible wrapper for VLM `language_model`
- return that wrapper from `MLXModel.text_model`
- prove it with one sequential regression, one batch regression, and the real `text -> image -> text` runtime smoke
