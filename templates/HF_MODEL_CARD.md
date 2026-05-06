<!--
Template: LibraxisAI Hugging Face model card
Usage: copy to <model_dir>/README.md, fill placeholders, ship.

Placeholders:
  {{MODEL_ID}}              — full HF id, e.g. LibraxisAI/Svetliq-11B-v3.0-mlx-7000-v1-preview
  {{MODEL_NAME}}            — short name, e.g. Svetliq-11B-v3.0-mlx-7000-v1-preview
  {{LICENSE}}               — apache-2.0 / mit / other-spdx
  {{LANGUAGE}}              — list of ISO codes, one per line; or omit for multilingual
  {{BASE_MODEL_ID}}         — upstream HF id this is derived from
  {{LIBRARY}}               — typically "mlx" for our MLX-converted/finetuned models
  {{PIPELINE_TAG}}          — text-generation / image-text-to-text / automatic-speech-recognition / etc.
  {{TAGS}}                  — list, one per line; always include `mlx`, `apple-silicon`
  {{INFERENCE_FLAG}}        — true | false (false for gated/preview/large)
  {{TASK_NAME}}             — human-readable eval task description
  {{METRIC_TYPE}}           — loss / accuracy / pass@1 / etc.
  {{METRIC_VALUE}}          — number
  {{METRIC_NAME}}           — Validation loss / Eval accuracy / etc.
  {{WIDGET_PROMPT_1}}       — example prompt #1 (optional, drop if not text-gen)
  {{WIDGET_TITLE_1}}
  {{WIDGET_PROMPT_2}}
  {{WIDGET_TITLE_2}}
  {{HEADLINE_DESCRIPTION}}  — one-paragraph elevator pitch
  {{INTENDED_USE_BULLETS}}  — markdown list
  {{OUT_OF_SCOPE_BULLETS}}  — markdown list
  {{TRAINING_TABLE_ROWS}}   — rows of `| Parameter | Value |` table (tab/pipe-separated)
  {{TRAINING_NOTES}}        — paragraph about dataset availability, adapter status, etc.
  {{EXAMPLE_PROMPT}}        — single representative prompt for code blocks
  {{EXAMPLE_INPUT}}         — block-quoted real input
  {{EXAMPLE_OUTPUT}}        — block-quoted real model output
  {{EXAMPLE_COMMENT}}       — one paragraph framing the example
  {{COMPARISON_TABLE_ROWS}} — `| Aspect | Base | This model |` rows
  {{LIMITATIONS_BULLETS}}   — honest list; if benchmark numbers are unknown, say so
  {{CITATION_BIBTEX}}       — full bibtex entry
  {{RELATED_BULLETS}}       — non-Inference cross-links (base model, dataset, sibling models)

Discipline (see memory feedback_model_card_discipline.md):
- The ## Inference tested on section is FIXED — do not edit, do not extend with curl, do not add api.libraxis.cloud or api-router.
- Cards describe the MODEL. Cross-links to LibraxisAI infra ONLY via:
    a) the fixed `## Inference tested on` section (single line)
    b) at most a single `## Related` entry if it makes sense for that specific model
- No /v1/responses, no /v1/chat/completions, no LBRX_API_KEY, no localhost:8088 — those belong on the api-router / mlx-batch-server own READMEs, never on a model card.
- Canonical signature is the literal Vibecrafted bytes (with the `.` after Vibecrafted, year range 2024-2026, suffix `LibraxisAI`). Do not paraphrase.
-->
---
license: {{LICENSE}}
language:
{{LANGUAGE}}
base_model:
  - {{BASE_MODEL_ID}}
library_name: {{LIBRARY}}
pipeline_tag: {{PIPELINE_TAG}}
tags:
{{TAGS}}
inference: {{INFERENCE_FLAG}}
model-index:
  - name: {{MODEL_NAME}}
    results:
      - task:
          type: {{PIPELINE_TAG}}
          name: {{TASK_NAME}}
        metrics:
          - type: {{METRIC_TYPE}}
            value: {{METRIC_VALUE}}
            name: {{METRIC_NAME}}
widget:
  - text: "{{WIDGET_PROMPT_1}}"
    example_title: "{{WIDGET_TITLE_1}}"
  - text: "{{WIDGET_PROMPT_2}}"
    example_title: "{{WIDGET_TITLE_2}}"
---

# {{MODEL_NAME}}

{{HEADLINE_DESCRIPTION}}

## Intended use

{{INTENDED_USE_BULLETS}}

## Out of scope

{{OUT_OF_SCOPE_BULLETS}}

## Training

| Parameter | Value |
|---|---|
{{TRAINING_TABLE_ROWS}}

{{TRAINING_NOTES}}

## Usage

### CLI

```bash
pip install mlx-lm

mlx_lm.generate \
  --model {{MODEL_ID}} \
  --prompt "{{EXAMPLE_PROMPT}}" \
  --max-tokens 400
```

### Python

```python
from mlx_lm import load, generate

model, tokenizer = load("{{MODEL_ID}}")

prompt = "{{EXAMPLE_PROMPT}}"
response = generate(model, tokenizer, prompt=prompt, max_tokens=400)
print(response)
```

### Multi-turn (chat template)

The model inherits the chat template from `{{BASE_MODEL_ID}}`. For multi-turn dialogue, apply the tokenizer's `apply_chat_template` before generation:

```python
from mlx_lm import load, generate

model, tokenizer = load("{{MODEL_ID}}")

messages = [
    {"role": "user", "content": "{{EXAMPLE_PROMPT}}"}
]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
response = generate(model, tokenizer, prompt=prompt, max_tokens=400)
print(response)
```

## Example output

**Input:**

> {{EXAMPLE_INPUT}}

**Output:**

> {{EXAMPLE_OUTPUT}}

{{EXAMPLE_COMMENT}}

## Comparison with the base model

| Aspect | {{BASE_MODEL_ID}} (base) | {{MODEL_NAME}} |
|---|---|---|
{{COMPARISON_TABLE_ROWS}}

## Limitations

{{LIMITATIONS_BULLETS}}

## License

{{LICENSE}}.

## Citation

```bibtex
{{CITATION_BIBTEX}}
```

## Inference tested on

[`LibraxisAI/mlx-batch-server`](https://github.com/LibraxisAI/mlx-batch-server)

## Related

{{RELATED_BULLETS}}

---

𝚅𝚒𝚋𝚎𝚌𝚛𝚊𝚏𝚝𝚎𝚍. with AI Agents by VetCoders (c)2024-2026 LibraxisAI
