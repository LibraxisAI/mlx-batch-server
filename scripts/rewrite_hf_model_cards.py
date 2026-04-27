#!/usr/bin/env python3
# ruff: noqa: RUF001, PLR0911, PLR0912, PLR0915
"""Rewrite LibraxisAI Hugging Face model cards from the canonical card shape.

The script intentionally treats Hugging Face as the live tree:

- model list and README.md are fetched fresh during each run
- only README.md is uploaded
- every uploaded card uses one commit message
- metrics and base lineage are only emitted when present in existing card data
  or repository config metadata

Dry-run is the default. Pass ``--apply`` to upload cards.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

ORG = "LibraxisAI"
GOLD_STANDARD = "LibraxisAI/Svetliq-11B-v3.0-mlx-7000-v1-preview"
EXPLICIT_EMPTY_EXCLUSIONS = {
    "LibraxisAI/sVetliq-11b-v2-Preview-400",
    "LibraxisAI/sVetliq-11b-v1.5-Preview-3600",
}
COMMIT_MSG = "card: full rewrite from canonical template"
INFERENCE_SECTION = (
    "## Inference tested on\n\n"
    "[`LibraxisAI/mlx-batch-server`](https://github.com/LibraxisAI/mlx-batch-server)"
)
CANONICAL_FOOTER = "𝚅𝚒𝚋𝚎𝚌𝚛𝚊𝚏𝚝𝚎𝚍. with AI Agents by VetCoders (c)2024-2026 LibraxisAI"
FORBIDDEN = [
    "/v1/chat/completions",
    "/v1/responses",
    "api.libraxis.cloud",
    "api-router",
    "Vista",
    "LBRX_API_KEY",
    "noreply@anthropic.com",
    "localhost:8088",
]
MODEL_FILE_SUFFIXES = (
    ".safetensors",
    ".npz",
    ".gguf",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
)


@dataclass
class CardResult:
    repo_id: str
    status: str
    commit_sha: str = ""
    language: str = ""
    pipeline_tag: str = ""
    tags_count: int = 0
    metrics_present: str = "no"
    base_model: str = ""
    notes: list[str] = field(default_factory=list)


def as_plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [as_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: as_plain(v) for k, v in value.items()}
    try:
        return dict(value)
    except Exception:
        return value


def parse_frontmatter(readme: str) -> dict[str, Any]:
    match = re.match(r"---\n(.*?)\n---\n", readme, re.DOTALL)
    if not match:
        return {}
    data = yaml.safe_load(match.group(1)) or {}
    return data if isinstance(data, dict) else {}


def listify(value: Any) -> list[str]:
    value = as_plain(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif item is not None:
                out.append(str(item))
        return out
    return [str(value)]


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def load_json_from_hf(repo_id: str, filename: str) -> dict[str, Any]:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    except Exception:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def model_size_hint(repo_id: str) -> int | None:
    match = re.search(r"(\d+)\s*[bB]", repo_id)
    if not match:
        return None
    return int(match.group(1))


def quantization(repo_id: str, tags: list[str]) -> str:
    haystack = " ".join([repo_id, *tags]).lower()
    if "mxfp8" in haystack:
        return "MXFP8"
    if "mxfp4" in haystack:
        return "MXFP4"
    if "nvfp4" in haystack:
        return "NVFP4"
    if "bf16" in haystack or "fp16" in haystack:
        return "BF16/FP16"
    for bit in ("q8", "q6", "q5", "q4"):
        if bit in haystack:
            return bit.upper()
    return "Not declared"


def infer_pipeline(repo_id: str, existing: dict[str, Any], tags: list[str]) -> str:
    declared = existing.get("pipeline_tag")
    if declared:
        return str(declared)
    lower = repo_id.lower()
    tags_lower = {t.lower() for t in tags}
    if "whisper" in lower:
        return "automatic-speech-recognition"
    if "colqwen" in lower or "visual-retrieval" in tags_lower:
        return "visual-document-retrieval"
    if "vl" in lower or "huihui4" in lower or "image-text-to-text" in tags_lower:
        return "image-text-to-text"
    return "text-generation"


def infer_language(
    repo_id: str, existing: dict[str, Any], tags: list[str]
) -> list[str]:
    declared = listify(existing.get("language"))
    if declared:
        return declared
    lower = repo_id.lower()
    tags_lower = {t.lower() for t in tags}
    if "svetliq" in lower or "bielik" in lower or "polish" in tags_lower:
        return ["pl"]
    if "whisper" in lower:
        return ["en", "multilingual"]
    if "colqwen" in lower or "huihui4" in lower:
        return ["en", "pl", "multilingual"]
    return ["en"]


def normalize_tags(repo_id: str, existing: dict[str, Any], pipeline: str) -> list[str]:
    lower = repo_id.lower()
    tags = listify(existing.get("tags"))
    tags.extend(["mlx", "apple-silicon"])
    if "svetliq" in lower:
        tags.extend(["veterinary", "medical", "polish", "clinical", "bielik"])
    if "bielik" in lower:
        tags.extend(["bielik", "polish"])
    if "whisper" in lower:
        tags.extend(["whisper", "speech-to-text"])
    if "qwen" in lower:
        tags.append("qwen")
    if "vl" in lower or pipeline == "image-text-to-text":
        tags.extend(["vision", "multimodal"])
    if "colqwen" in lower:
        tags.extend(["visual-retrieval", "document-understanding", "colbert"])
    if "huihui" in lower:
        tags.extend(["huihui", "moe"])
    if "gpt-oss" in lower:
        tags.append("gpt-oss")
    q = quantization(repo_id, tags).lower()
    if q != "not declared":
        tags.extend(["quantized", q])
    return unique(tags)


def choose_base(existing: dict[str, Any], config: dict[str, Any]) -> list[str]:
    base = listify(existing.get("base_model"))
    clean_base = [b for b in base if b and not b.startswith(("/", "."))]
    if clean_base:
        return clean_base

    for key in ("_name_or_path", "name_or_path"):
        value = config.get(key)
        if isinstance(value, str) and "/" in value and not value.startswith(("/", ".")):
            return [value]
    return []


def choose_library(existing: dict[str, Any], repo_id: str, siblings: list[str]) -> str:
    declared = existing.get("library_name")
    if declared:
        return str(declared)
    lower = repo_id.lower()
    if "whisper" in lower:
        return "mlx-whisper"
    if "vl" in lower or "colqwen" in lower:
        return "mlx-vlm"
    if any(s.endswith(".safetensors") for s in siblings):
        return "mlx"
    return "mlx"


def inference_flag(repo_id: str) -> bool:
    lower = repo_id.lower()
    size = model_size_hint(repo_id)
    if size and size > 30:
        return False
    if "preview" in lower or "private" in lower:
        return False
    return False


def yaml_frontmatter(meta: dict[str, Any]) -> str:
    return yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()


def link_model(repo_id: str) -> str:
    return f"[`{repo_id}`](https://huggingface.co/{repo_id})"


def headline(
    repo_id: str, pipeline: str, base_models: list[str], tags: list[str]
) -> str:
    name = repo_id.split("/", 1)[1]
    base = base_models[0] if base_models else "the declared upstream model"
    q = quantization(repo_id, tags)
    lower = repo_id.lower()
    if "svetliq" in lower:
        return (
            f"`{name}` is a Polish veterinary clinical checkpoint in MLX format, "
            f"derived from `{base}` and packaged for local Apple Silicon inference."
        )
    if "bielik" in lower:
        return (
            f"`{name}` is an MLX {q} packaging of `{base}` for Polish and multilingual "
            "instruction-style generation on Apple Silicon."
        )
    if "whisper" in lower:
        return (
            f"`{name}` is an MLX-ready Whisper speech-to-text checkpoint derived from "
            f"`{base}` for local transcription on Apple Silicon."
        )
    if pipeline == "visual-document-retrieval":
        return (
            f"`{name}` is an MLX visual document retrieval model derived from `{base}`, "
            "built for image/page and text-query embedding workflows."
        )
    if pipeline == "image-text-to-text":
        return (
            f"`{name}` is an MLX vision-language checkpoint derived from `{base}`, "
            "packaged for local multimodal prompting on Apple Silicon."
        )
    return (
        f"`{name}` is an MLX {q} checkpoint derived from `{base}`, intended for local "
        "text generation on Apple Silicon."
    )


def intended_use(repo_id: str, pipeline: str) -> list[str]:
    lower = repo_id.lower()
    if "svetliq" in lower:
        return [
            "Polish veterinary clinical drafting and case reasoning for practitioner review",
            "Differential diagnosis, triage notes, drug-reference style explanations, and care-plan drafts",
            "Local Apple Silicon inference where data locality and operator control matter",
        ]
    if pipeline == "automatic-speech-recognition":
        return [
            "Local speech-to-text transcription on Apple Silicon",
            "Batch or interactive audio transcription experiments",
            "Multilingual ASR workflows when supported by the upstream Whisper checkpoint",
        ]
    if pipeline == "visual-document-retrieval":
        return [
            "Visual document retrieval over page images and text queries",
            "Late-interaction ranking experiments for PDFs, scans, and visually rich documents",
            "Apple Silicon local retrieval pipelines that need MLX-native weights",
        ]
    if pipeline == "image-text-to-text":
        return [
            "Local image-and-text reasoning on Apple Silicon",
            "Document, screenshot, chart, and visual question answering experiments",
            "Operator-controlled multimodal prototyping where hosted inference is not desired",
        ]
    return [
        "Local text generation and chat-style prompting on Apple Silicon",
        "MLX-LM experimentation with the declared upstream model family",
        "Offline or operator-controlled inference workflows",
    ]


def out_of_scope(repo_id: str, pipeline: str) -> list[str]:
    lower = repo_id.lower()
    items = [
        "Safety-critical decisions without domain expert review",
        "Claims of benchmark superiority not backed by published evaluation data",
        "Non-MLX runtime guarantees; this card documents the shipped HF checkpoint, not every possible serving stack",
    ]
    if "svetliq" in lower:
        items.insert(0, "Direct-to-owner veterinary diagnosis or treatment decisions")
        items.insert(1, "Languages other than Polish unless independently evaluated")
    if pipeline == "automatic-speech-recognition":
        items.append(
            "Speaker diarization, clinical interpretation, or audio enhancement"
        )
    if pipeline in {"image-text-to-text", "visual-document-retrieval"}:
        items.append("High-stakes visual interpretation without human review")
    return items


def training_rows(
    repo_id: str,
    base_models: list[str],
    library: str,
    pipeline: str,
    tags: list[str],
    config: dict[str, Any],
    siblings: list[str],
) -> list[tuple[str, str]]:
    rows = [
        ("Repository", f"`{repo_id}`"),
        (
            "Base model",
            ", ".join(f"`{b}`" for b in base_models)
            if base_models
            else "Not declared in public metadata",
        ),
        ("Task", f"`{pipeline}`"),
        ("Library", f"`{library}`"),
        ("Format", "MLX / Apple Silicon checkpoint"),
        ("Quantization", quantization(repo_id, tags)),
        (
            "Architecture",
            ", ".join(listify(config.get("architectures"))) or "Not declared in config",
        ),
        (
            "Model files",
            str(len([s for s in siblings if s.endswith(MODEL_FILE_SUFFIXES)])),
        ),
    ]
    if "model_type" in config:
        rows.append(("Config model_type", f"`{config['model_type']}`"))
    return rows


def usage_section(repo_id: str, pipeline: str, base_models: list[str]) -> str:
    prompt_pl = "Opisz krótko objawy odwodnienia u psa i kiedy pilnie skontaktować się z lekarzem weterynarii."
    prompt_en = (
        "Summarize the key signals in this document and list the next action items."
    )
    lower = repo_id.lower()
    prompt = prompt_pl if ("svetliq" in lower or "bielik" in lower) else prompt_en

    if pipeline == "automatic-speech-recognition":
        return f"""## Usage

### Python

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "audio.wav",
    path_or_hf_repo="{repo_id}",
)
print(result["text"])
```

### Notes

- Use local audio files supported by `mlx_whisper`.
- For long recordings, split audio into manageable chunks before transcription.
"""

    if pipeline == "visual-document-retrieval":
        return f"""## Usage

### Python

```python
# Example shape for MLX document-retrieval workflows.
# Use the model-specific retrieval wrapper in your application code.
model_id = "{repo_id}"
query = "Which page discusses treatment protocol changes?"
document_image = "page.png"
```

### Notes

- This checkpoint is for retrieval embeddings rather than free-form chat.
- Pair it with a ColBERT/MaxSim-style ranking implementation that supports the model layout.
"""

    if pipeline == "image-text-to-text":
        return f"""## Usage

### CLI

```bash
pip install mlx-vlm

python -m mlx_vlm.generate \\
  --model {repo_id} \\
  --image image.jpg \\
  --prompt "{prompt}" \\
  --max-tokens 256
```

### Python

```python
from mlx_vlm import generate, load

model, processor = load("{repo_id}")
response = generate(
    model,
    processor,
    prompt="{prompt}",
    image="image.jpg",
    max_tokens=256,
)
print(response)
```
"""

    base = base_models[0] if base_models else "the upstream model"
    return f"""## Usage

### CLI

```bash
pip install mlx-lm

mlx_lm.generate \\
  --model {repo_id} \\
  --prompt "{prompt}" \\
  --max-tokens 400
```

### Python

```python
from mlx_lm import load, generate

model, tokenizer = load("{repo_id}")

prompt = "{prompt}"
response = generate(model, tokenizer, prompt=prompt, max_tokens=400)
print(response)
```

### Multi-turn with the chat template

This checkpoint follows the tokenizer/chat-template contract inherited from `{base}` when the
template is present in the repository:

```python
from mlx_lm import load, generate

model, tokenizer = load("{repo_id}")

messages = [
    {{"role": "user", "content": "{prompt}"}},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
response = generate(model, tokenizer, prompt=prompt, max_tokens=400)
print(response)
```
"""


def comparison_section(repo_id: str, base_models: list[str], tags: list[str]) -> str:
    base = ", ".join(f"`{b}`" for b in base_models) if base_models else "Not declared"
    q = quantization(repo_id, tags)
    lower = repo_id.lower()
    if "svetliq" in lower:
        return f"""## Comparison with the base model

| Aspect | Base | This checkpoint |
|---|---|---|
| Lineage | {base} | Polish veterinary-domain checkpoint in MLX format |
| Domain emphasis | General instruction behavior from the base family | Veterinary clinical drafting, Polish case reasoning, and practitioner-facing assistance |
| Published benchmark delta | Not declared in public metadata | Not declared in public metadata |
"""
    if q != "Not declared":
        return f"""## Quantization notes

| Aspect | Original/base checkpoint | This checkpoint |
|---|---|---|
| Lineage | {base} | `{repo_id}` |
| Runtime target | Upstream runtime format | MLX on Apple Silicon |
| Quantization | Base precision or upstream-declared format | {q} |
| Published quality delta | Not declared in public metadata | Not declared in public metadata |
"""
    return f"""## Comparison with the base model

| Aspect | Base | This checkpoint |
|---|---|---|
| Lineage | {base} | `{repo_id}` |
| Runtime target | Upstream runtime format | MLX on Apple Silicon |
| Published benchmark delta | Not declared in public metadata | Not declared in public metadata |
"""


def limitations(
    repo_id: str, base_models: list[str], metrics_present: bool
) -> list[str]:
    items = [
        "No public benchmark claims are made by this card unless listed in the frontmatter.",
        "Validate outputs on your own domain data before relying on this checkpoint.",
        "Memory use and speed depend heavily on the exact Apple Silicon generation, unified-memory size, and prompt length.",
    ]
    if not metrics_present:
        items.insert(
            0,
            "No public benchmarks for this checkpoint are declared in the model metadata.",
        )
    if not base_models:
        items.append(
            "Base-model lineage is not declared in public metadata or config and is intentionally not guessed here."
        )
    if "svetliq" in repo_id.lower():
        items.append("Veterinary outputs require review by a licensed veterinarian.")
    return items


def related(repo_id: str, base_models: list[str]) -> list[str]:
    links = [f"Base model: {link_model(base)}" for base in base_models if "/" in base]
    return links or ["No related public model metadata is declared."]


def bibtex(repo_id: str) -> str:
    name = repo_id.split("/", 1)[1]
    key = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return f"""@misc{{libraxisai-{key},
  title = {{{name}}},
  author = {{LibraxisAI}},
  year = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}},
  note = {{MLX checkpoint published by LibraxisAI}}
}}"""


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def markdown_table(rows: list[tuple[str, str]]) -> str:
    return "\n".join(f"| {k} | {v} |" for k, v in rows)


def has_metrics(frontmatter: dict[str, Any]) -> bool:
    model_index = frontmatter.get("model-index")
    if not isinstance(model_index, list):
        return False
    return "metrics:" in yaml.safe_dump(model_index, allow_unicode=True)


def valid_model_index(frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a conservative model-index block, or [] if any metric is unsafe."""
    model_index = frontmatter.get("model-index")
    if not isinstance(model_index, list) or not model_index:
        return []
    for entry in model_index:
        if not isinstance(entry, dict):
            return []
        results = entry.get("results")
        if not isinstance(results, list) or not results:
            return []
        for result in results:
            if not isinstance(result, dict):
                return []
            metrics = result.get("metrics")
            if not isinstance(metrics, list) or not metrics:
                return []
            for metric in metrics:
                if not isinstance(metric, dict):
                    return []
                value = metric.get("value")
                if not isinstance(value, int | float):
                    return []
                if not metric.get("name") or not metric.get("type"):
                    return []
    return model_index


def build_card(
    repo_id: str, info: Any, existing_readme: str, config: dict[str, Any]
) -> tuple[str, CardResult]:
    cd = as_plain(info.cardData) or {}
    if not isinstance(cd, dict):
        cd = {}
    existing_fm = parse_frontmatter(existing_readme)
    merged = {**existing_fm, **cd}
    siblings = [s.rfilename for s in (info.siblings or [])]
    name = repo_id.split("/", 1)[1]
    pipeline = infer_pipeline(repo_id, merged, listify(merged.get("tags")))
    tags = normalize_tags(repo_id, merged, pipeline)
    language = infer_language(repo_id, merged, tags)
    base_models = choose_base(merged, config)
    library = choose_library(merged, repo_id, siblings)
    license_name = str(merged.get("license") or "other")
    model_index = valid_model_index(existing_fm)
    metrics = bool(model_index)
    metadata: dict[str, Any] = {
        "license": license_name,
        "language": language,
    }
    if base_models:
        metadata["base_model"] = base_models
    metadata.update(
        {
            "library_name": library,
            "pipeline_tag": pipeline,
            "tags": tags,
            "inference": inference_flag(repo_id),
        }
    )
    if "datasets" in merged:
        metadata["datasets"] = listify(merged.get("datasets"))
    if model_index:
        metadata["model-index"] = model_index
    if pipeline == "text-generation":
        if "svetliq" in repo_id.lower() or "bielik" in repo_id.lower():
            metadata["widget"] = [
                {
                    "text": "Wyjaśnij krótko różnicę między diagnostyką różnicową a rozpoznaniem.",
                    "example_title": "Polish instruction prompt",
                },
                {
                    "text": "Podsumuj najważniejsze ryzyka w planie wdrożenia.",
                    "example_title": "Polish reasoning prompt",
                },
            ]
        else:
            metadata["widget"] = [
                {
                    "text": "Summarize the operational risks in this deployment plan.",
                    "example_title": "Reasoning prompt",
                }
            ]

    parts = [
        f"---\n{yaml_frontmatter(metadata)}\n---",
        f"# {name}",
        headline(repo_id, pipeline, base_models, tags),
        "## Intended use",
        bullet_list(intended_use(repo_id, pipeline)),
        "## Out of scope",
        bullet_list(out_of_scope(repo_id, pipeline)),
        "## Training and conversion metadata",
        "| Parameter | Value |\n|---|---|\n"
        + markdown_table(
            training_rows(
                repo_id, base_models, library, pipeline, tags, config, siblings
            )
        ),
        (
            "This card only reports metadata present in the Hugging Face repository, "
            "existing card frontmatter, or public config files. Missing benchmark, dataset, "
            "or training-run details are left explicit rather than reconstructed."
        ),
        usage_section(repo_id, pipeline, base_models).rstrip(),
        "## Example output",
        "No public sample output is currently declared for this checkpoint. Run the usage example above against your own prompt or audio/image input to inspect behavior.",
        comparison_section(repo_id, base_models, tags).rstrip(),
        "## Limitations",
        bullet_list(limitations(repo_id, base_models, metrics)),
        "## License",
        f"`{license_name}`. Check the upstream/base model license as well when a base model is declared.",
        "## Citation",
        f"```bibtex\n{bibtex(repo_id)}\n```",
        INFERENCE_SECTION,
        "## Related",
        bullet_list(related(repo_id, base_models)),
        "---",
        CANONICAL_FOOTER,
        "",
    ]
    card = "\n\n".join(parts)
    validate_card(card)
    result = CardResult(
        repo_id=repo_id,
        status="ready",
        language=",".join(language),
        pipeline_tag=pipeline,
        tags_count=len(tags),
        metrics_present="yes" if metrics else "no",
        base_model=", ".join(base_models) if base_models else "",
    )
    if not base_models:
        result.notes.append("base lineage not declared")
    if not metrics:
        result.notes.append("no public benchmarks")
    return card, result


def validate_card(card: str) -> None:
    match = re.match(r"---\n(.*?)\n---\n", card, re.DOTALL)
    if not match:
        raise ValueError("missing frontmatter")
    fm = yaml.safe_load(match.group(1)) or {}
    if not fm.get("license"):
        raise ValueError("missing license")
    if not fm.get("library_name"):
        raise ValueError("missing library_name")
    if not fm.get("pipeline_tag"):
        raise ValueError("missing pipeline_tag")
    if not isinstance(fm.get("tags"), list) or len(fm["tags"]) < 2:
        raise ValueError("missing tags")
    if INFERENCE_SECTION not in card:
        raise ValueError("missing canonical inference section")
    if CANONICAL_FOOTER not in card:
        raise ValueError("missing canonical footer")
    for forbidden in FORBIDDEN:
        if forbidden in card:
            raise ValueError(f"forbidden token: {forbidden}")


def should_skip(repo_id: str, siblings: list[str]) -> str:
    if repo_id == GOLD_STANDARD:
        return "gold-standard exemplar"
    if repo_id in EXPLICIT_EMPTY_EXCLUSIONS:
        return "explicit empty-shell exclusion"
    meaningful = [s for s in siblings if s != ".gitattributes"]
    has_model_file = any(s.endswith(MODEL_FILE_SUFFIXES) for s in siblings)
    if not meaningful or not has_model_file:
        return "empty or card-only shell"
    return ""


def write_report(
    path: Path, results: list[CardResult], skipped: list[CardResult], applied: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LibraxisAI HF Model Card Rewrite Report",
        "",
        f"- mode: {'apply' if applied else 'dry-run'}",
        f"- rewritten: {sum(1 for r in results if r.status in {'uploaded', 'ready', 'unchanged'})}",
        f"- uploaded: {sum(1 for r in results if r.status == 'uploaded')}",
        f"- failed: {sum(1 for r in results if r.status == 'failed')}",
        f"- skipped: {len(skipped)}",
        "",
        "| repo_id | status | commit_sha | language | pipeline_tag | tags_count | metrics_present | base_model | notes |",
        "|---|---|---|---|---|---:|---|---|---|",
    ]
    for r in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.repo_id,
                    r.status,
                    r.commit_sha or "",
                    r.language or "",
                    r.pipeline_tag or "",
                    str(r.tags_count),
                    r.metrics_present,
                    r.base_model or "",
                    "; ".join(r.notes),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Skipped repos", ""])
    if skipped:
        for r in skipped:
            lines.append(f"- `{r.repo_id}`: {'; '.join(r.notes) or r.status}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Boundary report",
            "",
            "- The script rewrites only `README.md` and never touches weights, configs, tokenizers, or repository layout.",
            "- Cards intentionally omit model-index metrics unless metrics were already present in card metadata.",
            "- `Svetliq-11B-v3.0-mlx-7000-v1-preview` remains untouched as the operator-approved exemplar.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="upload README.md changes to HF"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="process first N eligible repos"
    )
    parser.add_argument(
        "--only", nargs="+", default=None, help="substring filter for repo ids"
    )
    parser.add_argument(
        "--sleep", type=float, default=1.2, help="seconds between uploads"
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="write markdown report"
    )
    parser.add_argument(
        "--preview-dir", type=Path, default=None, help="write generated README previews"
    )
    args = parser.parse_args()

    api = HfApi()
    print(f"# Listing models for org={ORG}", flush=True)
    models = list(api.list_models(author=ORG, limit=200))
    if args.only:
        models = [m for m in models if any(s in m.modelId for s in args.only)]
    print(f"# Found {len(models)} candidate models", flush=True)

    results: list[CardResult] = []
    skipped: list[CardResult] = []
    processed = 0

    for model in models:
        repo_id = model.modelId
        print(f"\n## {repo_id}", flush=True)
        try:
            info = api.model_info(repo_id, files_metadata=False)
            siblings = [s.rfilename for s in (info.siblings or [])]
            skip_reason = should_skip(repo_id, siblings)
            if skip_reason:
                print(f"  [skip] {skip_reason}", flush=True)
                skipped.append(
                    CardResult(repo_id=repo_id, status="skipped", notes=[skip_reason])
                )
                continue
            if args.limit is not None and processed >= args.limit:
                print("  [skip] over --limit", flush=True)
                skipped.append(
                    CardResult(
                        repo_id=repo_id, status="skipped", notes=["over --limit"]
                    )
                )
                continue
            processed += 1

            try:
                readme_path = hf_hub_download(
                    repo_id=repo_id, filename="README.md", repo_type="model"
                )
                existing_readme = Path(readme_path).read_text(encoding="utf-8")
            except EntryNotFoundError:
                existing_readme = ""
            config = load_json_from_hf(repo_id, "config.json")
            card, result = build_card(repo_id, info, existing_readme, config)
            if args.preview_dir:
                preview_path = args.preview_dir / repo_id.split("/", 1)[1] / "README.md"
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                preview_path.write_text(card, encoding="utf-8")

            if existing_readme.strip() == card.strip():
                result.status = "unchanged"
                print("  [unchanged]", flush=True)
            elif args.apply:
                commit_info = api.upload_file(
                    path_or_fileobj=card.encode("utf-8"),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=COMMIT_MSG,
                )
                result.status = "uploaded"
                result.commit_sha = getattr(commit_info, "oid", "") or ""
                print(f"  [uploaded] {result.commit_sha}", flush=True)
                time.sleep(args.sleep)
            else:
                result.status = "ready"
                print("  [ready]", flush=True)
            results.append(result)
        except (RepositoryNotFoundError, EntryNotFoundError) as exc:
            print(f"  [failed] {exc}", flush=True)
            results.append(
                CardResult(repo_id=repo_id, status="failed", notes=[str(exc)])
            )
        except Exception as exc:
            print(f"  [failed] {type(exc).__name__}: {exc}", flush=True)
            results.append(
                CardResult(
                    repo_id=repo_id,
                    status="failed",
                    notes=[f"{type(exc).__name__}: {exc}"],
                )
            )

    if args.report:
        write_report(args.report, results, skipped, args.apply)
        print(f"\n# Report written: {args.report}", flush=True)

    failed = [r for r in results if r.status == "failed"]
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
