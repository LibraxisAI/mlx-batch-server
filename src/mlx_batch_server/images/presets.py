import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageModelProfile:
    id: str
    label: str
    family: str
    model_path: str
    config: str
    steps: int
    guidance: float | None
    quantize: int | None = None


@dataclass(frozen=True)
class ImageLoraPreset:
    id: str
    label: str
    description: str
    prompt_guard: str
    lora_paths: tuple[str, ...]
    lora_scales: tuple[float, ...]
    steps: int | None = None


class ImagePresetCatalog:
    def __init__(self, payload: dict[str, Any]):
        self.models = {
            model_id: ImageModelProfile(id=model_id, **model)
            for model_id, model in payload["models"].items()
        }
        self.presets = {
            preset["id"]: ImageLoraPreset(
                id=preset["id"],
                label=preset["label"],
                description=preset["description"],
                prompt_guard=preset["prompt_guard"],
                lora_paths=tuple(preset.get("lora_paths", [])),
                lora_scales=tuple(preset.get("lora_scales", [])),
                steps=preset.get("steps"),
            )
            for preset in payload["presets"]
        }

    def model(self, model_id: str) -> ImageModelProfile | None:
        aliases = {
            "z-image-turbo": "mflux-z-image-turbo",
            "flux2-klein-4b": "mflux-flux2-klein-draft",
            "Qwen/Qwen-Image-Edit-2511": "mflux-qwen-image-edit",
        }
        return self.models.get(aliases.get(model_id, model_id))

    def preset(self, preset_id: str | None) -> ImageLoraPreset:
        selected = preset_id or "clean"
        try:
            return self.presets[selected]
        except KeyError as error:
            raise ValueError(f"Unknown MLX LoRA preset '{selected}'") from error

    def public_presets(self) -> list[dict[str, str]]:
        return [
            {
                "id": preset.id,
                "label": preset.label,
                "description": preset.description,
            }
            for preset in self.presets.values()
        ]


@lru_cache(maxsize=1)
def image_preset_catalog() -> ImagePresetCatalog:
    path = Path(__file__).with_name("image-presets.json")
    return ImagePresetCatalog(json.loads(path.read_text(encoding="utf-8")))
