import base64
import binascii
import gc
import os
import random
import re
import tempfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.z_image import ZImageTurbo
from mflux.utils.exceptions import StopImageGenerationException
from PIL import Image

from ..utils.logger import logger
from .presets import ImageLoraPreset, ImageModelProfile, image_preset_catalog
from .schema import (
    ImageEditRequest,
    ImageGenerationRequest,
    ImageObject,
    ResponseFormat,
)

_DATA_IMAGE = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", re.DOTALL)


class MFluxImageGenerator:
    """Image generator using mflux library"""

    def __init__(
        self,
        model_version: str = "dhairyashil/FLUX.1-schnell-mflux-4bit",
        preset: ImageLoraPreset | None = None,
    ):
        self.model_version = model_version
        self.profile = image_preset_catalog().model(model_version)
        self.preset = preset

        # Initialize model instance (lazy loading)
        self._flux = None

    def _extra_base_model(self, model_name: str):
        # List of supported base models
        supported_base_models = ["schnell", "dev", "dev-fill", "dev-depth", "dev-redux"]
        base_model = None
        # Extract base_model from model_name if it contains any of the supported keywords
        model_name_lower = model_name.lower()
        for base in supported_base_models:
            if base in model_name_lower:
                base_model = base
                logger.info(
                    f"Extracted base_model '{base_model}' from model_name '{model_name}'"
                )
                break

        # If we couldn't extract a base_model, set it to None
        if not base_model:
            logger.info(
                f"Could not extract base_model from model_name '{model_name}', using None"
            )

        return base_model

    def _get_flux(self, params: dict | None = None):
        """Get or initialize the configured mflux family instance."""
        if self._flux is None:
            if self.profile:
                self._flux = self._build_profile_model(self.profile)
                return self._flux

            # Extract model name from full path
            model_name = self.model_version

            # Get base_model from params or extract from model_name
            base_model = params.get("base-model") if params else None

            # If base_model is not provided, try to extract it from model_name
            if model_name.__contains__("/") and not base_model:
                base_model = self._extra_base_model(model_name)

            # Let mflux handle model configuration
            self._flux = Flux1(
                model_config=ModelConfig.from_name(
                    model_name=model_name, base_model=base_model
                ),
                quantize=(params or {}).get("quantize"),
                model_path=(params or {}).get("model_path"),
                lora_paths=params.get("lora-paths") if params else None,
                lora_scales=params.get("lora-scales") if params else None,
            )

        return self._flux

    def _build_profile_model(self, profile: ImageModelProfile):
        model_config = getattr(ModelConfig, profile.config)()
        common = {
            "model_config": model_config,
            "model_path": profile.model_path,
            "quantize": profile.quantize,
        }
        if profile.family == "qwen-image-edit":
            return QwenImageEdit(
                **common,
                lora_paths=list(self.preset.lora_paths) if self.preset else None,
                lora_scales=list(self.preset.lora_scales) if self.preset else None,
            )
        if profile.family == "flux2-klein":
            return Flux2Klein(**common)
        if profile.family == "z-image-turbo":
            return ZImageTurbo(**common)
        raise ValueError(f"Unsupported mflux image family '{profile.family}'")

    def _parse_size(self, size_str: str) -> tuple[int, int]:
        """Parse size string to width and height"""
        try:
            width, height = map(int, size_str.split("x"))
            return width, height
        except (ValueError, AttributeError):
            return 1024, 1024

    def generate(
        self,
        request: ImageGenerationRequest,
        output_path: str,
        **extra_params,
    ) -> Image.Image:
        """Generate image using mflux"""
        # Parse image dimensions
        width, height = self._parse_size(request.size or "1024x1024")

        # Get extra parameters from request
        request_extra_params = request.get_extra_params()

        # Merge all extra parameters, with passed extra_params taking precedence
        all_extra_params = {**request_extra_params, **extra_params}
        logger.info(f"all_extra_params: {all_extra_params}")

        # Generate random seed if not specified
        seed = all_extra_params.pop("seed", random.randint(0, 2**32 - 1))

        # Get or initialize model instance
        flux = self._get_flux(all_extra_params)

        if self.profile:
            steps = all_extra_params.pop("steps", self.profile.steps)
            guidance = all_extra_params.pop("guidance", self.profile.guidance)
            kwargs = {
                "seed": seed,
                "prompt": request.prompt,
                "num_inference_steps": steps,
                "height": height,
                "width": width,
            }
            if guidance is not None:
                kwargs["guidance"] = guidance
            image = flux.generate_image(**kwargs)
            self._save_image(image, output_path)
            return image

        # Generate image
        low_memory_mode = all_extra_params.get("low_arm", True)
        memory_saver = None
        if low_memory_mode:
            memory_saver = MemorySaver(model=flux, keep_transformer=seed > 1)
            CallbackRegistry().register(memory_saver)

        try:
            # Generate image
            image = flux.generate_image(
                seed=seed,
                prompt=request.prompt,
                num_inference_steps=all_extra_params.pop("steps", 4),
                height=height,
                width=width,
                guidance=all_extra_params.pop("guidance", 4.0),
            )

            # Save image
            self._save_image(image, output_path)
            return image
        except StopImageGenerationException as e:
            raise Exception(f"Image generation interrupted: {e!s}") from e
        except Exception as e:
            raise Exception(f"Error generating image: {e!s}") from e
        finally:
            if memory_saver:
                print(memory_saver.memory_stats())

    def edit(
        self,
        request: ImageEditRequest,
        image_paths: list[str],
        output_path: str,
    ) -> Image.Image:
        if not self.profile or self.profile.family not in {
            "qwen-image-edit",
            "flux2-klein",
        }:
            raise ValueError(
                f"Model '{self.model_version}' does not support image edits"
            )

        seed = (
            request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        )
        width, height = self._edit_size(request, image_paths[0])
        prompt = request.prompt
        if self.preset and self.profile.family == "qwen-image-edit":
            prompt = f"{prompt}\n\n{self.preset.prompt_guard}"
        steps = (
            request.steps
            or (self.preset.steps if self.preset else None)
            or self.profile.steps
        )
        guidance = (
            request.guidance if request.guidance is not None else self.profile.guidance
        )

        if self.profile.family == "flux2-klein":
            if not isinstance(self._flux, Flux2KleinEdit):
                self._flux = Flux2KleinEdit(
                    model_config=getattr(ModelConfig, self.profile.config)(),
                    model_path=self.profile.model_path,
                    quantize=self.profile.quantize,
                )
            model = self._flux
        else:
            model = self._get_flux({})
        if model is None:
            raise RuntimeError(f"Model '{self.model_version}' did not initialize")

        kwargs = {
            "seed": seed,
            "prompt": prompt,
            "image_paths": image_paths,
            "num_inference_steps": steps,
            "height": height,
            "width": width,
        }
        if guidance is not None:
            kwargs["guidance"] = guidance
        try:
            image = model.generate_image(**kwargs)
            self._save_image(image, output_path)
            return image
        except StopImageGenerationException as error:
            raise Exception(f"Image edit interrupted: {error!s}") from error
        except Exception as error:
            raise Exception(f"Error editing image: {error!s}") from error

    @staticmethod
    def _save_image(image, output_path: str) -> None:
        if isinstance(image, Image.Image):
            image.save(output_path)
        else:
            image.save(path=output_path, export_json_metadata=False)

    def _edit_size(self, request: ImageEditRequest, first_path: str) -> tuple[int, int]:
        if request.size:
            return self._parse_size(request.size.value)
        with Image.open(first_path) as source:
            return source.size


class ImagesService:
    def __init__(self):
        # Use system temporary directory
        self.output_dir = Path(tempfile.gettempdir()) / "mlx_batch_server" / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache loaded generator instances
        self._generator_cache: dict[str, MFluxImageGenerator] = {}

    def load_model(self, model_name: str) -> bool:
        """Preload an image generation model. Returns True if it was already loaded."""
        generator = self._get_generator(model_name=model_name)
        already_loaded = generator._flux is not None
        # Use an empty params dict to satisfy _get_flux internals.
        generator._get_flux({})
        return already_loaded

    def unload_model(self, model_name: str) -> bool:
        """Unload a specific image model. Returns True if it was loaded."""
        keys = [
            key for key in self._generator_cache if key.split(":", 1)[0] == model_name
        ]
        for key in keys:
            self._generator_cache.pop(key)._flux = None
        if keys:
            gc.collect()
            mx.clear_cache()
        return bool(keys)

    def clear_models(self) -> list[str]:
        """Unload all image models and return the unloaded model IDs."""
        unloaded = list(
            dict.fromkeys(key.split(":", 1)[0] for key in self._generator_cache)
        )
        for generator in self._generator_cache.values():
            generator._flux = None
        self._generator_cache.clear()
        if unloaded:
            gc.collect()
            mx.clear_cache()
        return unloaded

    def _get_generator(
        self,
        model_name: str,
        preset_id: str | None = None,
    ) -> MFluxImageGenerator:
        """Get or create image generator instance"""
        profile = image_preset_catalog().model(model_name)
        preset = (
            image_preset_catalog().preset(preset_id)
            if profile and profile.family == "qwen-image-edit"
            else None
        )
        cache_key = f"{model_name}:{preset.id if preset else '-'}"
        if cache_key not in self._generator_cache:
            # Image models are large, and Qwen's load peak is especially high.
            # Keep exactly one resident image family/preset in this process.
            # This also prevents a draft -> high request from retaining the
            # FLUX.2 transformer while Qwen is being materialized.
            self._evict_generators_except(cache_key)
            self._generator_cache[cache_key] = MFluxImageGenerator(
                model_version=model_name,
                preset=preset,
            )
        return self._generator_cache[cache_key]

    def _evict_generators_except(self, keep_key: str) -> None:
        evicted = False
        for key in list(self._generator_cache):
            if key == keep_key:
                continue
            self._generator_cache.pop(key)._flux = None
            evicted = True
        if evicted:
            gc.collect()
            mx.clear_cache()

    def _get_output_path(self, uid: str) -> str:
        """Generate unique output path for image"""
        return str(self.output_dir / f"{uid}.png")

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _cleanup_image(self, image_path: str):
        """Clean up temporary image file"""
        try:
            Path(image_path).unlink(missing_ok=True)
        except OSError as error:
            logger.warning(f"Error cleaning up image {image_path}: {error!s}")

    def generate_images(
        self,
        request: ImageGenerationRequest,
    ) -> list[ImageObject]:
        """Generate images based on the request"""
        generated_images = []
        generator = self._get_generator(model_name=request.model or "flux")

        for i in range(1 if request.n is None else request.n):
            # Generate unique identifier for this image
            uid = f"{int(time.time())}_{i}"
            output_path = self._get_output_path(uid)

            try:
                # Generate the image
                generator.generate(
                    request=request, output_path=output_path, low_memory_mode=True
                )

                # Create response object based on format
                image_object = ImageObject(revised_prompt=request.prompt)

                # Response format
                if request.response_format == ResponseFormat.B64_JSON:
                    image_object.b64_json = self._image_to_base64(output_path)
                else:  # URL format
                    # This is a runner-owned local artifact URI, never a user host.
                    # nosemgrep: python.django.security.injection.tainted-url-host.tainted-url-host  # noqa: ERA001
                    image_object.url = f"file://{output_path}"

                generated_images.append(image_object)

            except Exception as e:
                raise Exception(f"Error generating image: {e!s}") from e
            finally:
                # Clean up temporary file if using base64 format
                if request.response_format == ResponseFormat.B64_JSON:
                    self._cleanup_image(output_path)

        return generated_images

    def edit_images(self, request: ImageEditRequest) -> list[ImageObject]:
        generator = self._get_generator(request.model, request.preset)
        generated_images: list[ImageObject] = []

        with TemporaryDirectory(prefix="mlx-image-edit-") as input_dir:
            image_paths = [
                self._materialize_data_url(url, Path(input_dir), index)
                for index, url in enumerate(request.input_urls())
            ]
            for index in range(request.n):
                uid = f"{time.time_ns()}_{index}"
                output_path = self._get_output_path(uid)
                try:
                    generator.edit(request, image_paths, output_path)
                    image_object = ImageObject(revised_prompt=request.prompt)
                    if request.response_format == ResponseFormat.URL:
                        # This is a runner-owned local artifact URI, never a user host.
                        # nosemgrep: python.django.security.injection.tainted-url-host.tainted-url-host  # noqa: ERA001
                        image_object.url = f"file://{output_path}"
                    else:
                        image_object.b64_json = self._image_to_base64(output_path)
                    generated_images.append(image_object)
                finally:
                    if request.response_format == ResponseFormat.B64_JSON:
                        self._cleanup_image(output_path)
        return generated_images

    @staticmethod
    def _materialize_data_url(url: str, directory: Path, index: int) -> str:
        match = _DATA_IMAGE.match(url)
        if not match:
            raise ValueError("MLX image edit accepts data:image base64 inputs only")
        mime, encoded = match.groups()
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Invalid base64 image input") from error
        if len(payload) > 20 * 1024 * 1024:
            raise ValueError("Image input exceeds 20 MiB")
        suffix = ".png" if mime == "image/png" else ".jpg"
        path = directory / f"source-{index}{suffix}"
        path.write_bytes(payload)
        try:
            with Image.open(path) as source:
                source.verify()
        except Exception as error:
            path.unlink(missing_ok=True)
            raise ValueError("Image input is not a valid bitmap") from error
        return str(path)

    def list_presets(self) -> list[dict[str, str]]:
        return image_preset_catalog().public_presets()


_images_service: ImagesService | None = None


def get_images_service() -> ImagesService:
    """Return a shared images service instance."""
    global _images_service
    if _images_service is None:
        _images_service = ImagesService()
    return _images_service


def run_image_worker_operation(
    operation: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """Process-worker entrypoint keeping MLX on that process's main thread."""
    service = get_images_service()
    if operation == "generate":
        generation_request = ImageGenerationRequest.model_validate(payload)
        images = service.generate_images(generation_request)
    elif operation == "edit":
        edit_request = ImageEditRequest.model_validate(payload)
        images = service.edit_images(edit_request)
    elif operation == "load":
        model_name = str(payload["model"])
        return {
            "already_loaded": service.load_model(model_name),
            "worker_pid": os.getpid(),
        }
    elif operation == "unload":
        model_name = str(payload["model"])
        return {
            "unloaded": service.unload_model(model_name),
            "worker_pid": os.getpid(),
        }
    elif operation == "clear":
        return {
            "unloaded_models": service.clear_models(),
            "worker_pid": os.getpid(),
        }
    else:
        raise ValueError(f"Unknown image operation '{operation}'")
    return {
        "data": [image.model_dump() for image in images],
        "worker_pid": os.getpid(),
    }
