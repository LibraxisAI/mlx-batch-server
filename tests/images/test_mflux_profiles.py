import base64
from pathlib import Path

import pytest
from PIL import Image

from mlx_batch_server.images.images_service import ImagesService, MFluxImageGenerator
from mlx_batch_server.images.presets import image_preset_catalog
from mlx_batch_server.images.schema import ImageEditRequest


def tiny_data_url() -> str:
    payload = base64.b64encode(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    ).decode()
    return f"data:image/png;base64,{payload}"


def test_catalog_owns_three_model_lanes_and_lora_presets():
    catalog = image_preset_catalog()

    assert catalog.model("mflux-z-image-turbo").steps == 9
    assert catalog.model("mflux-flux2-klein-draft").steps == 4
    assert catalog.model("z-image-turbo").family == "z-image-turbo"
    assert catalog.model("flux2-klein-4b").family == "flux2-klein"
    assert catalog.model("Qwen/Qwen-Image-Edit-2511").family == "qwen-image-edit"
    qwen = catalog.model("mflux-qwen-image-edit")
    assert qwen.quantize == 8
    assert qwen.steps == 30
    assert catalog.preset("lightning").steps == 8
    assert catalog.preset("lightning").lora_paths == (
        "lightx2v/Qwen-Image-Lightning:Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors",
    )
    assert catalog.preset("fixbody").lora_paths == (
        "nbeerbower/FIXBODYr128-QwenImageEdit2509:pytorch_lora_weights.safetensors",
    )
    assert catalog.preset("angle").lora_paths == (
        "dx8152/Qwen-Edit-2509-Multiple-angles:镜头转换.safetensors",
    )


def test_edit_request_accepts_studio_single_and_multi_image_shapes():
    single = ImageEditRequest(prompt="edit", image={"url": tiny_data_url()})
    assert len(single.input_urls()) == 1

    multi = ImageEditRequest(
        prompt="edit",
        images=[{"url": tiny_data_url()}, {"url": tiny_data_url()}],
    )
    assert len(multi.input_urls()) == 2

    studio = ImageEditRequest(
        prompt="edit",
        image={"url": tiny_data_url()},
        lora_preset="angle",
    )
    assert studio.preset == "angle"

    with pytest.raises(ValueError, match="at most three"):
        ImageEditRequest(
            prompt="edit",
            images=[{"url": tiny_data_url()} for _ in range(4)],
        )


def test_data_url_materialization_is_bounded_and_verified(tmp_path: Path):
    path = ImagesService._materialize_data_url(tiny_data_url(), tmp_path, 0)
    with Image.open(path) as image:
        assert image.size == (1, 1)

    with pytest.raises(ValueError, match="data:image"):
        ImagesService._materialize_data_url(
            "https://example.test/image.png", tmp_path, 1
        )


def test_image_service_keeps_only_one_resident_profile():
    service = ImagesService()

    service._get_generator("z-image-turbo")
    service._get_generator("flux2-klein-4b")

    assert list(service._generator_cache) == ["flux2-klein-4b:-"]


def test_qwen_prompt_guard_and_profile_defaults_reach_model(
    tmp_path: Path, monkeypatch
):
    generator = MFluxImageGenerator(
        "mflux-qwen-image-edit",
        image_preset_catalog().preset("clean"),
    )
    seen = {}

    class FakeModel:
        def generate_image(self, **kwargs):
            seen.update(kwargs)
            return Image.new("RGB", (8, 8), "black")

    monkeypatch.setattr(generator, "_get_flux", lambda _params: FakeModel())
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "white").save(source)
    output = tmp_path / "output.png"
    request = ImageEditRequest(
        prompt="change the light", image={"url": tiny_data_url()}
    )

    generator.edit(request, [str(source)], str(output))

    assert "Preserve the exact person" in seen["prompt"]
    assert seen["num_inference_steps"] == 30
    assert seen["guidance"] == 2.5
    assert output.exists()
