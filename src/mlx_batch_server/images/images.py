import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from fastapi import APIRouter, Depends, HTTPException

from ..auth.dependency import verify_auth
from .images_service import get_images_service, run_image_operation
from .schema import (
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageObject,
    ImagePresetsResponse,
    ImagePresetSummary,
)

router = APIRouter(tags=["images"])
_process_pool: ProcessPoolExecutor | None = None


def _image_process_pool() -> ProcessPoolExecutor:
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context("spawn"),
        )
    return _process_pool


async def _run_image_process(
    operation: str,
    request: ImageGenerationRequest | ImageEditRequest,
) -> list[ImageObject]:
    payload = await asyncio.get_running_loop().run_in_executor(
        _image_process_pool(),
        run_image_operation,
        operation,
        request.model_dump(mode="json"),
    )
    return [ImageObject.model_validate(item) for item in payload]


@router.post("/images/generations")
@router.post("/v1/images/generations")
async def create_image(
    request: ImageGenerationRequest,
    _auth: dict = Depends(verify_auth),
) -> ImageGenerationResponse:
    """
    Creates an image given a prompt.
    """
    try:
        # Generate images
        images = await _run_image_process("generate", request)

        # Create response
        return ImageGenerationResponse(created=int(time.time()), data=images)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/images/edits")
@router.post("/v1/images/edits")
async def edit_image(
    request: ImageEditRequest,
    _auth: dict = Depends(verify_auth),
) -> ImageGenerationResponse:
    """Edit one source image with up to two additional references."""
    try:
        images = await _run_image_process("edit", request)
        return ImageGenerationResponse(created=int(time.time()), data=images)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.get("/images/presets")
@router.get("/v1/images/presets")
async def list_image_presets(
    _auth: dict = Depends(verify_auth),
) -> ImagePresetsResponse:
    """Return the runner-owned MLX LoRA preset catalog for clients."""
    return ImagePresetsResponse(
        data=[
            ImagePresetSummary(**preset)
            for preset in get_images_service().list_presets()
        ]
    )
