import time

from fastapi import APIRouter, Depends, HTTPException

from ..auth.dependency import verify_auth
from .image_runtime import get_image_runtime_pool
from .images_service import get_images_service
from .schema import (
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePresetsResponse,
    ImagePresetSummary,
)

router = APIRouter(tags=["images"])


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
        images = await get_image_runtime_pool().generate(request)

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
        images = await get_image_runtime_pool().edit(request)
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
