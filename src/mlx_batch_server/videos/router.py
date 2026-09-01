from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..auth.dependency import verify_auth
from .schema import VideoCapabilities, VideoGenerationRequest, VideoGenerationResponse
from .video_runtime import get_video_runtime
from .video_service import get_video_adapter

router = APIRouter(tags=["videos"])


@router.get("/v1/videos/capabilities")
async def video_capabilities(
    _auth: dict = Depends(verify_auth),
) -> VideoCapabilities:
    return get_video_adapter().capabilities()


@router.post("/v1/videos/generations")
async def create_video(
    request: VideoGenerationRequest,
    _auth: dict = Depends(verify_auth),
) -> VideoGenerationResponse:
    try:
        artifact = await get_video_runtime().generate(request)
        return VideoGenerationResponse(created=int(time.time()), data=[artifact])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.get("/v1/videos/artifacts/{artifact_id}")
async def get_video_artifact(
    artifact_id: str,
    _auth: dict = Depends(verify_auth),
) -> FileResponse:
    try:
        path = get_video_adapter().artifact_path(artifact_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video artifact not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
