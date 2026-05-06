import json
from collections.abc import Generator

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_batch_server.auth.dependency import verify_auth
from mlx_batch_server.chat.mlx.chat_generator import ChatGenerator
from mlx_batch_server.chat.mlx.runtime_policy import endpoint_runtime_session
from mlx_batch_server.chat.openai.openai_adapter import OpenAIAdapter
from mlx_batch_server.chat.openai.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

router = APIRouter(tags=["chat—completions"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    _auth: dict = Depends(verify_auth),
) -> JSONResponse | StreamingResponse:
    """Create a chat completion"""
    extra_params = request.get_extra_params()
    draft_model_id = extra_params.get("draft_model_id") or extra_params.get(
        "draft_model"
    )
    adapter_path = extra_params.get("adapter_path")
    if not request.stream:
        async with endpoint_runtime_session(
            model_id=request.model,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        ):
            text_model = _create_text_model(
                request.model,
                adapter_path,
                draft_model_id,
            )
            completion = text_model.generate(request)
            return JSONResponse(content=completion.model_dump(exclude_none=True))

    async def event_generator() -> Generator[str, None, None]:
        async with endpoint_runtime_session(
            model_id=request.model,
            adapter_path=adapter_path,
            draft_model_id=draft_model_id,
        ):
            text_model = _create_text_model(
                request.model,
                adapter_path,
                draft_model_id,
            )
            for chunk in text_model.generate_stream(request):
                yield f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _create_text_model(
    model_id: str,
    adapter_path: str | None = None,
    draft_model: str | None = None,
) -> OpenAIAdapter:
    """Create a text model based on the model parameters.

    Uses the shared wrapper cache to get or create ChatGenerator instance.
    This avoids expensive model reloading when the same model configuration
    is used across different requests or API endpoints.
    """
    # Get cached or create new ChatGenerator
    wrapper = ChatGenerator.get_or_create(
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model,
    )

    # Create OpenAIAdapter with the cached wrapper directly
    return OpenAIAdapter(wrapper=wrapper)


# Legacy caching variables removed - now using shared wrapper_cache
# This eliminates duplicate caching logic and enables sharing between endpoints
