"""One canonical call traverses every owner, and every mutation turns it red.

This verifier is deliberately cross-layer: it drives a single
``message -> function_call -> function_call_output`` conversation through the
canonical Responses mapper, the fused request preparer, the immutable
prepared-message digest, and the tensor prompt validator, in that exact order.
No existing test file owns all four layers, so the whole chain lives here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import pytest

from mlx_batch_server.responses.runtime_mapper import (
    CanonicalResponsesMapper,
    ResponsesMappingError,
)
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    RequestModality,
    RuntimeKey,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.media import (
    PreparedQwen4Message,
    bind_prepared_messages,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.media_resolver import (
    ResolvedImage,
    ResolvedMediaBundle,
    source_identity_digest,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.model.tensor import (
    _render_prepared_messages,
    _require_prepared_vision_prompt,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.request_preparation import (
    Qwen4ExpRequestPreparer,
)
from mlx_batch_server.vision.input import (
    MediaSourceField,
    MultimodalInputCapabilities,
)

CALL_ID = "call_inspect"
CALL_NAME = "inspect_region"
CALL_ARGUMENTS = '{"region":"top"}'
CALL_ITEM_ID = "fc_inspect"
RECEIPT = '{"finding":"lesion"}'

_CONVERSATION: tuple[Mapping[str, Any], ...] = (
    {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Inspect this photo."},
            {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
        ],
    },
    {
        "type": "function_call",
        "id": CALL_ITEM_ID,
        "call_id": CALL_ID,
        "name": CALL_NAME,
        "arguments": CALL_ARGUMENTS,
        "status": "completed",
    },
    {
        "type": "function_call_output",
        "call_id": CALL_ID,
        "output": RECEIPT,
    },
)


class _Projection:
    def observe(self, event: Any) -> None:
        del event

    def terminal_envelope(self) -> Mapping[str, Any]:
        return {"id": "unused", "status": "completed"}


def _resolve_runtime(**kwargs: Any) -> RuntimeKey:
    return RuntimeKey(
        model_id=kwargs["model"],
        revision=kwargs["revision"],
        adapter_path=kwargs["adapter_path"],
        draft_model_id=kwargs["draft_model_id"],
        backend=BackendKind.FUSED_MTP_MLX,
    )


class _Resolver:
    """One whole-request media bundle; it must never be reached on a refusal."""

    def __init__(self) -> None:
        self.calls = 0

    async def resolve(self, plan: Any) -> ResolvedMediaBundle:
        self.calls += 1
        items = []
        for descriptor in plan.media:
            content = f"image-{descriptor.part_index}".encode()
            items.append(
                ResolvedImage(
                    part_index=descriptor.part_index,
                    item_index=0,
                    media_type="image/png",
                    content=content,
                    width=8,
                    height=8,
                    source_digest=source_identity_digest(descriptor),
                    content_digest=hashlib.sha256(content).hexdigest(),
                )
            )
        images = tuple(items)
        return ResolvedMediaBundle(
            items=images,
            images=images,
            texts=(),
            source_bytes=sum(len(item.content) for item in images),
            materialized_image_bytes=sum(len(item.content) for item in images),
            materialized_text_bytes=0,
            total_pixels=sum(item.width * item.height for item in images),
            digest=hashlib.sha256(b"resolved-bundle").hexdigest(),
        )


class _Cancel:
    cancelled = False
    reason: str | None = None


def _map_request(
    conversation: tuple[Mapping[str, Any], ...] = _CONVERSATION,
) -> GenerationRequest:
    mapper = CanonicalResponsesMapper(
        resolve_runtime=_resolve_runtime,
        projection_factory=lambda prepared: _Projection(),
    )
    prepared = mapper.prepare(
        {"model": "flash-next", "input": list(conversation)},
        response_id="resp_seal",
        owner_id="principal:owner",
        parent_messages=(),
    )
    return prepared.request


async def _seal(request: GenerationRequest) -> Any:
    preparer = Qwen4ExpRequestPreparer(
        resolver=_Resolver(),
        capabilities=MultimodalInputCapabilities(
            frozenset({MediaSourceField.IMAGE_URL, MediaSourceField.IMAGE_BASE64})
        ),
    )
    prepared = await preparer.prepare(request, _Cancel())
    assert prepared.modality is RequestModality.VISION
    return prepared.backend_payload


def _replace_message(
    request: GenerationRequest,
    index: int,
    message: Mapping[str, Any],
) -> GenerationRequest:
    messages = list(request.messages)
    messages[index] = message
    return GenerationRequest(
        response_id=request.response_id,
        runtime=request.runtime,
        messages=tuple(messages),
        media=request.media,
        tools=request.tools,
        sampling=request.sampling,
        reasoning=request.reasoning,
        lineage=request.lineage,
        metadata=request.metadata,
    )


@pytest.mark.asyncio
async def test_one_call_traverses_mapper_preparation_seal_and_tensor_in_order() -> None:
    request = _map_request()

    # 1. Mapper: the typed call reaches the runtime with its exact identity and
    #    no assistant-visible text.
    call = request.messages[1]
    assert call["type"] == "function_call"
    assert call["role"] == "assistant"
    assert call["content"] == ()
    assert call["call_id"] == CALL_ID
    assert call["name"] == CALL_NAME
    assert call["arguments"] == CALL_ARGUMENTS
    assert call["id"] == CALL_ITEM_ID
    assert call["status"] == "completed"

    # 2. Preparation + seal: identity survives, order survives, digest binds.
    prompt = await _seal(request)
    assert [message.item_type for message in prompt.messages] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    sealed = prompt.messages[1]
    assert (
        sealed.id,
        sealed.status,
        sealed.call_id,
        sealed.name,
        sealed.arguments,
    ) == (CALL_ITEM_ID, "completed", CALL_ID, CALL_NAME, CALL_ARGUMENTS)
    assert sealed.items == ()

    # 3. Tensor: raw and sealed identities agree, so the prompt is admitted.
    assert _require_prepared_vision_prompt(request, prompt) is prompt

    # 4. Only then is the typed object rendered for the checkpoint template.
    rendered, image_count = _render_prepared_messages(prompt)
    assert image_count == 1
    assert rendered[1]["type"] == "function_call"
    assert rendered[1]["name"] == CALL_NAME
    assert rendered[1]["arguments"] == CALL_ARGUMENTS
    assert rendered[1]["content"] == ""
    assert CALL_ARGUMENTS not in rendered[0]["content"]
    assert CALL_ARGUMENTS not in rendered[2]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    (
        {"id": "fc_other"},
        {"status": "incomplete"},
        {"call_id": "call_other"},
        {"name": "inspect_other"},
        {"arguments": '{"region":"bottom"}'},
        {"role": "user"},
        {"type": "message"},
    ),
)
async def test_mutating_the_raw_call_after_the_seal_fails_before_rendering(
    mutation: dict,
) -> None:
    """The seal is signed before this; the tensor must catch the divergence."""

    request = _map_request()
    prompt = await _seal(request)
    tampered = _replace_message(request, 1, {**request.messages[1], **mutation})

    with pytest.raises(ValueError):
        _require_prepared_vision_prompt(tampered, prompt)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    (
        {"id": "fc_other"},
        {"status": "incomplete"},
        {"call_id": "call_other"},
        {"name": "inspect_other"},
        {"arguments": '{"region":"bottom"}'},
    ),
)
async def test_mutating_the_sealed_call_fails_before_rendering(
    mutation: dict,
) -> None:
    request = _map_request()
    prompt = await _seal(request)
    original = prompt.messages[1]
    swapped = PreparedQwen4Message(
        message_index=original.message_index,
        role=original.role,
        items=original.items,
        item_type=original.item_type,
        id=mutation.get("id", original.id),
        status=mutation.get("status", original.status),
        call_id=mutation.get("call_id", original.call_id),
        name=mutation.get("name", original.name),
        arguments=mutation.get("arguments", original.arguments),
    )
    messages = list(prompt.messages)
    messages[1] = swapped

    # The digest moves with the field, so the swap cannot hide under the seal.
    resealed = bind_prepared_messages(prompt, tuple(messages))
    assert resealed.content_digest != prompt.content_digest

    with pytest.raises(ValueError):
        _require_prepared_vision_prompt(request, resealed)


@pytest.mark.asyncio
async def test_reordering_the_sealed_conversation_fails_before_rendering() -> None:
    request = _map_request()
    prompt = await _seal(request)
    swapped = _replace_message(request, 1, request.messages[2])
    swapped = _replace_message(swapped, 2, request.messages[1])

    with pytest.raises(ValueError):
        _require_prepared_vision_prompt(swapped, prompt)


@pytest.mark.asyncio
async def test_arguments_never_become_visible_assistant_text() -> None:
    """The W2 stopgap rendered the call as JSON text; that path is gone."""

    request = _map_request()
    prompt = await _seal(request)
    rendered, _ = _render_prepared_messages(prompt)

    assistant_text = [
        message["content"] for message in rendered if message["role"] == "assistant"
    ]
    assert assistant_text == [""]
    for message in prompt.messages:
        for item in message.items:
            assert CALL_ARGUMENTS not in getattr(item, "text", "")


def test_a_non_terminal_call_status_is_refused_at_admission() -> None:
    conversation = list(_CONVERSATION)
    conversation[1] = {**conversation[1], "status": "in_progress"}

    with pytest.raises(ResponsesMappingError) as error:
        _map_request(tuple(conversation))

    assert error.value.param == "input[1].status"


def test_a_result_for_an_incomplete_call_is_refused_at_admission() -> None:
    conversation = list(_CONVERSATION)
    conversation[1] = {**conversation[1], "status": "incomplete"}

    with pytest.raises(ResponsesMappingError) as error:
        _map_request(tuple(conversation))

    assert error.value.param == "input[2].call_id"


def test_a_duplicated_call_identity_is_refused_at_admission() -> None:
    conversation = list(_CONVERSATION)
    conversation.insert(2, dict(conversation[1]))

    with pytest.raises(ResponsesMappingError) as error:
        _map_request(tuple(conversation))

    assert error.value.param == "input[2].call_id"


def test_a_duplicated_result_is_refused_at_admission() -> None:
    conversation = list(_CONVERSATION)
    conversation.append(dict(conversation[2]))

    with pytest.raises(ResponsesMappingError) as error:
        _map_request(tuple(conversation))

    assert error.value.param == "input[3].call_id"
