"""RED contracts for pre-admission Qwen4Exp request preparation.

Compile Embargo is HOLD: this file is intentionally not executed yet.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading

import pytest

from mlx_batch_server.runtime.contracts import (
    GenerationRequest,
    RequestModality,
    RuntimeKey,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.media import (
    PreparedMediaItem,
    PreparedTextItem,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.media_resolver import (
    ResolvedImage,
    ResolvedMediaBundle,
    source_identity_digest,
)
from mlx_batch_server.runtime.fusion.qwen4_exp.request_preparation import (
    Qwen4ExpRequestPreparationError,
    Qwen4ExpRequestPreparer,
)
from mlx_batch_server.vision.input import (
    ImageInput,
    MediaSourceField,
    MultimodalInputCapabilities,
)


class _Cancel:
    def __init__(self) -> None:
        self.cancelled = False
        self.reason: str | None = None

    def cancel(self, reason: str) -> bool:
        if self.cancelled:
            return False
        self.cancelled = True
        self.reason = reason
        return True


class _Resolver:
    def __init__(self) -> None:
        self.calls = 0
        self.thread_ids: list[int] = []

    async def resolve(self, plan):
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        items = []
        for descriptor in plan.media:
            assert isinstance(descriptor, ImageInput)
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


def _preparer(resolver: _Resolver) -> Qwen4ExpRequestPreparer:
    return Qwen4ExpRequestPreparer(
        resolver=resolver,
        capabilities=MultimodalInputCapabilities(
            frozenset({MediaSourceField.IMAGE_BASE64})
        ),
    )


@pytest.mark.asyncio
async def test_mixed_messages_preserve_roles_and_global_part_order() -> None:
    runtime = RuntimeKey("flash")
    request = GenerationRequest(
        response_id="resp_mixed",
        runtime=runtime,
        messages=(
            {
                "role": "user",
                "content": (
                    {"type": "input_text", "text": "before"},
                    {"type": "input_text", "text": "after"},
                ),
            },
            {
                "role": "developer",
                "content": ({"type": "input_text", "text": "tail"},),
            },
        ),
        media=(
            {
                "type": "input_image",
                "image_base64": "aW1hZ2U=",
                "_role": "user",
                "_message_index": 0,
                "_content_index": 1,
            },
            {
                "type": "input_image",
                "image_base64": "aW1hZ2Uy",
                "_role": "developer",
                "_message_index": 1,
                "_content_index": 0,
            },
        ),
    )
    resolver = _Resolver()

    prepared = await _preparer(resolver).prepare(request, _Cancel())

    assert prepared.request is request
    assert prepared.modality is RequestModality.VISION
    prompt = prepared.backend_payload
    assert prompt is not None
    assert [message.role for message in prompt.messages] == ["user", "developer"]
    assert [type(item) for item in prompt.messages[0].items] == [
        PreparedTextItem,
        PreparedMediaItem,
        PreparedTextItem,
    ]
    assert [type(item) for item in prompt.messages[1].items] == [
        PreparedMediaItem,
        PreparedTextItem,
    ]
    assert [item.part_index for item in prompt.items] == [0, 1, 2, 3, 4]
    assert resolver.calls == 1
    assert resolver.thread_ids == [threading.get_ident()]


@pytest.mark.asyncio
async def test_vision_preparation_seals_function_output_identity() -> None:
    output = '{"canvas":"ready"}'
    request = GenerationRequest(
        response_id="resp_tool_vision",
        runtime=RuntimeKey("flash"),
        messages=(
            {
                "type": "function_call_output",
                "role": "tool",
                "call_id": "call_inspect_canvas",
                "output": output,
                "content": ({"type": "input_text", "text": output},),
            },
            {
                "type": "message",
                "role": "user",
                "content": ({"type": "input_text", "text": "Inspect this."},),
            },
        ),
        media=(
            {
                "type": "input_image",
                "image_base64": "aW1hZ2U=",
                "_role": "user",
                "_message_index": 1,
                "_content_index": 1,
            },
        ),
    )

    prepared = await _preparer(_Resolver()).prepare(request, _Cancel())

    prompt = prepared.backend_payload
    receipt = prompt.messages[0]
    assert receipt.item_type == "function_call_output"
    assert receipt.role == "tool"
    assert receipt.call_id == "call_inspect_canvas"
    assert receipt.output == output
    assert receipt.items[0].text == output
    assert receipt.is_error is None
    assert prompt.messages[1].item_type == "message"


@pytest.mark.asyncio
async def test_vision_preparation_seals_is_error_and_nested_tool_image() -> None:
    output = "see photo"
    success_request = GenerationRequest(
        response_id="resp_tool_image_ok",
        runtime=RuntimeKey("flash"),
        messages=(
            {
                "type": "function_call_output",
                "role": "tool",
                "call_id": "toolu_1",
                "output": output,
                "is_error": False,
                "content": ({"type": "input_text", "text": output},),
            },
        ),
        media=(
            {
                "type": "input_image",
                "image_base64": "aW1hZ2U=",
                "_role": "tool",
                "_message_index": 0,
                "_content_index": 1,
            },
        ),
    )
    error_request = GenerationRequest(
        response_id="resp_tool_image_err",
        runtime=RuntimeKey("flash"),
        messages=(
            {
                "type": "function_call_output",
                "role": "tool",
                "call_id": "toolu_1",
                "output": output,
                "is_error": True,
                "content": ({"type": "input_text", "text": output},),
            },
        ),
        media=success_request.media,
    )

    success = await _preparer(_Resolver()).prepare(success_request, _Cancel())
    error = await _preparer(_Resolver()).prepare(error_request, _Cancel())

    success_receipt = success.backend_payload.messages[0]
    error_receipt = error.backend_payload.messages[0]
    assert success_receipt.is_error is False
    assert error_receipt.is_error is True
    assert success_receipt.output == output
    assert error_receipt.output == output
    assert success_receipt.call_id == "toolu_1"
    assert [type(item) for item in success_receipt.items] == [
        PreparedTextItem,
        PreparedMediaItem,
    ]
    assert (
        success.backend_payload.content_digest != error.backend_payload.content_digest
    )


@pytest.mark.asyncio
async def test_text_only_is_pass_through_and_never_calls_resolver() -> None:
    request = GenerationRequest(
        response_id="resp_text",
        runtime=RuntimeKey("flash"),
        messages=({"role": "user", "content": "hello"},),
    )
    resolver = _Resolver()

    prepared = await _preparer(resolver).prepare(request, _Cancel())

    assert prepared.request is request
    assert prepared.modality is RequestModality.TEXT
    assert prepared.backend_payload is None
    assert resolver.calls == 0


@pytest.mark.asyncio
async def test_cancelled_request_never_enters_media_resolution() -> None:
    request = GenerationRequest(
        response_id="resp_cancelled",
        runtime=RuntimeKey("flash"),
        messages=({"role": "user", "content": ()},),
        media=(
            {
                "type": "input_image",
                "image_base64": "aW1hZ2U=",
                "_role": "user",
                "_message_index": 0,
                "_content_index": 0,
            },
        ),
    )
    resolver = _Resolver()
    cancel = _Cancel()
    cancel.cancel("client cancelled")

    with pytest.raises(asyncio.CancelledError):
        await _preparer(resolver).prepare(request, cancel)

    assert resolver.calls == 0


@pytest.mark.asyncio
async def test_raw_media_without_mapper_provenance_fails_closed() -> None:
    request = GenerationRequest(
        response_id="resp_unsealed",
        runtime=RuntimeKey("flash"),
        messages=({"role": "user", "content": ()},),
        media=({"type": "input_image", "image_base64": "aW1hZ2U="},),
    )

    with pytest.raises(Qwen4ExpRequestPreparationError, match="_message_index"):
        await _preparer(_Resolver()).prepare(request, _Cancel())


def _photo_plus_tool_messages(arguments: str = '{"region":"top"}') -> tuple[dict, ...]:
    """The W2 continuation baton: photo, typed call, typed result, next turn."""

    return (
        {
            "type": "message",
            "role": "user",
            "content": ({"type": "input_text", "text": "Inspect this photo."},),
        },
        {
            "type": "function_call",
            "role": "assistant",
            "id": "fc_1",
            "call_id": "call_inspect",
            "name": "inspect_region",
            "arguments": arguments,
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "role": "tool",
            "call_id": "call_inspect",
            "output": '{"finding":"lesion"}',
            "content": ({"type": "input_text", "text": '{"finding":"lesion"}'},),
        },
        {
            "type": "message",
            "role": "user",
            "content": ({"type": "input_text", "text": "What now?"},),
        },
    )


def _photo_plus_tool_request(**overrides) -> GenerationRequest:
    messages = overrides.pop("messages", _photo_plus_tool_messages())
    return GenerationRequest(
        response_id="resp_photo_tool",
        runtime=RuntimeKey("flash"),
        messages=messages,
        media=(
            {
                "type": "input_image",
                "image_base64": "aW1hZ2U=",
                "_role": "user",
                "_message_index": 0,
                "_content_index": 1,
            },
        ),
        **overrides,
    )


@pytest.mark.asyncio
async def test_photo_plus_tool_continuation_reaches_preparation_as_a_typed_call() -> (
    None
):
    """The admitted call stays a typed call — never an ordinary text message."""

    prepared = await _preparer(_Resolver()).prepare(
        _photo_plus_tool_request(), _Cancel()
    )
    prompt = prepared.backend_payload
    assert prompt is not None

    assert [message.item_type for message in prompt.messages] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert [message.role for message in prompt.messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    call = prompt.messages[1]
    assert call.call_id == "call_inspect"
    assert call.name == "inspect_region"
    assert call.arguments == '{"region":"top"}'
    assert call.items == ()
    assert call.output is None
    assert prompt.messages[2].call_id == "call_inspect"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"call_id": ""}, "call_id must not be empty"),
        ({"name": "   "}, "name must not be empty"),
        ({"arguments": {"region": "top"}}, "arguments must be text"),
        ({"role": "user"}, "assistant role"),
        ({"output": "leaked"}, "belongs only to function_call_output"),
    ),
)
async def test_malformed_call_identity_is_refused_not_degraded(
    mutation: dict,
    match: str,
) -> None:
    messages = list(_photo_plus_tool_messages())
    messages[1] = {**messages[1], **mutation}
    with pytest.raises(Qwen4ExpRequestPreparationError, match=match):
        await _preparer(_Resolver()).prepare(
            _photo_plus_tool_request(messages=tuple(messages)), _Cancel()
        )


@pytest.mark.asyncio
async def test_orphan_tool_result_is_refused_when_typed_calls_are_present() -> None:
    messages = list(_photo_plus_tool_messages())
    messages[2] = {**messages[2], "call_id": "call_nobody_made"}
    with pytest.raises(
        Qwen4ExpRequestPreparationError,
        match="no preceding function_call",
    ):
        await _preparer(_Resolver()).prepare(
            _photo_plus_tool_request(messages=tuple(messages)), _Cancel()
        )


@pytest.mark.asyncio
async def test_a_typed_call_may_not_own_media_or_message_content() -> None:
    messages = list(_photo_plus_tool_messages())
    messages[1] = {
        **messages[1],
        "content": ({"type": "input_text", "text": "smuggled"},),
    }
    with pytest.raises(
        Qwen4ExpRequestPreparationError,
        match="cannot carry message content",
    ):
        await _preparer(_Resolver()).prepare(
            _photo_plus_tool_request(messages=tuple(messages)), _Cancel()
        )

    request = GenerationRequest(
        response_id="resp_photo_tool",
        runtime=RuntimeKey("flash"),
        messages=_photo_plus_tool_messages(),
        media=(
            {
                "type": "input_image",
                "image_base64": "aW1hZ2U=",
                "_role": "assistant",
                "_message_index": 1,
                "_content_index": 0,
            },
        ),
    )
    with pytest.raises(Qwen4ExpRequestPreparationError, match="cannot own media"):
        await _preparer(_Resolver()).prepare(request, _Cancel())
