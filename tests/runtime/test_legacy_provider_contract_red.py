"""RED contracts for the concrete, cache-borrowing legacy provider."""

from __future__ import annotations

import ast
import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from mlx_batch_server.runtime.backends.legacy_mlx import LegacyPortContractError
from mlx_batch_server.runtime.backends.legacy_provider import CachedLegacyPortProvider
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    LoadConfig,
    ModelSpec,
    RuntimeKey,
)
from mlx_batch_server.runtime.events import (
    ContentPartCompleted,
    ContentPartStarted,
    OutputItemCompleted,
    OutputItemStarted,
    ReasoningCompleted,
    ReasoningDelta,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    UsageUpdate,
)

RUNTIME = RuntimeKey(model_id="local/legacy", backend=BackendKind.LEGACY_MLX)


class _Model:
    supports_multimodal = True


class _Wrapper:
    model = _Model()


class _Cache:
    def __init__(self) -> None:
        self.wrapper = _Wrapper()
        self.calls: list[dict[str, Any]] = []
        self.loaded = True
        self.unload_calls = 0

    def get_wrapper(self, model_id: str, **kwargs: Any) -> _Wrapper:
        self.calls.append({"model_id": model_id, **kwargs})
        return self.wrapper

    def is_runtime_loaded(self, model_id: str, **kwargs: Any) -> bool:
        self.calls.append({"is_loaded": model_id, **kwargs})
        return self.loaded

    def get_cache_info(self) -> Mapping[str, Any]:
        return {"cache_size": 1}

    def unload_model(self, *_args: Any, **_kwargs: Any) -> bool:
        self.unload_calls += 1
        return True


class _StreamFactory:
    def __init__(self, events: tuple[Mapping[str, Any], ...]) -> None:
        self.events = events
        self.calls: list[tuple[Any, Mapping[str, Any]]] = []
        self.closed = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False

    def __call__(
        self, cache: Any, payload: Mapping[str, Any]
    ) -> AsyncIterator[Mapping[str, Any]]:
        self.calls.append((cache, payload))

        async def stream() -> AsyncIterator[Mapping[str, Any]]:
            try:
                if self.block:
                    await self.release.wait()
                for event in self.events:
                    yield event
            finally:
                self.closed.set()

        return stream()


def _events() -> tuple[Mapping[str, Any], ...]:
    return (
        {"type": "response.created", "response": {"id": "foreign"}},
        {"type": "response.in_progress", "response": {"id": "foreign"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "rs_1", "type": "reasoning"},
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "item_id": "rs_1",
            "delta": "think",
        },
        {
            "type": "response.reasoning_summary_text.done",
            "output_index": 0,
            "item_id": "rs_1",
            "text": "think",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"id": "rs_1", "type": "reasoning"},
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"id": "msg_1", "type": "message"},
        },
        {
            "type": "response.content_part.added",
            "output_index": 1,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        },
        {
            "type": "response.output_text.delta",
            "output_index": 1,
            "content_index": 0,
            "delta": "answer",
        },
        {
            "type": "response.output_text.done",
            "output_index": 1,
            "content_index": 0,
            "text": "answer",
        },
        {
            "type": "response.content_part.done",
            "output_index": 1,
            "content_index": 0,
            "part": {"type": "output_text", "text": "answer"},
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {"id": "msg_1", "type": "message"},
        },
        {
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "inspect",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 2,
            "item_id": "fc_1",
            "delta": '{"x":1}',
        },
        {
            "type": "response.function_call_arguments.done",
            "output_index": 2,
            "item_id": "fc_1",
            "call_id": "call_1",
            "name": "inspect",
            "arguments": '{"x":1}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 2,
            "item": {"id": "fc_1", "type": "function_call"},
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        },
    )


def _request(*, media: bool = False) -> GenerationRequest:
    return GenerationRequest(
        response_id="resp_local",
        runtime=RUNTIME,
        messages=(
            {
                "type": "function_call_output",
                "role": "tool",
                "call_id": "call_parent",
                "output": "receipt",
                "content": ({"type": "input_text", "text": "receipt"},),
            },
            {
                "type": "message",
                "role": "user",
                "content": ({"type": "input_text", "text": "inspect"},),
            },
        ),
        media=(
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,eA==",
                "_role": "user",
                "_message_index": 1,
                "_content_index": 1,
            },
        )
        if media
        else (),
        tools=({"type": "function", "name": "inspect", "parameters": {}},),
        sampling={"max_output_tokens": 32, "temperature": 0.2},
    )


async def _collect(port: Any, request: GenerationRequest) -> list[Any]:
    return [event async for event in port.events(request, object())]


@pytest.mark.asyncio
async def test_borrows_exact_cache_and_translates_complete_nonterminal_lifecycle() -> (
    None
):
    cache = _Cache()
    stream = _StreamFactory(_events())
    port = await CachedLegacyPortProvider(cache=cache, stream_factory=stream).acquire(
        RUNTIME, LoadConfig(max_admitted_requests=2)
    )

    observed = await _collect(port, _request(media=True))

    assert cache.calls[0] == {
        "model_id": "local/legacy",
        "adapter_path": None,
        "draft_model_id": None,
        "surface": "llm",
    }
    assert stream.calls[0][0] is cache
    payload = stream.calls[0][1]
    assert payload["input"][0] == {
        "type": "function_call_output",
        "call_id": "call_parent",
        "output": "receipt",
    }
    assert [part["type"] for part in payload["input"][1]["content"]] == [
        "input_text",
        "input_image",
    ]
    assert payload["stream"] is True
    assert payload["store"] is False
    assert port.stats()["event_queue_size"] == 256
    assert [type(event) for event in observed] == [
        OutputItemStarted,
        ContentPartStarted,
        ReasoningDelta,
        ReasoningCompleted,
        ContentPartCompleted,
        OutputItemCompleted,
        OutputItemStarted,
        ContentPartStarted,
        TextDelta,
        TextCompleted,
        ContentPartCompleted,
        OutputItemCompleted,
        OutputItemStarted,
        ToolDelta,
        ToolCompleted,
        OutputItemCompleted,
        UsageUpdate,
    ]
    assert not any(type(event).__name__.startswith("Turn") for event in observed)
    assert observed[-1] == UsageUpdate(7, 3, 10)
    assert cache.unload_calls == 0


@pytest.mark.asyncio
async def test_cancel_ack_waits_until_stream_finalizer_releases_execution() -> None:
    cache = _Cache()
    stream = _StreamFactory(())
    stream.block = True
    port = await CachedLegacyPortProvider(cache=cache, stream_factory=stream).acquire(
        RUNTIME, LoadConfig(options={"legacy_cancel_timeout_s": 1.0})
    )
    task = asyncio.create_task(_collect(port, _request()))
    while not stream.calls:
        await asyncio.sleep(0.01)

    accepted = await asyncio.to_thread(port.cancel, "resp_local", "disconnect")

    assert accepted is True
    assert stream.closed.is_set()
    await task
    assert port.stats()["active_executions"] == 0


@pytest.mark.asyncio
async def test_revision_and_sequential_generation_fail_closed_before_cache_access() -> (
    None
):
    cache = _Cache()
    provider = CachedLegacyPortProvider(cache=cache, stream_factory=_StreamFactory(()))
    revisioned = RuntimeKey(
        model_id="local/legacy",
        revision="immutable-but-unkeyed",
        backend=BackendKind.LEGACY_MLX,
    )

    with pytest.raises(LegacyPortContractError, match="requires an exact model_dir"):
        await provider.acquire(revisioned, LoadConfig())
    with pytest.raises(LegacyPortContractError, match="no truthful cooperative"):
        await provider.acquire(
            RUNTIME,
            LoadConfig(options={"legacy_generation_mode": "sequential"}),
        )

    assert cache.calls == []


@pytest.mark.asyncio
async def test_revision_uses_exact_existing_snapshot_as_shared_cache_key(
    tmp_path: Path,
) -> None:
    revision = "0123456789abcdef"
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    cache = _Cache()
    runtime = RuntimeKey(
        model_id="owner/model",
        revision=revision,
        backend=BackendKind.LEGACY_MLX,
    )

    port = await CachedLegacyPortProvider(
        cache=cache, stream_factory=_StreamFactory(())
    ).acquire(runtime, LoadConfig(options={"model_dir": str(snapshot)}))

    assert cache.calls[0]["model_id"] == str(snapshot.resolve())
    assert port.runtime_key == runtime
    assert port.stats()["runtime_key"]["model_ref"] == str(snapshot.resolve())


@pytest.mark.asyncio
async def test_untracked_wrapper_is_rejected_instead_of_creating_second_pool() -> None:
    cache = _Cache()
    cache.loaded = False

    with pytest.raises(LegacyPortContractError, match="duplicate residency"):
        await CachedLegacyPortProvider(
            cache=cache, stream_factory=_StreamFactory(())
        ).acquire(RUNTIME, LoadConfig())

    assert cache.unload_calls == 0


@pytest.mark.asyncio
async def test_sampling_not_supported_by_existing_responses_stream_fails_closed() -> (
    None
):
    cache = _Cache()
    port = await CachedLegacyPortProvider(
        cache=cache, stream_factory=_StreamFactory(())
    ).acquire(RUNTIME, LoadConfig())
    request = GenerationRequest(
        response_id="resp_seeded",
        runtime=RUNTIME,
        messages=(
            {"role": "user", "content": ({"type": "input_text", "text": "hi"},)},
        ),
        sampling={"seed": 7},
    )

    with pytest.raises(
        LegacyPortContractError, match="cannot preserve sampling fields: seed"
    ):
        await _collect(port, request)


def test_probe_is_explicit_about_cache_revision_and_sequential_limits() -> None:
    capability = CachedLegacyPortProvider(
        cache=_Cache(), stream_factory=_StreamFactory(())
    ).probe(
        ModelSpec(
            model_id="local/vision-legacy",
            architecture="LegacyVisionForConditionalGeneration",
        )
    )

    assert capability.supported is True
    assert capability.text is True
    assert capability.vision is True
    assert capability.tools is True
    assert capability.continuous_batching is False
    assert capability.facts["residency_owner"] == "MLXWrapperCache"
    assert capability.facts["sequential_generation"] is False
    assert capability.facts["revision_identity"] == "snapshot_directory_required"

    revisioned = CachedLegacyPortProvider(
        cache=_Cache(), stream_factory=_StreamFactory(())
    ).probe(ModelSpec(model_id="local/text", revision="sha256:unkeyed"))
    assert revisioned.supported is False
    assert revisioned.rejection_reasons == (
        "legacy cache cannot key or verify model revisions",
    )


def test_module_has_no_eager_mlx_or_second_cache_construction() -> None:
    source_path = (
        Path(__file__).parents[2]
        / "src/mlx_batch_server/runtime/backends/legacy_provider.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = [
        node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)
    ]

    assert not any(
        "mlx_lm" in ast.unparse(node) or "mlx_vlm" in ast.unparse(node)
        for node in top_level_imports
    )
    assert "MLXWrapperCache(" not in source
    assert ".unload_model(" not in source
    assert ".clear_cache(" not in source
    assert "queue.Queue(" in source
    assert "maxsize=event_queue_size" in source


def _typed_call_request() -> GenerationRequest:
    """The same continuation baton the fused preparer receives."""

    return GenerationRequest(
        response_id="resp_legacy_tool",
        runtime=RUNTIME,
        messages=(
            {
                "type": "message",
                "role": "user",
                "content": ({"type": "input_text", "text": "inspect"},),
            },
            {
                "type": "function_call",
                "role": "assistant",
                "id": "fc_1",
                "call_id": "call_inspect",
                "name": "inspect",
                "arguments": '{"region":"top"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "role": "tool",
                "call_id": "call_inspect",
                "output": "receipt",
                "content": ({"type": "input_text", "text": "receipt"},),
            },
            {
                "type": "message",
                "role": "user",
                "content": ({"type": "input_text", "text": "what now"},),
            },
        ),
        tools=({"type": "function", "name": "inspect", "parameters": {}},),
        sampling={"max_output_tokens": 32},
    )


async def _payload_for(request: GenerationRequest) -> Mapping[str, Any]:
    cache = _Cache()
    stream = _StreamFactory(_events())
    port = await CachedLegacyPortProvider(cache=cache, stream_factory=stream).acquire(
        RUNTIME, LoadConfig(max_admitted_requests=2)
    )
    await _collect(port, request)
    return stream.calls[0][1]


@pytest.mark.asyncio
async def test_legacy_renderer_seals_a_typed_call_instead_of_assistant_text() -> None:
    payload = await _payload_for(_typed_call_request())

    assert [item.get("type") for item in payload["input"]] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert payload["input"][1] == {
        "type": "function_call",
        "call_id": "call_inspect",
        "name": "inspect",
        "arguments": '{"region":"top"}',
        "id": "fc_1",
        "status": "completed",
    }
    assert "role" not in payload["input"][1]
    assert "content" not in payload["input"][1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"call_id": " "}, "call_id must not be empty"),
        ({"name": ""}, "name must not be empty"),
        ({"arguments": {"region": "top"}}, "arguments must be text"),
        ({"role": "user"}, "assistant role"),
        ({"status": "cancelled"}, "status is unsupported"),
        (
            {"content": ({"type": "input_text", "text": "smuggled"},)},
            "cannot carry message content",
        ),
    ),
)
async def test_legacy_renderer_refuses_malformed_call_identity(
    mutation: dict[str, Any],
    match: str,
) -> None:
    request = _typed_call_request()
    messages = list(request.messages)
    messages[1] = {**messages[1], **mutation}
    with pytest.raises(LegacyPortContractError, match=match):
        await _payload_for(
            GenerationRequest(
                response_id=request.response_id,
                runtime=request.runtime,
                messages=tuple(messages),
                tools=request.tools,
                sampling=request.sampling,
            )
        )


@pytest.mark.asyncio
async def test_legacy_renderer_refuses_an_orphan_tool_result() -> None:
    request = _typed_call_request()
    messages = list(request.messages)
    messages[2] = {**messages[2], "call_id": "call_nobody_made"}
    with pytest.raises(LegacyPortContractError, match="no preceding function_call"):
        await _payload_for(
            GenerationRequest(
                response_id=request.response_id,
                runtime=request.runtime,
                messages=tuple(messages),
                tools=request.tools,
                sampling=request.sampling,
            )
        )


@pytest.mark.asyncio
async def test_legacy_adapter_starts_a_tool_item_with_its_typed_identity() -> None:
    cache = _Cache()
    stream = _StreamFactory(_events())
    port = await CachedLegacyPortProvider(cache=cache, stream_factory=stream).acquire(
        RUNTIME, LoadConfig(max_admitted_requests=2)
    )

    observed = await _collect(port, _request())
    started = next(
        event
        for event in observed
        if isinstance(event, OutputItemStarted) and event.kind == "function_call"
    )
    assert (started.call_id, started.name) == ("call_1", "inspect")
    assert all(
        event.call_id is None and event.name is None
        for event in observed
        if isinstance(event, OutputItemStarted) and event.kind != "function_call"
    )
