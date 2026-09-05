"""Local implementations of non-generating OpenAI Responses operations."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any

from .compaction import LocalCompactionCodec, compacted_user_messages

if TYPE_CHECKING:
    from .controller import PreparedResponse, ResponsesController

_RUNTIME_SELECTION_FIELDS = frozenset(
    {
        "adapter_path",
        "backend",
        "draft_model",
        "draft_model_id",
        "model_revision",
        "revision",
        "runtime_role",
    }
)
_COUNT_FIELDS = frozenset(
    {
        "conversation",
        "input",
        "instructions",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "reasoning",
        "text",
        "tool_choice",
        "tools",
        "truncation",
        *_RUNTIME_SELECTION_FIELDS,
    }
)
_COMPACT_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "model",
        "previous_response_id",
        "prompt_cache_key",
        *_RUNTIME_SELECTION_FIELDS,
    }
)


class ResponsesOperationError(ValueError):
    """A non-generating operation cannot be honored without guessing."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        param: str | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.param = param
        self.status_code = status_code


class LocalResponsesTokenCounter:
    """Count text prompts with the tokenizer owned by the selected local role."""

    def __init__(self, model_directories: Mapping[str, str]) -> None:
        directories: dict[str, str] = {}
        for model, directory in model_directories.items():
            if not isinstance(model, str) or not model.strip():
                raise ValueError("token counter model names must be non-empty")
            if not isinstance(directory, str) or not directory.strip():
                raise ValueError("token counter model directories must be non-empty")
            directories[model.strip()] = directory.strip()
        self._model_directories = directories
        self._tokenizers: dict[str, Any] = {}
        self._lock = threading.Lock()

    def count(self, prepared: PreparedResponse) -> int:
        request = prepared.request
        model_id = request.runtime.model_id
        try:
            model_directory = self._model_directories[model_id]
        except KeyError:
            raise ResponsesOperationError(
                "the selected local model has no tokenizer directory",
                code="input_token_count_unavailable",
                param="model",
                status_code=503,
            ) from None
        tokenizer = self._tokenizer(model_id, model_directory)
        try:
            token_ids = tokenizer.apply_chat_template(
                _chat_messages(request.messages),
                tools=_chat_tools(request.tools, request.sampling.get("tool_choice")),
                tokenize=True,
                add_generation_prompt=True,
                **_reasoning_template_options(request.reasoning),
            )
        except ResponsesOperationError:
            raise
        except Exception as error:
            raise ResponsesOperationError(
                "the selected local tokenizer could not render this input",
                code="input_token_count_failed",
                param="input",
                status_code=503,
            ) from error
        try:
            return len(token_ids)
        except TypeError as error:
            raise ResponsesOperationError(
                "the selected local tokenizer returned an invalid token sequence",
                code="input_token_count_failed",
                param="input",
                status_code=503,
            ) from error

    def _tokenizer(self, model_id: str, model_directory: str) -> Any:
        with self._lock:
            cached = self._tokenizers.get(model_id)
            if cached is not None:
                return cached
            try:
                # Keep tokenizer/model machinery out of the non-generating API's
                # import path until a count operation actually needs it.
                load_tokenizer = import_module("mlx_lm.tokenizer_utils").load

                tokenizer = load_tokenizer(model_directory)
            except Exception as error:
                raise ResponsesOperationError(
                    "the selected local tokenizer is unavailable",
                    code="input_token_count_unavailable",
                    param="model",
                    status_code=503,
                ) from error
            self._tokenizers[model_id] = tokenizer
            return tokenizer


class LocalResponsesOperations:
    """Strict compact and input-token operations over controller-owned mapping."""

    def __init__(
        self,
        *,
        controller: ResponsesController,
        compaction_codec: LocalCompactionCodec,
        token_counter: LocalResponsesTokenCounter,
    ) -> None:
        self._controller = controller
        self._compaction_codec = compaction_codec
        self._token_counter = token_counter

    async def count_input_tokens(
        self,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> Mapping[str, Any]:
        body = _operation_body(payload, allowed=_COUNT_FIELDS)
        if "conversation" in body:
            raise ResponsesOperationError(
                "conversation is not supported by the process-local response store",
                code="unsupported_parameter",
                param="conversation",
            )
        truncation = body.pop("truncation", None)
        if truncation not in (None, "disabled"):
            raise ResponsesOperationError(
                "automatic input truncation is not supported",
                code="unsupported_parameter",
                param="truncation",
            )
        prepared = self._controller.inspect(body, owner_id=owner_id)
        count = await asyncio.to_thread(self._token_counter.count, prepared)
        return {"object": "response.input_tokens", "input_tokens": count}

    async def compact(
        self,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> Mapping[str, Any]:
        body = _operation_body(payload, allowed=_COMPACT_FIELDS)
        if "prompt_cache_key" in body:
            raise ResponsesOperationError(
                "prompt_cache_key is not supported by local compaction",
                code="unsupported_parameter",
                param="prompt_cache_key",
            )
        prepared = self._controller.inspect(body, owner_id=owner_id)
        input_tokens = await asyncio.to_thread(self._token_counter.count, prepared)
        messages = tuple(prepared.materialized_messages)
        encrypted = self._compaction_codec.seal(messages, owner_id=owner_id)
        output = [*compacted_user_messages(messages)]
        output.append(
            {
                "id": f"cmp_{uuid.uuid4().hex}",
                "type": "compaction",
                "encrypted_content": encrypted,
                "created_by": "mlx-batch-server",
            }
        )
        return {
            "id": f"resp_compact_{uuid.uuid4().hex}",
            "created_at": int(time.time()),
            "object": "response.compaction",
            "output": output,
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input_tokens,
            },
        }


def _operation_body(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Responses operation payload must be a mapping")
    body = dict(payload)
    unsupported = sorted(set(body) - allowed)
    if unsupported:
        field = unsupported[0]
        raise ResponsesOperationError(
            f"unsupported Responses parameter: {field}",
            code="unsupported_parameter",
            param=field,
        )
    return body


def _chat_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    instructions: list[str] = []
    rendered: list[Mapping[str, Any]] = []
    conversation_started = False
    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            if conversation_started:
                raise ResponsesOperationError(
                    "instructions must precede conversation messages",
                    code="invalid_input",
                    param="input",
                )
            instructions.append(_text_content(message.get("content")))
            continue
        conversation_started = True
        _text_content(message.get("content"))
        rendered.append(dict(message))
    if instructions:
        rendered.insert(0, {"role": "system", "content": "\n\n".join(instructions)})
    return rendered


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ResponsesOperationError(
            "input token counting requires canonical text content",
            code="unsupported_input_token_count",
            param="input",
        )
    text: list[str] = []
    for part in value:
        if not isinstance(part, Mapping) or part.get("type") != "input_text":
            raise ResponsesOperationError(
                "media input token counting requires a runtime vision token seam",
                code="unsupported_input_token_count",
                param="input",
            )
        value_text = part.get("text")
        if not isinstance(value_text, str):
            raise ResponsesOperationError(
                "input text must be a string",
                code="invalid_input",
                param="input",
            )
        text.append(value_text)
    return "".join(text)


def _chat_tools(
    tools: Sequence[Mapping[str, Any]],
    choice: Any,
) -> list[Mapping[str, Any]] | None:
    rendered = list(tools)
    if choice is None or (isinstance(choice, str) and choice in {"auto", "required"}):
        return rendered or None
    if choice == "none":
        return None
    if not isinstance(choice, Mapping) or choice.get("type") != "function":
        raise ResponsesOperationError(
            "only function tool_choice is supported",
            code="unsupported_tool_choice",
            param="tool_choice",
        )
    name = choice.get("name")
    selected = [tool for tool in rendered if tool.get("name") == name]
    if len(selected) != 1:
        raise ResponsesOperationError(
            "function tool_choice must name exactly one request tool",
            code="invalid_tool_choice",
            param="tool_choice.name",
        )
    return selected


def _reasoning_template_options(reasoning: Mapping[str, Any]) -> Mapping[str, Any]:
    if reasoning.get("enabled") is False or reasoning.get("effort") in {"none", "off"}:
        return {"enable_thinking": False}
    effort = reasoning.get("effort")
    if effort is None:
        return {"enable_thinking": True} if reasoning.get("enabled") is True else {}
    return {
        "enable_thinking": True,
        "reasoning_effort": "xhigh" if effort == "high" else effort,
    }


__all__ = [
    "LocalResponsesOperations",
    "LocalResponsesTokenCounter",
    "ResponsesOperationError",
]
