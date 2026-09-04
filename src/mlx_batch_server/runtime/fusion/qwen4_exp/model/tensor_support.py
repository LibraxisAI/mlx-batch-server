# SPDX-License-Identifier: Apache-2.0
"""Explicit request context and optional-kernel gates for Qwen4Exp tensors.

The tensor trunk is adapted from MTPLX commit
``6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab``. Native accelerators are not
part of the baseline ABI: a target owner enables a feature and injects every
callable used by that feature. Importing the tensor model never imports an
optional extension merely to discover whether it exists.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class Qwen4ExpTensorCapabilities:
    enabled: frozenset[str] = frozenset()
    kernels: Mapping[str, Callable[..., Any]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    qsa_gather_min_context: int = 16_384
    qsa_gather_max_rows: int = 8
    qsa_score_tile_rows: int = 0
    qsa_prefill_min_rows: int = 32
    qsa_prefill_min_context: int = 32_768
    qsa_prefill_flash_min_context: int = 32_768
    qsa_prefill_score_workspace_bytes: int = 128 * 1024 * 1024
    qsa_prefill_compile_rows: int = 2_048
    qsa_prefill_gather_tile_rows: int = 64

    def __post_init__(self) -> None:
        if any(not isinstance(name, str) or not name for name in self.enabled):
            raise TypeError("enabled tensor capabilities must be named")
        if any(
            not isinstance(name, str) or not name or not callable(kernel)
            for name, kernel in self.kernels.items()
        ):
            raise TypeError("native kernels must map names to callables")
        positive = (
            self.qsa_gather_min_context,
            self.qsa_gather_max_rows,
            self.qsa_prefill_min_rows,
            self.qsa_prefill_min_context,
            self.qsa_prefill_flash_min_context,
            self.qsa_prefill_score_workspace_bytes,
            self.qsa_prefill_compile_rows,
            self.qsa_prefill_gather_tile_rows,
        )
        if any(value < 1 for value in positive) or self.qsa_score_tile_rows < 0:
            raise ValueError("QSA capability geometry is invalid")

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> Qwen4ExpTensorCapabilities:
        names = options.get("qwen4_exp_native_capabilities", ())
        kernels = options.get("qwen4_exp_native_kernels", {})
        if isinstance(names, str | bytes):
            raise TypeError("qwen4_exp_native_capabilities must be a sequence")
        if not isinstance(kernels, Mapping):
            raise TypeError("qwen4_exp_native_kernels must be a mapping")
        return cls(
            enabled=frozenset(names),
            kernels=MappingProxyType(dict(kernels)),
            qsa_gather_min_context=_positive_option(
                options, "qsa_gather_min_context", 16_384
            ),
            qsa_gather_max_rows=_positive_option(options, "qsa_gather_max_rows", 8),
            qsa_score_tile_rows=_nonnegative_option(options, "qsa_score_tile_rows", 0),
            qsa_prefill_min_rows=_positive_option(options, "qsa_prefill_min_rows", 32),
            qsa_prefill_min_context=_positive_option(
                options, "qsa_prefill_min_context", 32_768
            ),
            qsa_prefill_flash_min_context=_positive_option(
                options, "qsa_prefill_flash_min_context", 32_768
            ),
            qsa_prefill_score_workspace_bytes=_positive_option(
                options, "qsa_prefill_score_workspace_bytes", 128 * 1024 * 1024
            ),
            qsa_prefill_compile_rows=_positive_option(
                options, "qsa_prefill_compile_rows", 2_048
            ),
            qsa_prefill_gather_tile_rows=_positive_option(
                options, "qsa_prefill_gather_tile_rows", 64
            ),
        )

    def has(self, name: str) -> bool:
        return name in self.enabled

    def has_kernel(self, name: str) -> bool:
        return name in self.kernels

    def kernel(self, name: str) -> Callable[..., Any]:
        try:
            return self.kernels[name]
        except KeyError as error:
            raise RuntimeError(f"native kernel was not injected: {name}") from error


_DEFAULT_CAPABILITIES = Qwen4ExpTensorCapabilities()
_CAPABILITIES: contextvars.ContextVar[Qwen4ExpTensorCapabilities | None] = (
    contextvars.ContextVar("qwen4_exp_tensor_capabilities", default=None)
)
_ATTENTION_PHASE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "qwen4_exp_attention_phase", default="decode"
)
_VISION_ROPE: contextvars.ContextVar[tuple[Any, int] | None] = contextvars.ContextVar(
    "qwen4_exp_vision_rope", default=None
)


def current_tensor_capabilities() -> Qwen4ExpTensorCapabilities:
    return _CAPABILITIES.get() or _DEFAULT_CAPABILITIES


def current_attention_phase() -> str:
    return _ATTENTION_PHASE.get()


def vision_rope_state() -> tuple[Any, int] | None:
    return _VISION_ROPE.get()


@contextlib.contextmanager
def tensor_capability_scope(
    capabilities: Qwen4ExpTensorCapabilities,
) -> Iterator[None]:
    token = _CAPABILITIES.set(capabilities)
    try:
        yield
    finally:
        _CAPABILITIES.reset(token)


@contextlib.contextmanager
def attention_phase_scope(phase: str) -> Iterator[None]:
    if phase not in {"prefill", "decode", "verify"}:
        raise ValueError(f"unsupported attention phase: {phase}")
    token = _ATTENTION_PHASE.set(phase)
    try:
        yield
    finally:
        _ATTENTION_PHASE.reset(token)


@contextlib.contextmanager
def vision_rope_scope(table: Any, delta: int) -> Iterator[None]:
    token = _VISION_ROPE.set((table, int(delta)))
    try:
        yield
    finally:
        _VISION_ROPE.reset(token)


def _kernel(name: str, *args: Any, **kwargs: Any) -> Any:
    return current_tensor_capabilities().kernel(name)(*args, **kwargs)


def _supported(name: str, *args: Any, **kwargs: Any) -> bool:
    capabilities = current_tensor_capabilities()
    if not capabilities.has_kernel(name):
        return False
    return bool(capabilities.kernel(name)(*args, **kwargs))


def _positive_option(options: Mapping[str, Any], name: str, default: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_option(options: Mapping[str, Any], name: str, default: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def fused_gdn_step(*args: Any, **kwargs: Any) -> Any:
    return _kernel("fused_gdn_step", *args, **kwargs)


def fused_gdn_conv_norm(*args: Any, **kwargs: Any) -> Any:
    return _kernel("fused_gdn_conv_norm", *args, **kwargs)


def fused_gdn_conv_norm_rows(*args: Any, **kwargs: Any) -> Any:
    return _kernel("fused_gdn_conv_norm_rows", *args, **kwargs)


def fused_gdn_out(*args: Any, **kwargs: Any) -> Any:
    return _kernel("fused_gdn_out", *args, **kwargs)


def device_supports_gdn_conv_norm() -> bool:
    return _supported("device_supports_gdn_conv_norm")


def device_supports_gdn_conv_norm_rows() -> bool:
    return _supported("device_supports_gdn_conv_norm_rows")


def device_supports_hyper_v3() -> bool:
    return _supported("device_supports_hyper_v3")


def prepare_v3_pack(*args: Any, **kwargs: Any) -> Any:
    return _kernel("prepare_v3_pack", *args, **kwargs)


def fused_hyper_read_v3(*args: Any, **kwargs: Any) -> Any:
    return _kernel("fused_hyper_read_v3", *args, **kwargs)


def fused_hyper_read(*args: Any, **kwargs: Any) -> Any:
    return _kernel("fused_hyper_read", *args, **kwargs)


def moe_glu_decode(*args: Any, **kwargs: Any) -> Any:
    return _kernel("moe_glu_decode", *args, **kwargs)


def moe_glu_verify(*args: Any, **kwargs: Any) -> Any:
    return _kernel("moe_glu_verify", *args, **kwargs)


def qsa_indexer_select_nax_available() -> bool:
    return _supported("qsa_indexer_select_nax_available")


def qsa_indexer_prepare_supported(*args: Any, **kwargs: Any) -> bool:
    return _supported("qsa_indexer_prepare_supported", *args, **kwargs)


def qsa_indexer_pool_keys_metal(*args: Any, **kwargs: Any) -> Any:
    return _kernel("qsa_indexer_pool_keys_metal", *args, **kwargs)


def qsa_indexer_prepare_queries_metal(*args: Any, **kwargs: Any) -> Any:
    return _kernel("qsa_indexer_prepare_queries_metal", *args, **kwargs)


def qsa_indexer_select_blocks_metal(*args: Any, **kwargs: Any) -> Any:
    return _kernel("qsa_indexer_select_blocks_metal", *args, **kwargs)


def qsa_indexer_select_dense_mask_metal(*args: Any, **kwargs: Any) -> Any:
    return _kernel("qsa_indexer_select_dense_mask_metal", *args, **kwargs)


def qsa_indexer_select_row_tokens_metal(*args: Any, **kwargs: Any) -> Any:
    return _kernel("qsa_indexer_select_row_tokens_metal", *args, **kwargs)


def QSACompiledIndexerCore(*args: Any, **kwargs: Any) -> Any:
    return _kernel("QSACompiledIndexerCore", *args, **kwargs)


def qsa_indexer_prefill_blocks_metal(*args: Any, **kwargs: Any) -> Any:
    return _kernel("qsa_indexer_prefill_blocks_metal", *args, **kwargs)


def qsa_flash_skip(*args: Any, **kwargs: Any) -> Any:
    return _kernel("qsa_flash_skip", *args, **kwargs)


def qsa_prefill_flash(*args: Any, **kwargs: Any) -> Any:
    return _kernel("qsa_prefill_flash", *args, **kwargs)


def qsa_prefill_flash_supported(*args: Any, **kwargs: Any) -> bool:
    return _supported("qsa_prefill_flash_supported", *args, **kwargs)


__all__ = [
    "Qwen4ExpTensorCapabilities",
    "attention_phase_scope",
    "current_attention_phase",
    "current_tensor_capabilities",
    "tensor_capability_scope",
    "vision_rope_scope",
    "vision_rope_state",
]
