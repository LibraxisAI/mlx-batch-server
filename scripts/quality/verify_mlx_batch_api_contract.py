#!/usr/bin/env python3
"""Verify W1 MLX Batch API source contracts without importing production code."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ("openai", "anthropic", "safe-fetch", "multirow")
MARKERS = {
    "openai": "OPENAI_SOURCE_CONTRACT",
    "anthropic": "ANTHROPIC_SOURCE_CONTRACT",
    "safe-fetch": "SAFE_PUBLIC_FETCH_SOURCE_CONTRACT",
    "multirow": "MULTIROW_SOURCE_CONTRACT",
}


@dataclass(frozen=True, slots=True)
class SourceCheck:
    name: str
    passed: bool
    requirement: str


@dataclass(frozen=True, slots=True)
class SectionResult:
    section: str
    checks: tuple[SourceCheck, ...]

    @property
    def green(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def state(self) -> str:
        return "green" if self.green else "red"

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(check.requirement for check in self.checks if not check.passed)


def evaluate_section(root: Path, section: str) -> SectionResult:
    """Evaluate one deterministic source-only contract section."""

    evaluators: dict[str, Callable[[Path], tuple[SourceCheck, ...]]] = {
        "openai": _openai_checks,
        "anthropic": _anthropic_checks,
        "safe-fetch": _safe_fetch_checks,
        "multirow": _multirow_checks,
    }
    try:
        evaluator = evaluators[section]
    except KeyError as error:
        raise ValueError(f"unknown contract section: {section}") from error
    return SectionResult(section=section, checks=evaluator(root))


def _openai_checks(root: Path) -> tuple[SourceCheck, ...]:
    router = root / "src/mlx_batch_server/responses/runtime_router.py"
    response_sources = tuple((root / "src/mlx_batch_server/responses").glob("*.py"))
    router_tree = _parse_or_none(router)
    routes = _fastapi_routes(router_tree) if router_tree is not None else set()
    response_trees = tuple(
        tree for path in response_sources if (tree := _parse_or_none(path)) is not None
    )
    return (
        SourceCheck(
            "compact-endpoint",
            ("POST", "/v1/responses/compact") in routes,
            "POST /v1/responses/compact is absent from the canonical runtime router",
        ),
        SourceCheck(
            "input-token-endpoint",
            ("POST", "/v1/responses/input_tokens") in routes,
            "POST /v1/responses/input_tokens is absent from the canonical runtime router",
        ),
        SourceCheck(
            "pending-steer-event",
            any(
                _contains_dict_type(tree, "response.steer.pending")
                for tree in response_trees
            ),
            "response.steer.pending has no emitted typed event payload",
        ),
    )


def _anthropic_checks(root: Path) -> tuple[SourceCheck, ...]:
    directory = root / "src/mlx_batch_server/chat/anthropic"
    paths = tuple(directory.glob("*.py"))
    trees = tuple(tree for path in paths if (tree := _parse_or_none(path)) is not None)
    modules = _imported_modules(trees)
    constants = _string_constants(trees)
    strict_request = any(_has_strict_model_config(tree) for tree in trees)
    capabilities_path = directory / "capabilities.py"
    router_path = directory / "router.py"
    engine_path = directory / "messages_engine.py"
    mapper_path = directory / "request_mapper.py"
    content_mapper_path = directory / "content_mapper.py"
    projector_path = directory / "projector.py"
    capabilities_tree = _parse_or_none(capabilities_path)
    router_tree = _parse_or_none(router_path)
    mapper_tree = _parse_or_none(mapper_path)
    create_message = _function_or_none(router_tree, "create_message")
    enforce_line = _first_call_line(create_message, "enforce_capabilities")
    mapping_line = _first_call_line(create_message, "build_turn")
    stream_line = _first_call_line(create_message, "StreamingResponse")
    engine_source = _read_or_empty(engine_path)
    mapper_source = _read_or_empty(mapper_path)
    content_mapper_source = _read_or_empty(content_mapper_path)
    projector_source = _read_or_empty(projector_path)
    unsupported_keys = _classification_keys(capabilities_tree, "_unsupported")
    explicit_refusals = {
        "cache_control",
        "citations",
        "container",
        "inference_geo",
        "output_config",
        "output_config.format",
        "output_config.effort",
        "effort",
        "content.server_tool_use",
        "content.web_search_tool_result",
        "content.container_upload",
    }
    represented_blocks = {
        "RequestServerToolUseBlock",
        "RequestWebSearchToolResultBlock",
        "RequestContainerUploadBlock",
        "RequestThinkingBlock",
        "RequestRedactedThinkingBlock",
    }
    return (
        SourceCheck(
            "shared-runtime-owner",
            "mlx_batch_server.chat.mlx.chat_generator" not in modules
            and "mlx_batch_server.runtime.events" in modules,
            "Anthropic still owns a legacy ChatGenerator path instead of typed runtime events",
        ),
        SourceCheck(
            "strict-request-schema",
            strict_request,
            "Anthropic request models do not fail closed with ConfigDict(extra='forbid')",
        ),
        SourceCheck(
            "tool-json-streaming",
            "input_json_delta" in constants,
            "Anthropic streaming has no input_json_delta tool-argument event",
        ),
        SourceCheck(
            "typed-error-request-id",
            "request_id" in constants and "error" in constants,
            "Anthropic error projection has no request_id-bearing error contract",
        ),
        SourceCheck(
            "single-capability-owner",
            _top_level_definition_count(trees, "enforce_capabilities") == 1
            and _class_or_none(capabilities_tree, "AnthropicCapabilityProfile")
            is not None
            and _function_or_none(capabilities_tree, "enforce_capabilities")
            is not None,
            "Anthropic capability admission is not owned exactly once by capabilities.py",
        ),
        SourceCheck(
            "pre-sse-capability-rejection",
            enforce_line is not None
            and mapping_line is not None
            and stream_line is not None
            and enforce_line < mapping_line < stream_line
            and _call_count(create_message, "enforce_capabilities") == 1,
            "Anthropic capability admission is not completed before mapping and StreamingResponse creation",
        ),
        SourceCheck(
            "thinking-signature-gate",
            {
                "thinking.enabled",
                "content.thinking",
                "content.redacted_thinking",
            }
            <= unsupported_keys
            and "ThinkingProjection.refused()" in engine_source
            and "ThinkingProjection.signed_by(owner)" in engine_source
            and "thinking_signature_owner" in engine_source
            and "signature_delta" in projector_source,
            "Anthropic thinking can reach the wire without explicit request and signature ownership gates",
        ),
        SourceCheck(
            "ordered-rich-content-mapping",
            _call_count(
                _function_or_none(mapper_tree, "_map_user_content"),
                "map_anthropic_content",
            )
            == 1
            and all(
                marker in content_mapper_source
                for marker in (
                    "RequestImageBlock",
                    "RequestDocumentBlock",
                    "RequestSearchResultBlock",
                    "content_index",
                    "CALLER-SUPPLIED UNTRUSTED SEARCH RESULT",
                )
            )
            and "media.extend(canonical.media)" in mapper_source,
            "Anthropic rich blocks are not mapped once in caller order onto the canonical content/media ABI",
        ),
        SourceCheck(
            "explicit-no-silent-ignore",
            strict_request
            and explicit_refusals <= unsupported_keys
            and represented_blocks <= _imported_names(capabilities_tree)
            and represented_blocks <= set(_mapping_key_names(capabilities_tree)),
            "Anthropic official fields or content discriminators can bypass an explicit fail-closed classification",
        ),
    )


def _safe_fetch_checks(root: Path) -> tuple[SourceCheck, ...]:
    path = root / "src/mlx_batch_server/utils/safe_public_fetch.py"
    tree = _parse_or_none(path)
    source = _read_or_empty(path)
    definitions = _top_level_definitions(tree)
    modules = _imported_modules((tree,)) if tree is not None else set()
    attributes = (
        {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        if tree is not None
        else set()
    )
    media_paths = (
        root / "src/mlx_batch_server/runtime/fusion/qwen4_exp/media_resolver.py",
        root / "src/mlx_batch_server/runtime/fusion/qwen4_exp/media_adapters.py",
    )
    media_source = "\n".join(_read_or_empty(item) for item in media_paths)
    interface = {
        "SafePublicFetch",
        "SafePublicFetchError",
        "SafePublicFetchLimits",
        "FetchedResource",
    }
    blocked_predicates = {
        "is_private",
        "is_loopback",
        "is_link_local",
        "is_multicast",
        "is_unspecified",
        "is_reserved",
    }
    return (
        SourceCheck(
            "owned-interface",
            path.is_file() and interface <= definitions,
            "target-owned SafePublicFetch interface module is absent or incomplete",
        ),
        SourceCheck(
            "address-classification",
            {"ipaddress", "socket"} <= modules
            and blocked_predicates <= attributes
            and "100.64.0.0/10" in source
            and "169.254.169.254" in source,
            "public-address validation lacks private/special, CGNAT-Tailscale, or metadata checks",
        ),
        SourceCheck(
            "dns-and-redirect-revalidation",
            "getaddrinfo" in source
            and "max_redirects" in source
            and "location" in source.lower(),
            "DNS answers and redirect hops are not both represented in the safe fetch boundary",
        ),
        SourceCheck(
            "streaming-resource-limits",
            all(
                marker in source
                for marker in ("max_bytes", "timeout", "media_type", "max_redirects")
            ),
            "safe fetch does not expose byte, timeout, MIME, and redirect limits",
        ),
        SourceCheck(
            "fused-media-wiring",
            "SafePublicFetch" in media_source
            and "URL origin is not explicitly allowed" not in media_source
            and "AllowedUrlPolicy" in media_source,
            "fused media is not routed through SafePublicFetch with optional origin lockdown",
        ),
    )


def _multirow_checks(root: Path) -> tuple[SourceCheck, ...]:
    path = root / "src/mlx_batch_server/runtime/fusion/qwen4_exp/model/tensor.py"
    benchmark_path = root / "scripts/benchmark_live_responses.py"
    tree = _parse_or_none(path)
    benchmark_tree = _parse_or_none(benchmark_path)
    runtime = _class_or_none(tree, "_Qwen4ExpTensorRuntime")
    telemetry = _class_or_none(tree, "_TensorForwardTelemetry")
    execute = _method_or_none(runtime, "execute")
    stats = _method_or_none(runtime, "stats")
    decode_batch = _method_or_none(runtime, "_decode_batch")
    decode_mtp_batch = _method_or_none(runtime, "_decode_mtp_batch")
    target_batch = _method_or_none(runtime, "_target_forward_batch")
    mtp_forward_batch = _method_or_none(runtime, "_mtp_forward_batch")
    mtp_update_batch = _method_or_none(runtime, "_mtp_update_batch")
    mtp_singleton = _method_or_none(runtime, "_decode_mtp_one")
    ar_singleton = _method_or_none(runtime, "_decode_ar_one")
    target_singleton = _method_or_none(runtime, "_target_forward")
    benchmark_main = _function_or_none(benchmark_tree, "_main")
    benchmark_parser = _function_or_none(benchmark_tree, "_parser")
    execute_source = ast.unparse(execute) if execute is not None else ""
    decode_source = ast.unparse(decode_batch) if decode_batch is not None else ""
    mtp_source = ast.unparse(decode_mtp_batch) if decode_mtp_batch is not None else ""
    stats_source = ast.unparse(stats) if stats is not None else ""
    telemetry_source = ast.unparse(telemetry) if telemetry is not None else ""
    benchmark_main_source = (
        ast.unparse(benchmark_main) if benchmark_main is not None else ""
    )
    benchmark_parser_source = (
        ast.unparse(benchmark_parser) if benchmark_parser is not None else ""
    )
    tensor_source = _read_or_empty(path)
    benchmark_file_source = _read_or_empty(benchmark_path)
    scatter = _function_or_none(tree, "_scatter_batch_cache")
    scatter_source = ast.unparse(scatter) if scatter is not None else ""
    model_calls = _self_model_calls(target_batch)
    draft_model_calls = _self_model_attribute_calls(mtp_forward_batch, "mtp_forward")
    tensor_batch_builders = sum(
        1
        for method in (decode_batch, target_batch, mtp_forward_batch)
        if method is not None
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"stack", "concatenate"}
    )
    return (
        SourceCheck(
            "batch-seam-wired",
            decode_batch is not None
            and target_batch is not None
            and "self._decode_batch(" in execute_source,
            "tensor execution has no _decode_batch -> _target_forward_batch seam",
        ),
        SourceCheck(
            "no-row-serial-model-loop",
            execute is not None and not _has_row_serial_decode_loop(execute),
            "execute still invokes _decode_ar_one/_decode_mtp_one once per decode row",
        ),
        SourceCheck(
            "one-tensor-forward",
            len(model_calls) == 1
            and tensor_batch_builders >= 1
            and "self._target_forward_batch(" in decode_source
            and not _named_call_inside_loop(target_batch, "model"),
            "multirow decode does not build a batch and issue exactly one model forward",
        ),
        SourceCheck(
            "multirow-mtp-wired",
            decode_mtp_batch is not None
            and mtp_singleton is not None
            and _self_method_call_count(decode_batch, "_decode_mtp_batch") == 1
            and "mtp_decision.enabled" in decode_source
            and "MULTIROW_NOT_PROVEN" not in execute_source
            and "MULTIROW_NOT_PROVEN" not in decode_source
            and "MULTIROW_NOT_PROVEN" not in mtp_source,
            "_decode_batch does not route admitted cohorts into concrete multi-row MTP",
        ),
        SourceCheck(
            "batched-recursive-drafts",
            mtp_forward_batch is not None
            and len(draft_model_calls) == 1
            and _call_inside_loop(decode_mtp_batch, "_mtp_forward_batch")
            and not _named_call_inside_loop(mtp_forward_batch, "mtp_forward")
            and "mx.concatenate" in ast.unparse(mtp_forward_batch)
            and "mx.stack" in ast.unparse(mtp_forward_batch),
            "recursive draft depths are not issued as one batched MTP model call",
        ),
        SourceCheck(
            "shared-verify-and-variable-commit",
            _call_with_keyword_value(
                decode_mtp_batch,
                "_target_forward_batch",
                "phase",
                "verify",
            )
            and len(
                _self_model_attribute_calls(
                    decode_mtp_batch,
                    "commit_verified_window",
                )
            )
            == 1
            and "accepted_counts[row_index]" in mtp_source
            and "verified_tokens=len(verify_token_rows[row_index])" in mtp_source,
            "multi-row MTP lacks one shared verify call or row-local verified-window commits",
        ),
        SourceCheck(
            "capture-scatter-and-batched-repair",
            "_qwen4_exp_verify_rows" in scatter_source
            and "_qwen4_exp_verify_ple" in scatter_source
            and "correction_indices" in mtp_source
            and _self_method_call_count(decode_mtp_batch, "_mtp_update_batch") == 1
            and "history_depth" in mtp_source,
            "verify captures, stochastic corrections, or MTP history are not scattered by row",
        ),
        SourceCheck(
            "tensor-forward-telemetry-schema",
            telemetry is not None
            and "qwen4-exp.tensor-forward.v1" in tensor_source
            and "per_tensor_runtime_instance" in tensor_source
            and all(
                phase in tensor_source
                for phase in (
                    "target_decode",
                    "mtp_draft",
                    "target_verify",
                    "target_correction",
                    "mtp_history_update",
                )
            )
            and "tensor_forward" in stats_source
            and "runtime_instance_id" in telemetry_source
            and "completed_calls_by_shape" in tensor_source,
            "tensor stats lack the versioned per-runtime physical-forward schema",
        ),
        SourceCheck(
            "completed-physical-forward-recorders",
            _calls_are_ordered(
                target_batch,
                ("model", "eval", "_scatter_batch_cache", "_record_tensor_forward"),
            )
            and _call_has_literal_argument(
                target_batch,
                "_record_tensor_forward",
                None,
            )
            and _calls_are_ordered(
                mtp_forward_batch,
                (
                    "mtp_forward",
                    "eval",
                    "_scatter_batch_cache",
                    "_record_tensor_forward",
                ),
            )
            and _call_has_literal_argument(
                mtp_forward_batch,
                "_record_tensor_forward",
                "mtp_draft",
            )
            and _calls_are_ordered(
                mtp_update_batch,
                (
                    "mtp_update_cache",
                    "eval",
                    "_scatter_batch_cache",
                    "_record_tensor_forward",
                ),
            )
            and _call_has_literal_argument(
                mtp_update_batch,
                "_record_tensor_forward",
                "mtp_history_update",
            )
            and not _named_call_inside_loop(mtp_update_batch, "mtp_update_cache")
            and _calls_are_ordered(
                decode_mtp_batch,
                ("commit_verified_window", "_record_tensor_forward"),
            )
            and _call_has_literal_argument(
                decode_mtp_batch,
                "_record_tensor_forward",
                "target_verify",
            )
            and _call_with_keyword_value(
                decode_mtp_batch,
                "_target_forward_batch",
                "telemetry_phase",
                "target_correction",
            )
            and _calls_are_ordered(
                target_singleton,
                ("model", "eval", "_record_tensor_forward"),
            )
            and all(
                _call_has_literal_argument(
                    mtp_singleton,
                    "_record_tensor_forward",
                    phase,
                )
                for phase in ("mtp_draft", "target_verify", "mtp_history_update")
            )
            and _call_with_keyword_value(
                mtp_singleton,
                "_target_forward",
                "telemetry_phase",
                "target_correction",
            )
            and _calls_are_ordered(
                ar_singleton,
                ("mtp_update_cache", "eval", "_record_tensor_forward"),
            )
            and _call_has_literal_argument(
                ar_singleton,
                "_record_tensor_forward",
                "mtp_history_update",
            ),
            "physical target/MTP completion recorders are missing, early, or row-serial",
        ),
        SourceCheck(
            "live-histogram-delta-requirements",
            "--require-target-batch-rows" in benchmark_parser_source
            and "--require-mtp-batch-rows" in benchmark_parser_source
            and "same_runtime_instance" in benchmark_file_source
            and "completed_calls_by_shape" in benchmark_file_source
            and "_has_positive_batch_delta" in benchmark_file_source
            and "_tensor_forward_requirements(" in benchmark_main_source
            and "tensor_forward_before" in benchmark_main_source
            and "tensor_forward_after" in benchmark_main_source,
            "live benchmark lacks same-instance positive tensor histogram delta gates",
        ),
    )


def _read_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_or_none(path: Path) -> ast.Module | None:
    source = _read_or_empty(path)
    if not source:
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None


def _fastapi_routes(tree: ast.Module) -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            if not isinstance(decorator.func, ast.Attribute):
                continue
            method = decorator.func.attr.upper()
            path = decorator.args[0]
            if (
                method in {"GET", "POST", "DELETE", "PATCH", "PUT"}
                and isinstance(path, ast.Constant)
                and isinstance(path.value, str)
            ):
                routes.add((method, path.value))
    return routes


def _contains_dict_type(tree: ast.Module, event_type: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        pairs = zip(node.keys, node.values, strict=True)
        if any(
            isinstance(key, ast.Constant)
            and key.value == "type"
            and isinstance(value, ast.Constant)
            and value.value == event_type
            for key, value in pairs
        ):
            return True
    return False


def _imported_modules(trees: Iterable[ast.Module | None]) -> set[str]:
    modules: set[str] = set()
    for tree in trees:
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
    return modules


def _string_constants(trees: Iterable[ast.Module]) -> set[str]:
    return {
        node.value
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _has_strict_model_config(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "model_config"
            for target in node.targets
        ):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        if not isinstance(value.func, ast.Name) or value.func.id != "ConfigDict":
            continue
        if any(
            keyword.arg == "extra"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "forbid"
            for keyword in value.keywords
        ):
            return True
    return False


def _top_level_definitions(tree: ast.Module | None) -> set[str]:
    if tree is None:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _class_or_none(tree: ast.Module | None, name: str) -> ast.ClassDef | None:
    if tree is None:
        return None
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == name
        ),
        None,
    )


def _function_or_none(
    tree: ast.Module | None,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if tree is None:
        return None
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == name
        ),
        None,
    )


def _top_level_definition_count(trees: Iterable[ast.Module], name: str) -> int:
    return sum(
        1
        for tree in trees
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == name
    )


def _call_count(
    node: ast.AST | None,
    name: str,
) -> int:
    if node is None:
        return 0
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_name(child) == name
    )


def _first_call_line(node: ast.AST | None, name: str) -> int | None:
    if node is None:
        return None
    return min(
        (
            child.lineno
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and _call_name(child) == name
        ),
        default=None,
    )


def _classification_keys(tree: ast.Module | None, helper: str) -> set[str]:
    if tree is None:
        return set()
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == helper
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def _imported_names(tree: ast.Module | None) -> set[str]:
    if tree is None:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom | ast.Import):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _mapping_key_names(tree: ast.Module | None) -> tuple[str, ...]:
    if tree is None:
        return ()
    return tuple(
        key.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Name)
    )


def _method_or_none(
    class_node: ast.ClassDef | None,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if class_node is None:
        return None
    return next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == name
        ),
        None,
    )


def _has_row_serial_decode_loop(
    execute: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for node in ast.walk(execute):
        if not isinstance(node, ast.For):
            continue
        if "decode" not in ast.unparse(node.iter):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in {"_decode_ar_one", "_decode_mtp_one"}
            ):
                return True
    return False


def _self_model_calls(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> tuple[ast.Call, ...]:
    if method is None:
        return ()
    return tuple(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "model"
    )


def _self_model_attribute_calls(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    name: str,
) -> tuple[ast.Call, ...]:
    if method is None:
        return ()
    return tuple(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "model"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
    )


def _self_method_call_count(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    name: str,
) -> int:
    if method is None:
        return 0
    return sum(
        1
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )


def _call_inside_loop(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    name: str,
) -> bool:
    if method is None:
        return False
    for loop in (
        node for node in ast.walk(method) if isinstance(node, ast.For | ast.While)
    ):
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            for node in ast.walk(loop)
        ):
            return True
    return False


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _named_call_inside_loop(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    name: str,
) -> bool:
    if method is None:
        return False
    return any(
        _call_name(call) == name
        for loop in ast.walk(method)
        if isinstance(loop, ast.For | ast.While | ast.comprehension)
        for call in ast.walk(loop)
        if isinstance(call, ast.Call)
    )


def _calls_are_ordered(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    names: tuple[str, ...],
) -> bool:
    if method is None:
        return False
    positions = {
        name: min(
            (
                node.lineno
                for node in ast.walk(method)
                if isinstance(node, ast.Call) and _call_name(node) == name
            ),
            default=-1,
        )
        for name in names
    }
    return all(positions[name] >= 0 for name in names) and all(
        positions[left] < positions[right] for left, right in pairwise(names)
    )


def _call_has_literal_argument(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    name: str,
    value: str | None,
) -> bool:
    if method is None:
        return False
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or _call_name(node) != name:
            continue
        if value is None:
            return True
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == value
        ):
            return True
    return False


def _call_with_keyword_value(
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    name: str,
    keyword_name: str,
    keyword_value: object,
) -> bool:
    if method is None:
        return False
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != name:
            continue
        if any(
            keyword.arg == keyword_name
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == keyword_value
            for keyword in node.keywords
        ):
            return True
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--section", choices=SECTIONS)
    parser.add_argument("--expect", choices=("red", "green"), required=True)
    parser.add_argument(
        "--no-imports",
        action="store_true",
        help="required attestation that production modules must only be parsed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.no_imports:
        _parser().error("--no-imports is required for source-contract verification")
    sections = SECTIONS if args.all else (args.section,)
    matches = True
    for section in sections:
        result = evaluate_section(ROOT, section)
        marker = MARKERS[section]
        print(f"{marker}={result.state}")
        for check in result.checks:
            status = "pass" if check.passed else "missing"
            print(f"  {status}: {check.name}: {check.requirement}")
        matches = matches and result.state == args.expect
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
