#!/usr/bin/env python3
"""Verify W1 MLX Batch API source contracts without importing production code."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
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
    tree = _parse_or_none(path)
    runtime = _class_or_none(tree, "_Qwen4ExpTensorRuntime")
    execute = _method_or_none(runtime, "execute")
    decode_batch = _method_or_none(runtime, "_decode_batch")
    target_batch = _method_or_none(runtime, "_target_forward_batch")
    execute_source = ast.unparse(execute) if execute is not None else ""
    decode_source = ast.unparse(decode_batch) if decode_batch is not None else ""
    target_source = ast.unparse(target_batch) if target_batch is not None else ""
    model_calls = _self_model_calls(target_batch)
    tensor_batch_builders = sum(
        1
        for method in (decode_batch, target_batch)
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
            and "self._target_forward_batch(" in decode_source,
            "multirow decode does not build a batch and issue exactly one model forward",
        ),
        SourceCheck(
            "multirow-mtp-enabled",
            "mtp" in decode_source.lower()
            and "MULTIROW_NOT_PROVEN" not in execute_source
            and "MULTIROW_NOT_PROVEN" not in target_source,
            "multirow execution still disables MTP instead of preserving per-row state",
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
