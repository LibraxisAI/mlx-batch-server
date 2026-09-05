"""Source-only contract for the production hosted-tool composition cut."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "src/mlx_batch_server/core/config.py"
BRAVE_PATH = ROOT / "src/mlx_batch_server/tools/brave_search.py"
BOOTSTRAP_PATH = ROOT / "src/mlx_batch_server/responses/runtime_bootstrap.py"
MAIN_PATH = ROOT / "src/mlx_batch_server/main.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(
    tree: ast.AST,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name == name
    ]
    assert len(matches) == 1, f"expected one function {name!r}"
    return matches[0]


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_name(child.func) == name
    ]


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _keyword(call: ast.Call, name: str) -> ast.expr:
    matches = [item.value for item in call.keywords if item.arg == name]
    assert len(matches) == 1, f"expected one {name!r} keyword"
    return matches[0]


def _constant(node: ast.expr) -> Any:
    return ast.literal_eval(node)


def test_process_runtime_reads_settings_and_builds_one_catalog_once() -> None:
    function = _function(_tree(MAIN_PATH), "_compose_process_runtime")

    assert len(_calls(function, "get_settings")) == 1
    catalog_calls = _calls(function, "compose_production_hosted_catalog")
    assert len(catalog_calls) == 1
    compose_calls = _calls(function, "compose_role_responses_runtime")
    assert len(compose_calls) == 1

    catalog_assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and _call_name(node.value.func) == "compose_production_hosted_catalog"
    )
    assert len(catalog_assignment.targets) == 1
    assert isinstance(catalog_assignment.targets[0], ast.Name)
    catalog_name = catalog_assignment.targets[0].id
    injected = _keyword(compose_calls[0], "hosted_tools")
    assert isinstance(injected, ast.Name) and injected.id == catalog_name

    factory_key = _keyword(catalog_calls[0], "brave_api_key")
    assert isinstance(factory_key, ast.Name)
    assert len(_calls(function, "get_secret_value")) == 1
    assert not {"allowed_url_origins", "media_url_origins"} & {
        keyword.arg for keyword in catalog_calls[0].keywords
    }


def test_production_factory_has_exact_closed_catalog_and_public_fetch_policy() -> None:
    source = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    function = _function(ast.parse(source), "compose_production_hosted_catalog")
    function_source = ast.unparse(function)

    catalog_calls = _calls(function, "HostedToolCatalog")
    assert len(catalog_calls) == 1
    assert len(_calls(function, "HostedWebSearchTool")) == 1
    fetch_tool_calls = _calls(function, "HostedWebFetchTool")
    assert len(fetch_tool_calls) == 1
    fetch_calls = _calls(function, "SafePublicFetch")
    assert len(fetch_calls) == 1
    limit_calls = _calls(function, "SafePublicFetchLimits")
    assert len(limit_calls) == 1

    assert _constant(_keyword(fetch_calls[0], "allowed_origins")) == ()
    assert {
        keyword.arg: _constant(keyword.value) for keyword in limit_calls[0].keywords
    } == {
        "max_bytes": 1_048_576,
        "timeout": 20.0,
        "connect_timeout": 5.0,
        "write_timeout": 5.0,
        "pool_timeout": 5.0,
        "max_redirects": 3,
        "chunk_bytes": 65_536,
    }
    assert _constant(_keyword(fetch_tool_calls[0], "max_bytes")) == 1_048_576
    assert _constant(_keyword(fetch_tool_calls[0], "max_text_chars")) == 262_144
    assert _constant(_keyword(fetch_tool_calls[0], "accepted_media_types")) == (
        "text/html",
        "text/plain",
        "text/markdown",
        "application/json",
    )
    tools = catalog_calls[0].args[0]
    assert isinstance(tools, ast.Tuple)
    tool_names = [
        _call_name(item.func) for item in tools.elts if isinstance(item, ast.Call)
    ]
    assert tool_names == [
        "HostedWebSearchTool",
        "HostedWebFetchTool",
    ]
    assert _calls(function, "AsyncClient") == []
    assert "allowed_url_origins" not in function_source
    assert "media_url_origins" not in function_source
    assert "execute_web_search" not in function_source
    assert "compose_production_hosted_catalog" in source.rsplit("__all__", 1)[-1]


def test_brave_provider_source_is_fixed_single_attempt_and_secret_closed() -> None:
    source = BRAVE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    provider = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BraveSearchProvider"
    )
    slots = next(
        node
        for node in provider.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__slots__"
            for target in node.targets
        )
    )
    assert _constant(slots.value) == ("_api_key", "_transport")
    call = _function(ast.Module(body=provider.body, type_ignores=[]), "__call__")

    assert "https://api.search.brave.com/res/v1/web/search" in source
    assert "X-Subscription-Token" in source
    assert "description" in source and "snippet" in source
    assert "__slots__" in source
    assert "os.environ" not in source
    assert "logging" not in source
    assert "retry" not in source.casefold()
    assert "more_results_available" not in source

    client_calls = _calls(call, "AsyncClient")
    assert len(client_calls) == 1
    assert _constant(_keyword(client_calls[0], "trust_env")) is False
    assert _constant(_keyword(client_calls[0], "follow_redirects")) is False
    timeout_calls = _calls(call, "Timeout")
    assert len(timeout_calls) == 1
    assert {
        keyword.arg: _constant(keyword.value) for keyword in timeout_calls[0].keywords
    } == {"connect": 5.0, "read": 10.0, "write": 5.0, "pool": 5.0}

    get_calls = _calls(call, "get")
    assert len(get_calls) == 1
    params = _keyword(get_calls[0], "params")
    assert isinstance(params, ast.Dict)
    params_by_name = {
        _constant(key): value
        for key, value in zip(params.keys, params.values, strict=True)
        if key is not None
    }
    assert isinstance(params_by_name["q"], ast.Name)
    count = params_by_name["count"]
    assert isinstance(count, ast.Name)
    assert count.id == "_RESULT_COUNT"
    result_count_assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_RESULT_COUNT"
            for target in node.targets
        )
    ]
    assert len(result_count_assignments) == 1
    assert _constant(result_count_assignments[0].value) == 5


def test_lazy_and_cli_start_paths_delegate_to_the_same_process_owner() -> None:
    tree = _tree(MAIN_PATH)

    assert len(_calls(_function(tree, "_get_app"), "_compose_process_runtime")) == 1
    assert len(_calls(_function(tree, "start"), "_compose_process_runtime")) == 1


def test_catalog_membership_does_not_flow_into_protocol_admission() -> None:
    function = _function(_tree(BOOTSTRAP_PATH), "_compose_responses_runtime")

    assert len(_calls(function, "HostedToolExecutor")) == 1
    assert len(_calls(function, "HostedAgenticRuntimeStarter")) == 1
    mapper_calls = _calls(function, "CanonicalResponsesMapper")
    assert len(mapper_calls) == 1
    assert {keyword.arg for keyword in mapper_calls[0].keywords} == {
        "resolve_runtime",
        "projection_factory",
        "compaction_codec",
    }


def test_brave_provider_does_not_catch_cancellation_or_expose_raw_failures() -> None:
    tree = _tree(BRAVE_PATH)
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    handled = {
        ast.unparse(handler.type) for handler in handlers if handler.type is not None
    }

    assert "BaseException" not in handled
    assert "Exception" not in handled
    assert "asyncio.CancelledError" not in handled
    assert "CancelledError" not in handled
    assert "raise_for_status" not in BRAVE_PATH.read_text(encoding="utf-8")


def test_brave_setting_is_secret_bearing_and_masked() -> None:
    source = CONFIG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    settings = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Settings"
    )
    field = next(
        node
        for node in settings.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "brave_api_key"
    )
    assert ast.unparse(field.annotation) in {"SecretStr | None", "None | SecretStr"}

    to_dict = _function(ast.Module(body=settings.body, type_ignores=[]), "to_dict")
    returned = next(node for node in ast.walk(to_dict) if isinstance(node, ast.Return))
    assert isinstance(returned.value, ast.Dict)
    exported = {
        _constant(key): value
        for key, value in zip(returned.value.keys, returned.value.values, strict=True)
        if key is not None
    }
    assert "brave_api_key" in exported
    assert "***" in ast.unparse(exported["brave_api_key"])
    assert "get_secret_value" not in source


def test_canonical_composition_never_uses_the_legacy_search_registry() -> None:
    canonical_sources = (
        BRAVE_PATH.read_text(encoding="utf-8"),
        BOOTSTRAP_PATH.read_text(encoding="utf-8"),
        MAIN_PATH.read_text(encoding="utf-8"),
    )

    assert all("execute_web_search" not in source for source in canonical_sources)
    assert all("tools.builtin" not in source for source in canonical_sources)
