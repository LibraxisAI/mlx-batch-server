"""Embedded operator panel adapted from the Libraxis API admin backend."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from ..auth.dependency import verify_auth
from ..chat.openai.models.schema import (  # noqa: TC001
    ModelAliasRequest,
    ModelLoadRequest,
    ModelUnloadRequest,
)
from ..main import DEFAULT_CORS_ALLOW_ORIGINS

router = APIRouter(tags=["admin"])
_STARTED_AT = time.time()
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _tool_status(name: str) -> dict[str, Any]:
    bundled = _REPO_ROOT / "tools" / "bin" / "darwin-arm64" / name
    path = bundled if bundled.exists() else None
    source = "bundled" if path else None

    if path is None:
        resolved = shutil.which(name)
        path = Path(resolved) if resolved else None
        source = "path" if path else None

    status: dict[str, Any] = {
        "name": name,
        "available": path is not None,
        "source": source,
        "path": str(path) if path else None,
    }
    if path and path.exists():
        status["size_bytes"] = path.stat().st_size
    return status


def _readiness_checks(
    role_status: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cors = os.environ.get("MLX_BATCH_CORS", DEFAULT_CORS_ALLOW_ORIGINS)
    tools = [_tool_status(name) for name in ("loct", "aicx", "prview")]
    port_detail = (
        f"Role {role_status['role']} owns port {role_status['port']}."
        if role_status is not None
        else "Default runtime port is 10240; production roles are explicit."
    )
    runtime_detail: object = (
        role_status
        if role_status is not None
        else "/v1/models/loaded and /health are wired."
    )
    return [
        {
            "id": "port",
            "label": "Operator port",
            "ok": True,
            "detail": port_detail,
        },
        {
            "id": "cors",
            "label": "Tailscale CORS",
            "ok": "100.*" in cors,
            "detail": cors,
        },
        {
            "id": "runtime",
            "label": "Runtime introspection",
            "ok": True,
            "detail": runtime_detail,
        },
        {
            "id": "operator-tools",
            "label": "Operator tools",
            "ok": all(tool["available"] for tool in tools),
            "detail": tools,
        },
    ]


@router.get("/admin", response_class=HTMLResponse)
async def admin_panel(_auth: dict = Depends(verify_auth)) -> HTMLResponse:
    """Minimal landing page that points operators at the richer operator UI.

    The full htmx-based admin lives in ``mlx-batch-operator`` on port 10241;
    this inference-side ``/admin`` page is intentionally a thin stub so admins
    landing on the wrong port get routed correctly.
    """
    return HTMLResponse(_ADMIN_HTML)


@router.get("/api/admin/summary")
async def admin_summary(
    http_request: Request,
    _auth: dict = Depends(verify_auth),
) -> dict[str, Any]:
    """Return the compact admin state used by the panel."""
    role_control = getattr(http_request.app.state, "role_control_service", None)
    if role_control is None:
        from ..chat.mlx.runtime_aliases import get_runtime_aliases
        from ..chat.openai.models import models as model_routes

        health = await model_routes.health_check()
        loaded = await model_routes.list_loaded_models()
        aliases = get_runtime_aliases()
    else:
        health = role_control.health_payload()
        loaded = role_control.loaded_models_payload()
        aliases = role_control.aliases_payload()["aliases"]
    checks = _readiness_checks(
        None if role_control is None else role_control.role_status()
    )
    return {
        "status": "ok" if all(check["ok"] for check in checks) else "degraded",
        "pid": os.getpid(),
        "uptime_seconds": int(time.time() - _STARTED_AT),
        "health": health,
        "loaded": loaded,
        "aliases": aliases,
        "readiness": checks,
    }


@router.post("/api/admin/models/load")
async def admin_load_model(
    http_request: Request,
    request: ModelLoadRequest,
    _auth: dict = Depends(verify_auth),
) -> Any:
    """Load a model through the existing runtime-safe model endpoint."""
    role_control = getattr(http_request.app.state, "role_control_service", None)
    if role_control is not None:
        try:
            return await role_control.load_model(request)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    from ..chat.openai.models import models as model_routes

    return await model_routes.load_model(request)


@router.post("/api/admin/models/unload")
async def admin_unload_model(
    http_request: Request,
    request: ModelUnloadRequest | None = None,
    _auth: dict = Depends(verify_auth),
) -> Any:
    """Unload a model through the existing runtime-safe model endpoint."""
    role_control = getattr(http_request.app.state, "role_control_service", None)
    if role_control is not None:
        try:
            return await role_control.unload_model(request)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    from ..chat.openai.models import models as model_routes

    return await model_routes.unload_model(request)


@router.post("/api/admin/models/alias")
async def admin_create_alias(
    http_request: Request,
    request: ModelAliasRequest,
    _auth: dict = Depends(verify_auth),
) -> Any:
    """Create a runtime alias through the existing model endpoint."""
    role_control = getattr(http_request.app.state, "role_control_service", None)
    if role_control is not None:
        try:
            return role_control.register_alias(request)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    from ..chat.openai.models import models as model_routes

    return await model_routes.create_model_alias(request)


@router.get("/api/admin/logs/tail")
async def admin_tail_logs(
    file: str = Query(default="mlx-batch-server.log"),
    lines: int = Query(default=200, ge=1, le=2000),
    _auth: dict = Depends(verify_auth),
) -> dict[str, Any]:
    """Tail a local operator log file from the repo root."""
    path = (_REPO_ROOT / file).resolve()
    if _REPO_ROOT not in path.parents and path != _REPO_ROOT:
        raise HTTPException(status_code=400, detail="Log path must stay inside repo")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {file}")

    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"file": str(path), "lines": content[-lines:]}


_ADMIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLX Batch Operator</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111411;
      --panel: #181c18;
      --line: #30382f;
      --text: #eff5ea;
      --muted: #9daa97;
      --accent: #9cd66b;
      --warn: #e7b75f;
      --bad: #ef7d68;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    header, main { width: min(1180px, calc(100vw - 32px)); margin: 0 auto; }
    header {
      min-height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
    }
    h1 { font-size: 18px; margin: 0; letter-spacing: 0; }
    button, input, select {
      border: 1px solid var(--line);
      background: #0e100e;
      color: var(--text);
      border-radius: 6px;
      min-height: 34px;
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button:hover { border-color: var(--accent); }
    main { padding: 18px 0 28px; display: grid; gap: 14px; }
    .grid { display: grid; grid-template-columns: 1.1fr .9fr; gap: 14px; }
    section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
    }
    h2 {
      font-size: 13px;
      margin: 0 0 12px;
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
    }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .metric { border-left: 2px solid var(--accent); padding-left: 10px; min-width: 0; }
    .metric b { display: block; font-size: 22px; line-height: 1.1; }
    .metric span, .muted { color: var(--muted); }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px 6px; border-bottom: 1px solid var(--line); text-align: left; }
    th { color: var(--muted); font-weight: 600; }
    .ok { color: var(--accent); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .row { display: grid; grid-template-columns: 1fr 120px 120px; gap: 8px; }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 0;
      color: #d7dfd0;
      max-height: 320px;
      overflow: auto;
    }
    @media (max-width: 820px) {
      .grid, .metrics, .row { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; padding: 14px 0; gap: 10px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>MLX Batch Operator</h1>
      <div class="muted">localhost:10240 runtime console &middot; richer UI: <a href="http://127.0.0.1:10241/admin/" style="color:var(--accent)">operator on :10241</a></div>
    </div>
    <button id="refresh">Refresh</button>
  </header>
  <main>
    <section>
      <h2>Runtime</h2>
      <div class="metrics" id="metrics"></div>
    </section>
    <div class="grid">
      <section>
        <h2>Loaded Models</h2>
        <table><thead><tr><th>Model</th><th>Lanes</th><th>Surfaces</th></tr></thead><tbody id="models"></tbody></table>
      </section>
      <section>
        <h2>Readiness</h2>
        <table><tbody id="checks"></tbody></table>
      </section>
    </div>
    <section>
      <h2>Load Or Alias</h2>
      <div class="row">
        <input id="model" placeholder="model id or path">
        <input id="task" placeholder="task">
        <input id="alias" placeholder="alias">
      </div>
      <div style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap">
        <button id="load">Load</button>
        <button id="registerAlias">Alias Only</button>
        <button id="unloadAll">Unload All</button>
      </div>
      <pre id="action" style="margin-top:10px"></pre>
    </section>
    <section>
      <h2>Aliases</h2>
      <pre id="aliases"></pre>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const post = async (url, body) => {
      const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
      const data = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(data));
      return data;
    };
    async function refresh() {
      const data = await (await fetch("/api/admin/summary")).json();
      const mem = data.health.memory || {};
      $("metrics").innerHTML = [
        ["Status", data.status],
        ["Loaded", data.health.loaded_models_count ?? 0],
        ["RSS GB", mem.rss_gb ?? "-"],
        ["MLX GB", mem.mlx_active_gb ?? "-"],
      ].map(([k,v]) => `<div class="metric"><span>${k}</span><b>${v}</b></div>`).join("");
      const rows = data.loaded.data || [];
      $("models").innerHTML = rows.length ? rows.map((m) => `<tr><td>${m.id}</td><td>${(m.runtime?.active_lanes || []).join(", ")}</td><td>${(m.attached_tasks || []).join(", ")}</td></tr>`).join("") : `<tr><td colspan="3" class="muted">No resident models</td></tr>`;
      $("checks").innerHTML = data.readiness.map((c) => `<tr><td class="${c.ok ? "ok" : "bad"}">${c.ok ? "OK" : "FAIL"}</td><td>${c.label}</td></tr>`).join("");
      $("aliases").textContent = JSON.stringify(data.aliases, null, 2);
    }
    $("refresh").onclick = refresh;
    $("load").onclick = async () => {
      const body = {model: $("model").value, task: $("task").value || null, alias: $("alias").value || null};
      $("action").textContent = JSON.stringify(await post("/api/admin/models/load", body), null, 2);
      await refresh();
    };
    $("registerAlias").onclick = async () => {
      const body = {model: $("model").value, alias: $("alias").value};
      $("action").textContent = JSON.stringify(await post("/api/admin/models/alias", body), null, 2);
      await refresh();
    };
    $("unloadAll").onclick = async () => {
      $("action").textContent = JSON.stringify(await post("/api/admin/models/unload", {}), null, 2);
      await refresh();
    };
    refresh().catch((err) => { $("action").textContent = String(err); });
  </script>
</body>
</html>
"""
