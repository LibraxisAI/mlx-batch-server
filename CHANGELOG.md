# Changelog

All notable changes to mlx-batch-server are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/) loosely and the
project uses semver.

## [0.6.0] — 2026-04-27 (unreleased)

### Added

- Auth foundation (`mlx_batch_server.auth`): API keys, HMAC, session auth,
  rate-limit middleware, `/access` HTML registration page. Env-gated via
  `SECURITY_LEVEL` (default `0` = open).
- `/v1/ready` endpoint with rich readiness signal: process, models loaded,
  batch coordinators, config validation, plus an `auth_backends` block when
  `SECURITY_LEVEL>0`. `/health` stays as the lightweight liveness probe.
- Standalone operator backend (`mlx-batch-operator` CLI) with htmx admin UI
  on port `10241`. Tabs: Fleet, Sessions, Logs, Lifecycle, Playground.
- Operator-side conditional auth (`mlx_batch_server.operator.auth.operator_auth`)
  that inherits inference `SECURITY_LEVEL` by default and can be overridden
  per side via `MLX_BATCH_OPERATOR_SECURITY_LEVEL` / `MLX_BATCH_OPERATOR_REQUIRE_AUTH`.
- Optional install extras: `[auth]` (redis, pyjwt), `[operator]` (jinja2,
  python-multipart, ruamel.yaml).
- Integration test suite (`tests/integration/`) covering inference + operator
  + auth + loopback playground proxy across open and gated modes.
- HF model card publishing toolkit:
  - `templates/HF_MODEL_CARD.md` canonical card template (fixed
    `## Inference tested on` section, canonical Vibecrafted footer).
  - `scripts/rewrite_hf_model_cards.py` (commit `fcfaf1c`) — full rewrite of
    every LibraxisAI HF card from the template.
  - `scripts/backfill_hf_inference_section.py` — conservative backfill of the
    `## Inference tested on` link into cards that lack it.
  - `scripts/backfill_hf_canonical_footer.py` — conservative backfill of the
    canonical Vibecrafted footer into cards that lack any form of it.
  - Makefile targets: `hf-rewrite`, `hf-backfill-inference`,
    `hf-backfill-footer` (dry-run by default; `*-apply` variants push).

### Changed

- Inference admin endpoints (`/admin`, `/api/admin/summary`,
  `/api/admin/models/{load,unload,alias}`, `/api/admin/logs/tail`) now sit
  behind `verify_auth` — gated when `SECURITY_LEVEL>0`, transparent when `=0`.
- Inference-side `/admin` HTML landing now links to the richer operator UI
  on port `10241` so admins landing on the wrong port get pointed there.
- `vlm_batch.py` — `group_by_shape` is configurable per coordinator (commit
  `51f423c`).
- `responses/router.py` — VLM stream batches grouped by image shape (commit
  `51f423c`).
- `chat/mlx/tools/chat_template.py` — robust `apply_chat_template` signature
  support covering Gemma4 and legacy templates (commit `3e30f7a`).

### Fixed

- VLM stream batch contract for `mlx-vlm` (commit `cd90067`).
- Gemma4 template kwarg signature, VLM uid tuple unpacking, reasoning
  misclassification (commit `3e30f7a`).

### Security

- Business inference endpoints (`/v1/responses`, `/v1/chat/completions`,
  `/v1/embeddings`, etc.) now respect `SECURITY_LEVEL` — full inference surface
  requires auth when enabled.
- Inference admin endpoints (`/api/admin/models/*`, `/api/admin/summary`,
  `/api/admin/logs/tail`, `/admin`) now respect `SECURITY_LEVEL` — model
  load/unload/alias and log access require auth when enabled. Closes a real
  hole where the inference admin surface was open even with auth on the
  rest of the API.
- Operator surface (lifecycle, models, sessions, logs, playground proxy,
  htmx admin) respects `SECURITY_LEVEL` and inherits the inference posture
  unless explicitly overridden. `/health` and `/api/health` always open for
  load balancers.

### Removed

- Vendored `mlx-batch-server` git submodule (replaced by direct source —
  commit `a80ba11`).

---

The auth router family (`/auth/*`, `/hmac/*`, `/access`) is **opt-in** and
only mounts when at least one auth-related env var is configured. Existing
deployments keep working unchanged on `SECURITY_LEVEL=0`.
