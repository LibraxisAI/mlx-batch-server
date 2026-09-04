# MLX Batch Runner Agent Notes

This repo follows the VetCoders living-tree workflow:

- Work in this shared directory; do not create git worktrees.
- Re-read files before editing if time has passed.
- Treat local changes as user or peer work unless you made them.
- Prefer loctree mapping before broad refactors, deletes, or risky edits.
- Unbound local bind defaults to `10240`. Production roles live in
  `src/mlx_batch_server/runtime/manifests/runtime-roles-8100-8102.json`:
  `main`/`8100` fused Flash, `canary`/`8101`, `vision`/`8102` `legacy_mlx`.
  Do not treat `10240` as a second product.
- Verify runtime-facing changes with focused tests before committing.

Current product truth:

- `mlx-batch-server` is the sole OpenAI-compatible inference owner.
  `/v1/responses` (HTTP/SSE and multiplexed WebSocket) is the primary surface.
- Role `main` on `8100` runs fused Qwen Flash (`fused_mtp_mlx`,
  revision `000544f8cddcbde27c1bc302deac2b5b4d45a5b1`). Health must keep
  reporting `tensor_batch_mode=row_serial` and `text.batch_capable=false`
  until a proven multi-row QSA path exists. Do not market true tensor batching.
- MTPLX and oMLX are bounded donors inside the fused backend, not standalone
  inference products. The legacy mlx-lm `BatchRequestCoordinator` remains a
  compatibility lane for unsupported models, never a silent Flash fallback.
- `/v1/models/load`, `/v1/models/unload`, `/v1/models/alias`, and `/admin`
  are the operator surfaces for local model residency.
- Harmony and Qwen-style reasoning streams must never emit the same content on
  both `response.output_text.*` and `response.reasoning_summary_text.*`.
