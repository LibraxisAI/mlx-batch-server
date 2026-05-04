# MLX Batch Runner Agent Notes

This repo follows the VetCoders living-tree workflow:

- Work in this shared directory; do not create git worktrees.
- Re-read files before editing if time has passed.
- Treat local changes as user or peer work unless you made them.
- Prefer loctree mapping before broad refactors, deletes, or risky edits.
- Use port `10240` for local runtime checks; `8100` is production.
- Verify runtime-facing changes with focused tests before committing.

Current product truth:

- `/v1/responses` is the primary OpenAI-compatible surface.
- `/v1/models/load`, `/v1/models/unload`, `/v1/models/alias`, and `/admin`
  are the operator surfaces for local model residency.
- Harmony and Qwen-style reasoning streams must never emit the same content on
  both `response.output_text.*` and `response.reasoning_summary_text.*`.
