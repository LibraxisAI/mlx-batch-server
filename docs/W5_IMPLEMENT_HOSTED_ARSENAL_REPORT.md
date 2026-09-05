# W5-IMPLEMENT Hosted Arsenal Source Report

## W5c recovery correction

The production composition source contract again reads the actual working-tree
files for every owner, including the intentionally sequenced P6 wiring. It no
longer substitutes committed Git objects or references an undefined local.

After a hosted failure, the fixed runtime-authored disclosure remains the
minimum final user answer. The runtime still performs exactly one no-tools,
`tool_choice=none` model continuation. If inference itself completes but emits
no prose, the outer turn now completes on that disclosure; real inference
failure, cancellation and deadline expiry retain their failure semantics. The
existing literal contradiction filter is documented as a bounded guard, not
as proof of arbitrary semantic honesty.

The new source RED contract covers one failed receipt, one fixed disclosure,
one empty completed continuation and one `TurnCompleted` with no `TurnFailed`.
It was authored but not executed under Compile Embargo.

## W5b independent-review correction

This follow-up is source-only under Compile Embargo. It adds request-global
hosted `call_id` authority, action-round prose quarantine, a deterministic
runtime-authored failure disclosure, PDF extracted-text token limiting, and a
transport-issued SafePublicFetch receipt covering every redirect/DNS/pin/peer
hop plus bounded content decoding. A typed `fetch_rate_limited` outcome makes
Anthropic `too_many_requests` reachable without admitting response bodies or
topology into error text.

The production composition oracles read the actual working tree, including the
excluded P6 files, so local wiring changes cannot evade their structural
assertions. P6 remains sequenced after W2c and excluded from W5c staging.

Authored tests are RED contracts and were not executed. BUILD, LINT, TEST and
RUNTIME remain NOT_ASSESSED.

## Identity

- Runtime class: Fleet Worktree
- Worker root: `/Users/tester/.vibecrafted/worktrees/LibraxisAI/mlx-batch-server/2026_0906/W5-hosted-arsenal-implementation`
- Baseline branch: detached from canonical runner integration tip
- Baseline SHA: `d43f194fccf17c2097e6b193a785cadcca510bb3`
- Worker branch: `cut/W5-hosted-arsenal-implementation`
- Integration disposition: isolated; not integrated and not pushed
- Recovery marker after architecture convergence: exact `W2_STRUCTURALLY_CLOSED`

The historical resume payload named by the original conversation was absent at
the supplied path. This cut therefore uses the current Founder mandate,
canonical runner source, Compile Embargo contract and accepted HR1 design
quarry as its authority. It does not reconstruct missing history.

## Source shape completed in the core baton

- `tools/hosted.py` owns the immutable request plan, per-call execution policy,
  three round modes, four internal model descriptors, closed result/action
  unions and the single hosted error/receipt taxonomy.
- `tools/hosted_web.py` owns provider search, prior-context-bound OpenAI
  `open_page`/`find_in_page`, Anthropic `web_fetch`, the 250-character URL
  boundary, deterministic token truncation and bounded PDF base64 projection.
- `utils/safe_public_fetch.py` remains the sole network owner. It records real
  status/redirect provenance while preserving literal-IP connect pinning,
  mixed-answer refusal and redirect-hop re-resolution.
- `runtime/agentic.py` computes the plan once, derives policy per execution
  round, preserves action-selection `tool_choice`, forces successful follow-up
  to `auto`, and forces the only failure continuation to `tools=()` plus
  `tool_choice=none`. A failed hosted call is fed back as a typed tool result;
  it cannot directly become `tool_stream_failed`, an empty assistant success or
  `runtime_start_failed`.
- `runtime/events.py`, `runtime/turn.py`, Responses transport/projectors and the
  Anthropic mapper/projector close the OpenAI search/open/find and Anthropic
  text/PDF/error protocol mappings without `libraxis_tool_output`.
- `runtime/hosted_evidence.py` is a bounded, protocol-neutral, delivery-ordered
  ledger. Call and terminal candidates remain invisible until the corresponding
  sink sequence returns; failed delivery never satisfies acceptance.
- Anthropic native `request-id` is carried only through internal turn metadata
  to the evidence correlation seam.

## Source RED contracts authored or corrected

- `tests/runtime/test_hosted_evidence.py`
- `tests/runtime/test_agentic_hosted_tools.py`
- `tests/runtime/test_events_contract_red.py`
- `tests/responses/test_transport_contract_red.py`
- `tests/tools/test_hosted_web.py`
- `tests/tools/test_hosted_result_payloads.py`
- `tests/chat/anthropic/test_anthropic_hosted_web_fetch.py`

The source contracts cover initial and later-round failures, partial parallel
success/failure, a non-empty final model reply, exact failure-continuation
capabilities, the no-tool oracle, request-policy isolation, URL 250/251,
open/find bounds, PDF byte equality and evidence prepare/deliver/fail ordering.

## Sequenced P6 wiring, excluded from the core commit

W2b legitimately owns `src/mlx_batch_server/responses/runtime_bootstrap.py` to
remove the forgeable production `execution_factory` seam. The W5 hosted-profile
hunk in that file is therefore frozen and must land only after W2b. The
dependent `src/mlx_batch_server/main.py` hunk and new
`src/mlx_batch_server/runtime/hosted_evidence_router.py` are kept outside the
core baton with it. Replay must preserve W2b's constructor authority and must
not restore or forward `execution_factory`.

The pending P6 wiring composes provider-present/provider-absent profiles and an
authenticated localhost evidence route. Its current deadline-short draft is
not admissible yet: it still needs the accepted controlled blocking transport,
pre-admitted manifest slow URL, and exact bind/role receipt tests. It must not
be used as live deadline evidence in its present form.

## Deferred acceptance

Under the Compile Embargo this cut ran no imports, test runner, formatter,
linter, type checker, build, dependency resolution, service/model start or
runtime probe. `git diff --check` is the only mechanical source check allowed.

- BUILD: NOT_ASSESSED
- LINT: NOT_ASSESSED
- TEST: NOT_ASSESSED
- RUNTIME: NOT_ASSESSED
- LIVE ACCEPTANCE: NOT_ASSESSED

The post-W2b recovery point is: replay P6 without `execution_factory`, author
the deterministic deadline-short transport/profile tests, then wait for exact
`W2_STRUCTURALLY_CLOSED` before running the repository-owned W2/W3 gates and
three-profile live verifier/combiner. No source-only result in this report is a
claim that Buddy or hosted tools are live.
