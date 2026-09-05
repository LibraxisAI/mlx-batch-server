#!/usr/bin/env python3
"""Repository-owned, fail-closed Compile Embargo phase guard."""

from __future__ import annotations

import argparse
import os
import subprocess  # nosec B404 -- only fixed git/show reads are permitted
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_RELATIVE_PATH = Path(".vibecrafted/embargo.toml")
STATE_PATH = REPO_ROOT / STATE_RELATIVE_PATH

STATE_SCHEMA = "mlx-batch-compile-embargo.v2"
PLAN_ID = "mlx-batch-api-conformance-v1"
W1_PHASE = "W1_SOURCE_SHAPE"
W2_PHASE = "W2_INTEGRATION"
RELEASE_PHASE = "W2_STRUCTURALLY_CLOSED"
OPEN_PHASES = frozenset({W1_PHASE, W2_PHASE})
DEFERRED_GATES = ("mypy", "ruff", "ruff-format")
STATE_KEYS = frozenset({"schema", "plan_id", "phase", "deferred_gates"})

GATE_COMMAND_PREFIXES = {
    "ruff": ("ruff", "check"),
    "ruff-format": ("ruff", "format"),
    "mypy": ("mypy",),
    "bandit": ("bandit",),
}
NON_DEFERRED_PROBES = (
    "bandit",
    "semgrep",
    "provenance",
    "check-ast",
    "check-merge-conflict",
    "detect-private-key",
    "check-json",
    "check-toml",
    "check-added-large-files",
    "merge-markers-warn",
    "merge-markers-block",
    "pre-push",
    "ref-safety",
)


class PhaseGateError(ValueError):
    """The tracked embargo state or requested transition is unsafe."""


class GateDecision(StrEnum):
    RUN = "run"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class CompileEmbargoState:
    schema: str
    plan_id: str
    phase: str
    deferred_gates: tuple[str, ...]

    @property
    def is_open(self) -> bool:
        return self.phase in OPEN_PHASES


def parse_state_text(raw: str) -> CompileEmbargoState:
    """Parse the exact flat TOML schema and enforce phase-specific state."""

    try:
        payload = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise PhaseGateError("embargo state must be valid TOML") from error
    if set(payload) != STATE_KEYS:
        missing = sorted(STATE_KEYS - set(payload))
        extra = sorted(set(payload) - STATE_KEYS)
        raise PhaseGateError(
            f"embargo state keys do not match policy; missing={missing}, extra={extra}"
        )

    schema = _required_string(payload, "schema")
    plan_id = _required_string(payload, "plan_id")
    phase = _required_string(payload, "phase")
    deferred_raw = payload["deferred_gates"]
    if not isinstance(deferred_raw, list) or any(
        not isinstance(item, str) for item in deferred_raw
    ):
        raise PhaseGateError("deferred_gates must be a TOML string array")
    deferred = tuple(deferred_raw)
    if len(deferred) != len(set(deferred)):
        raise PhaseGateError("deferred_gates must not contain duplicates")

    if schema != STATE_SCHEMA:
        raise PhaseGateError(f"schema must equal {STATE_SCHEMA!r}")
    if plan_id != PLAN_ID:
        raise PhaseGateError(f"plan_id must equal {PLAN_ID!r}")
    if phase not in {*OPEN_PHASES, RELEASE_PHASE}:
        raise PhaseGateError(f"unsupported embargo phase {phase!r}")
    expected_deferred = DEFERRED_GATES if phase in OPEN_PHASES else ()
    if deferred != expected_deferred:
        raise PhaseGateError(
            f"phase {phase!r} requires deferred_gates={list(expected_deferred)!r}"
        )
    return CompileEmbargoState(schema, plan_id, phase, deferred)


def decide_gate(gate: str, state: CompileEmbargoState) -> GateDecision:
    """Defer only the exact allowlist while the tracked embargo is open."""

    if not gate:
        raise PhaseGateError("gate name must not be empty")
    if state.is_open and gate in state.deferred_gates:
        return GateDecision.DEFER
    return GateDecision.RUN


def load_index_state() -> CompileEmbargoState:
    """Read one state only when tracked index and worktree bytes are identical."""

    try:
        stat_result = STATE_PATH.lstat()
    except FileNotFoundError as error:
        raise PhaseGateError(f"tracked embargo state is missing: {STATE_PATH}") from error
    if STATE_PATH.is_symlink() or not STATE_PATH.is_file():
        raise PhaseGateError("tracked embargo state must be one regular non-symlink file")
    if stat_result.st_size > 4096:
        raise PhaseGateError("tracked embargo state exceeds 4096 bytes")

    worktree_raw = STATE_PATH.read_bytes()
    index_raw = _git_blob(":.vibecrafted/embargo.toml", source="index")
    if worktree_raw != index_raw:
        raise PhaseGateError("embargo state differs between index and worktree")
    return _parse_utf8(worktree_raw, source="index/worktree")


def load_head_state() -> CompileEmbargoState:
    """Read committed state for a ref-boundary decision."""

    raw = _git_blob("HEAD:.vibecrafted/embargo.toml", source="HEAD")
    return _parse_utf8(raw, source="HEAD")


def require_closed_at_ref_boundary(
    state: CompileEmbargoState, *, surface: str
) -> None:
    if state.phase != RELEASE_PHASE:
        raise PhaseGateError(
            f"open embargo phase {state.phase!r} is forbidden at {surface}"
        )


def _git_blob(spec: str, *, source: str) -> bytes:
    result = subprocess.run(  # nosec B603 B607 -- fixed git/show contract
        ("git", "show", spec),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise PhaseGateError(f"cannot read tracked embargo state from {source}: {detail}")
    return result.stdout


def _parse_utf8(raw: bytes, *, source: str) -> CompileEmbargoState:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PhaseGateError(f"embargo state from {source} must be UTF-8") from error
    return parse_state_text(text)


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise PhaseGateError(f"{key} must be a non-empty string")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--gate")
    action.add_argument("--validate-state", action="store_true")
    action.add_argument("--forbid-open", action="store_true")
    action.add_argument("--self-probe", action="store_true")
    parser.add_argument("--surface", default="repository")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _run_gate(args: argparse.Namespace, state: CompileEmbargoState) -> int:
    gate = str(args.gate)
    decision = decide_gate(gate, state)
    if decision is GateDecision.DEFER:
        print(
            f"PHASE_GATE=deferred gate={gate} plan={state.plan_id} phase={state.phase}"
        )
        return 0
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    expected_prefix = GATE_COMMAND_PREFIXES.get(gate)
    if not command or expected_prefix is None:
        raise PhaseGateError(f"gate {gate!r} has no permitted command")
    if tuple(command[: len(expected_prefix)]) != expected_prefix:
        raise PhaseGateError(
            f"gate {gate!r} command must start with {expected_prefix!r}"
        )
    print(f"PHASE_GATE=run gate={gate} phase={state.phase}", flush=True)
    os.execvp(command[0], command)  # nosec B606
    return 0


def _self_probe() -> int:
    state = load_index_state()
    if state.phase != W1_PHASE:
        raise PhaseGateError(f"W0 self-probe requires phase {W1_PHASE!r}")
    deferred = tuple(
        gate for gate in DEFERRED_GATES if decide_gate(gate, state) is GateDecision.DEFER
    )
    if deferred != DEFERRED_GATES:
        raise PhaseGateError("W0 self-probe did not defer the exact allowlist")
    if any(decide_gate(gate, state) is not GateDecision.RUN for gate in NON_DEFERRED_PROBES):
        raise PhaseGateError("W0 self-probe deferred a security/provenance/hygiene gate")

    closed = parse_state_text(
        f'schema = "{STATE_SCHEMA}"\n'
        f'plan_id = "{PLAN_ID}"\n'
        f'phase = "{RELEASE_PHASE}"\n'
        "deferred_gates = []\n"
    )
    if any(decide_gate(gate, closed) is not GateDecision.RUN for gate in DEFERRED_GATES):
        raise PhaseGateError("closed W2 state still defers a gate")
    require_closed_at_ref_boundary(closed, surface="self-probe")
    try:
        require_closed_at_ref_boundary(state, surface="self-probe")
    except PhaseGateError:
        pass
    else:
        raise PhaseGateError("open state passed the ref-boundary check")

    invalid_states = (
        STATE_PATH.read_text(encoding="utf-8") + 'extra = "forbidden"\n',
        STATE_PATH.read_text(encoding="utf-8").replace(PLAN_ID, "another-plan"),
        STATE_PATH.read_text(encoding="utf-8").replace(W1_PHASE, "UNKNOWN"),
        STATE_PATH.read_text(encoding="utf-8").replace(
            '["mypy", "ruff", "ruff-format"]', "[]"
        ),
        STATE_PATH.read_text(encoding="utf-8").replace(
            '"ruff-format"]', '"ruff-format", "bandit"]'
        ),
    )
    for raw in invalid_states:
        try:
            parse_state_text(raw)
        except PhaseGateError:
            continue
        raise PhaseGateError("W0 self-probe accepted a tampered state")

    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    for gate in DEFERRED_GATES:
        if config.count(f"--gate {gate} --") != 1:
            raise PhaseGateError(f"pre-commit must wrap {gate!r} exactly once")
    for gate in NON_DEFERRED_PROBES:
        if f"--gate {gate} --" in config:
            raise PhaseGateError(f"non-deferred gate {gate!r} is wrapped")
    if "--validate-state" not in config or "--forbid-open --surface pre-push" not in config:
        raise PhaseGateError("pre-commit/ref-boundary wiring is incomplete")

    print(
        "W0_SELF_PROBE=green "
        f"source=index+worktree schema={state.schema} plan={state.plan_id} "
        f"phase={state.phase} deferred={','.join(state.deferred_gates)}"
    )
    print(
        f"W0_RELEASE={RELEASE_PHASE} deferred=none "
        "ref_boundary_source=HEAD non_deferred=security,provenance,hygiene"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.self_probe:
            return _self_probe()
        if args.forbid_open:
            state = load_head_state()
            require_closed_at_ref_boundary(state, surface=args.surface)
            print(f"PHASE_GATE=run surface={args.surface} phase={state.phase}")
            return 0

        state = load_index_state()
        if args.validate_state:
            print(
                f"PHASE_GATE=valid plan={state.plan_id} phase={state.phase} "
                f"source=index+worktree"
            )
            return 0
        return _run_gate(args, state)
    except PhaseGateError as error:
        print(f"PHASE_GATE=blocked reason={error}", file=sys.stderr)
    except OSError as error:
        print(f"PHASE_GATE=blocked reason=cannot execute gate: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
