#!/usr/bin/env python3
"""Fail-closed phase gate for the MLX Batch API compile embargo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

MARKER_ENV = "MLX_BATCH_EMBARGO_V1"
MARKER_SCHEMA = "mlx-batch-compile-embargo.v1"
PLAN_ID = "mlx-batch-api-conformance-v1"
PHASE = "W1-source-shape"
RELEASE_ATTESTATION = "I1_STRUCTURALLY_CLOSED"
DEFERRED_GATES = frozenset({"ruff", "ruff-format", "mypy"})
GATE_COMMAND_PREFIXES = {
    "ruff": ("ruff", "check"),
    "ruff-format": ("ruff", "format"),
    "mypy": ("mypy",),
    "bandit": ("bandit",),
}
MARKER_KEYS = frozenset(
    {"schema", "plan_id", "phase", "deferred_gates", "release_attestation"}
)


class PhaseGateError(ValueError):
    """The embargo marker or invocation is unsafe."""


class GateDecision(StrEnum):
    RUN = "run"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class CompileEmbargoMarker:
    schema: str
    plan_id: str
    phase: str
    deferred_gates: frozenset[str]
    release_attestation: str


def marker_template() -> dict[str, object]:
    """Return the only marker payload accepted during W1."""

    return {
        "schema": MARKER_SCHEMA,
        "plan_id": PLAN_ID,
        "phase": PHASE,
        "deferred_gates": sorted(DEFERRED_GATES),
        "release_attestation": RELEASE_ATTESTATION,
    }


def parse_marker(raw: str) -> CompileEmbargoMarker:
    """Parse an exact marker; extra, missing, or mistyped fields fail closed."""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PhaseGateError(f"{MARKER_ENV} must be valid JSON") from error
    if not isinstance(payload, dict):
        raise PhaseGateError(f"{MARKER_ENV} must be a JSON object")
    if set(payload) != MARKER_KEYS:
        missing = sorted(MARKER_KEYS - set(payload))
        extra = sorted(set(payload) - MARKER_KEYS)
        raise PhaseGateError(
            f"{MARKER_ENV} keys do not match policy; missing={missing}, extra={extra}"
        )
    deferred = payload["deferred_gates"]
    if not isinstance(deferred, list) or any(
        not isinstance(item, str) for item in deferred
    ):
        raise PhaseGateError("deferred_gates must be a JSON string array")
    if len(deferred) != len(set(deferred)):
        raise PhaseGateError("deferred_gates must not contain duplicates")

    marker = CompileEmbargoMarker(
        schema=_required_string(payload, "schema"),
        plan_id=_required_string(payload, "plan_id"),
        phase=_required_string(payload, "phase"),
        deferred_gates=frozenset(deferred),
        release_attestation=_required_string(payload, "release_attestation"),
    )
    expected = CompileEmbargoMarker(
        schema=MARKER_SCHEMA,
        plan_id=PLAN_ID,
        phase=PHASE,
        deferred_gates=DEFERRED_GATES,
        release_attestation=RELEASE_ATTESTATION,
    )
    if marker != expected:
        raise PhaseGateError(
            f"{MARKER_ENV} does not exactly attest the repository W1 policy"
        )
    return marker


def decide_gate(gate: str, environ: Mapping[str, str]) -> GateDecision:
    """Return RUN without a marker and DEFER only for the exact allowlist."""

    if not gate:
        raise PhaseGateError("gate name must not be empty")
    raw = environ.get(MARKER_ENV)
    if raw is None:
        return GateDecision.RUN
    marker = parse_marker(raw)
    if gate in marker.deferred_gates:
        return GateDecision.DEFER
    return GateDecision.RUN


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise PhaseGateError(f"{key} must be a non-empty string")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--gate")
    action.add_argument("--validate-marker", action="store_true")
    action.add_argument("--forbid-marker", action="store_true")
    action.add_argument("--print-marker", action="store_true")
    parser.add_argument("--surface", default="repository")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _handle_policy_action(args: argparse.Namespace, raw: str | None) -> int | None:
    if args.print_marker:
        print(json.dumps(marker_template(), separators=(",", ":")))
        return 0
    if args.forbid_marker:
        if raw is not None:
            raise PhaseGateError(
                f"{MARKER_ENV} is forbidden at the {args.surface} ref boundary"
            )
        print(f"PHASE_GATE=run surface={args.surface} marker=absent")
        return 0
    if args.validate_marker:
        if raw is None:
            print("PHASE_GATE=run marker=absent")
        else:
            marker = parse_marker(raw)
            print(f"PHASE_GATE=valid plan={marker.plan_id} phase={marker.phase}")
        return 0
    return None


def _run_gate(args: argparse.Namespace) -> int:
    decision = decide_gate(args.gate, os.environ)
    if decision is GateDecision.DEFER:
        print(f"PHASE_GATE=deferred gate={args.gate} plan={PLAN_ID} phase={PHASE}")
        return 0
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise PhaseGateError(f"gate {args.gate!r} has no command to run")
    expected_prefix = GATE_COMMAND_PREFIXES.get(args.gate)
    if (
        expected_prefix is None
        or tuple(command[: len(expected_prefix)]) != expected_prefix
    ):
        raise PhaseGateError(
            f"gate {args.gate!r} command must start with {expected_prefix!r}"
        )
    print(f"PHASE_GATE=run gate={args.gate}", flush=True)
    # The executable and required subcommand are fixed by GATE_COMMAND_PREFIXES;
    # remaining argv is passed without a shell to the repository-owned tool.
    os.execvp(command[0], command)  # nosec B606
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy_result = _handle_policy_action(args, os.environ.get(MARKER_ENV))
        return policy_result if policy_result is not None else _run_gate(args)
    except PhaseGateError as error:
        print(f"PHASE_GATE=blocked reason={error}", file=sys.stderr)
    except OSError as error:
        print(
            f"PHASE_GATE=blocked reason=cannot execute gate: {error}", file=sys.stderr
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
