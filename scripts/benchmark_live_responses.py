#!/usr/bin/env python3
"""Benchmark a live OpenAI Responses SSE endpoint with usage-backed metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

EXPECTED_TEXT = " ".join(str(value) for value in range(1, 33))
VISIBLE_EVENTS = {
    "response.output_text.delta",
    "response.reasoning_summary_text.delta",
    "response.reasoning_text.delta",
}
TERMINAL_EVENTS = {
    "response.completed",
    "response.failed",
    "response.incomplete",
    "response.cancelled",
}
TENSOR_FORWARD_SCHEMA = "qwen4-exp.tensor-forward.v1"
TARGET_FORWARD_PHASES = (
    "target_decode",
    "target_verify",
    "target_correction",
)
MTP_FORWARD_PHASES = ("mtp_draft", "mtp_history_update")


@dataclass(frozen=True)
class Sample:
    concurrency: int
    round_index: int
    request_index: int
    prompt_nonce: str
    response_id: str | None
    status: str
    first_event_s: float | None
    visible_ttft_s: float | None
    latency_s: float
    decode_s: float | None
    input_tokens: int
    output_tokens: int
    decode_tps: float | None
    exact: bool
    output_text: str
    error: str | None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _jain(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    numerator = sum(positive) ** 2
    denominator = len(positive) * sum(value * value for value in positive)
    return numerator / denominator if denominator else None


def _usage_from_event(event: dict[str, Any]) -> tuple[int, int] | None:
    response = event.get("response")
    candidates = [event.get("usage")]
    if isinstance(response, dict):
        candidates.append(response.get("usage"))
    for usage in candidates:
        if isinstance(usage, dict):
            return int(usage.get("input_tokens") or 0), int(
                usage.get("output_tokens") or 0
            )
    return None


def _response_id_from_event(event: dict[str, Any]) -> str | None:
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    value = event.get("response_id") or event.get("id")
    return value if isinstance(value, str) else None


async def _snapshot(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for name, path in (("ready", "/v1/ready"), ("health", "/health")):
        try:
            response = await client.get(f"{base_url}{path}")
            response.raise_for_status()
            snapshots[name] = response.json()
        except Exception as exc:  # benchmark evidence must retain probe failures
            snapshots[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return snapshots


async def _run_sample(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    headers: dict[str, str],
    reasoning_effort: str,
    max_output_tokens: int,
    concurrency: int,
    round_index: int,
    request_index: int,
    prompt_nonce: str,
) -> Sample:
    prompt = (
        "Zignoruj identyfikator w nawiasie i zwroc dokladnie ponizszy ciag, "
        "bez komentarza i bez formatowania. "
        f"({prompt_nonce})\n{EXPECTED_TEXT}"
    )
    payload = {
        "model": model,
        "input": prompt,
        "stream": True,
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning_effort},
        "store": False,
    }
    started = time.perf_counter()
    first_event: float | None = None
    first_visible: float | None = None
    completed = started
    response_id: str | None = None
    status = "missing_terminal"
    input_tokens = 0
    output_tokens = 0
    output_parts: list[str] = []
    error: str | None = None

    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/responses",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                now = time.perf_counter()
                if first_event is None:
                    first_event = now
                response_id = response_id or _response_id_from_event(event)
                event_type = str(event.get("type") or "")
                delta = event.get("delta")
                if event_type in VISIBLE_EVENTS and isinstance(delta, str) and delta:
                    first_visible = first_visible or now
                if event_type == "response.output_text.delta" and isinstance(
                    delta, str
                ):
                    output_parts.append(delta)
                usage = _usage_from_event(event)
                if usage is not None:
                    input_tokens, output_tokens = usage
                if event_type in TERMINAL_EVENTS:
                    status = event_type.removeprefix("response.")
                    completed = now
    except Exception as exc:
        completed = time.perf_counter()
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    output_text = "".join(output_parts).strip()
    latency = completed - started
    decode_s = completed - first_visible if first_visible is not None else None
    decode_tps = (
        output_tokens / decode_s
        if output_tokens > 0 and decode_s is not None and decode_s > 0
        else None
    )
    return Sample(
        concurrency=concurrency,
        round_index=round_index,
        request_index=request_index,
        prompt_nonce=prompt_nonce,
        response_id=response_id,
        status=status,
        first_event_s=(first_event - started) if first_event is not None else None,
        visible_ttft_s=(first_visible - started) if first_visible is not None else None,
        latency_s=latency,
        decode_s=decode_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        decode_tps=decode_tps,
        exact=output_text == EXPECTED_TEXT,
        output_text=output_text,
        error=error,
    )


def _summarize(samples: list[Sample], wall_s: float) -> dict[str, Any]:
    completed = [sample for sample in samples if sample.status == "completed"]
    exact = [sample for sample in completed if sample.exact]
    ttft = [sample.visible_ttft_s for sample in exact if sample.visible_ttft_s]
    latency = [sample.latency_s for sample in exact]
    decode_tps = [sample.decode_tps for sample in exact if sample.decode_tps]
    exact_output_tokens = sum(sample.output_tokens for sample in exact)
    completed_output_tokens = sum(sample.output_tokens for sample in completed)
    return {
        "requests": len(samples),
        "completed": len(completed),
        "exact": len(exact),
        "invalid_outputs": len(completed) - len(exact),
        "errors": sum(sample.error is not None for sample in samples),
        "performance_cohort": "completed_and_exact",
        "wall_s": wall_s,
        "requests_per_s": len(exact) / wall_s if wall_s else None,
        "completed_requests_per_s": len(completed) / wall_s if wall_s else None,
        "aggregate_output_tps": exact_output_tokens / wall_s if wall_s else None,
        "completed_output_tps": completed_output_tokens / wall_s if wall_s else None,
        "visible_ttft_p50_s": statistics.median(ttft) if ttft else None,
        "visible_ttft_p95_s": _percentile(ttft, 0.95),
        "latency_p50_s": statistics.median(latency) if latency else None,
        "latency_p95_s": _percentile(latency, 0.95),
        "decode_tps_p10": _percentile(decode_tps, 0.10),
        "decode_tps_median": statistics.median(decode_tps) if decode_tps else None,
        "jain_decode_fairness": _jain(decode_tps) if len(decode_tps) >= 2 else None,
    }


def _mtp_counters(snapshot: dict[str, Any]) -> dict[str, Any]:
    try:
        stats = snapshot["ready"]["role_runtime"]["runtime_stats"]
        driver = stats["executor"]["driver"]
        return {
            "mtp": stats["mtp"],
            "autoregressive": stats["autoregressive"],
            "tensor_batch_mode": driver["tensor_batch_mode"],
            "prefill_rows": driver["prefill_rows"],
            "decode_rows": driver["decode_rows"],
        }
    except (KeyError, TypeError):
        return {}


def _tensor_forward_counters(snapshot: dict[str, Any]) -> dict[str, Any]:
    try:
        telemetry = snapshot["ready"]["role_runtime"]["runtime_stats"]["executor"][
            "driver"
        ]["tensor_forward"]
    except (KeyError, TypeError):
        return {}
    return telemetry if isinstance(telemetry, dict) else {}


def _counter(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _shape_rows(shape: str) -> int | None:
    batch, separator, sequence = shape.partition("x")
    if separator != "x" or not batch.isdigit() or not sequence.isdigit():
        return None
    batch_rows = int(batch)
    sequence_length = int(sequence)
    if batch_rows < 1 or sequence_length < 1:
        return None
    return batch_rows


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("batch row requirements must be positive")
    return parsed


def _tensor_forward_phase_delta(
    phase: str,
    before: object,
    after: object,
    errors: list[str],
) -> dict[str, Any]:
    phase_before = before if isinstance(before, dict) else {}
    phase_after = after if isinstance(after, dict) else {}
    if not isinstance(before, dict) or not isinstance(after, dict):
        errors.append(f"tensor forward phase is missing: {phase}")

    counters: dict[str, int] = {}
    for name in ("completed_calls", "completed_rows", "max_completed_rows"):
        before_value = _counter(phase_before.get(name))
        after_value = _counter(phase_after.get(name))
        if before_value is None or after_value is None:
            errors.append(f"tensor forward counter is invalid: {phase}.{name}")
            counters[name] = 0
            continue
        delta = after_value - before_value
        if name != "max_completed_rows" and delta < 0:
            errors.append(f"tensor forward counter regressed: {phase}.{name}")
        if name == "max_completed_rows" and after_value < before_value:
            errors.append(f"tensor forward maximum regressed: {phase}.{name}")
        counters[name] = delta

    shapes_before = phase_before.get("completed_calls_by_shape")
    shapes_after = phase_after.get("completed_calls_by_shape")
    if not isinstance(shapes_before, dict) or not isinstance(shapes_after, dict):
        errors.append(f"tensor forward histogram is missing: {phase}")
        shapes_before = {}
        shapes_after = {}
    histogram: dict[str, int] = {}
    for shape in sorted(set(shapes_before) | set(shapes_after)):
        if not isinstance(shape, str) or _shape_rows(shape) is None:
            errors.append(f"tensor forward shape is not canonical BxS: {shape!r}")
            continue
        before_value = _counter(shapes_before.get(shape, 0))
        after_value = _counter(shapes_after.get(shape, 0))
        if before_value is None or after_value is None:
            errors.append(f"tensor forward histogram count is invalid: {phase}.{shape}")
            continue
        delta = after_value - before_value
        if delta < 0:
            errors.append(f"tensor forward histogram regressed: {phase}.{shape}")
        histogram[shape] = delta
    return {**counters, "completed_calls_by_shape": histogram}


def _tensor_forward_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    before_schema = before.get("schema")
    after_schema = after.get("schema")
    if before_schema != TENSOR_FORWARD_SCHEMA or after_schema != TENSOR_FORWARD_SCHEMA:
        errors.append(f"tensor forward schema must be {TENSOR_FORWARD_SCHEMA}")
    before_instance = before.get("runtime_instance_id")
    after_instance = after.get("runtime_instance_id")
    same_instance = (
        isinstance(before_instance, str)
        and bool(before_instance)
        and before_instance == after_instance
    )
    if not same_instance:
        errors.append("tensor runtime instance changed between snapshots")

    before_phases = before.get("phases")
    after_phases = after.get("phases")
    if not isinstance(before_phases, dict) or not isinstance(after_phases, dict):
        errors.append("tensor forward phase maps are missing")
        before_phases = {}
        after_phases = {}

    phase_deltas: dict[str, dict[str, Any]] = {}
    for phase in (*TARGET_FORWARD_PHASES, *MTP_FORWARD_PHASES):
        phase_deltas[phase] = _tensor_forward_phase_delta(
            phase,
            before_phases.get(phase),
            after_phases.get(phase),
            errors,
        )

    return {
        "schema": TENSOR_FORWARD_SCHEMA,
        "runtime_instance_id": after_instance,
        "same_runtime_instance": same_instance,
        "valid": not errors,
        "errors": errors,
        "phases": phase_deltas,
    }


def _has_positive_batch_delta(
    phases: dict[str, Any],
    names: tuple[str, ...],
    minimum_rows: int,
) -> bool:
    for name in names:
        phase = phases.get(name)
        if not isinstance(phase, dict):
            continue
        histogram = phase.get("completed_calls_by_shape")
        if not isinstance(histogram, dict):
            continue
        for shape, count in histogram.items():
            rows = _shape_rows(shape) if isinstance(shape, str) else None
            count_value = _counter(count)
            if (
                rows is not None
                and rows >= minimum_rows
                and count_value is not None
                and count_value > 0
            ):
                return True
    return False


def _tensor_forward_requirements(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    target_rows: int | None,
    mtp_rows: int | None,
) -> dict[str, Any]:
    delta = _tensor_forward_delta(before, after)
    required = target_rows is not None or mtp_rows is not None
    if not required:
        return {
            "required": False,
            "required_target_batch_rows": None,
            "required_mtp_batch_rows": None,
            "passed": True,
            "errors": [],
            "delta": delta,
        }
    errors = list(delta["errors"])
    phases = delta["phases"]
    if target_rows is not None and not _has_positive_batch_delta(
        phases,
        TARGET_FORWARD_PHASES,
        target_rows,
    ):
        errors.append(
            f"no positive target forward histogram delta reached B>={target_rows}"
        )
    if mtp_rows is not None and not _has_positive_batch_delta(
        phases,
        MTP_FORWARD_PHASES,
        mtp_rows,
    ):
        errors.append(f"no positive MTP forward histogram delta reached B>={mtp_rows}")
    return {
        "required": True,
        "required_target_batch_rows": target_rows,
        "required_mtp_batch_rows": mtp_rows,
        "passed": not errors,
        "errors": errors,
        "delta": delta,
    }


async def _main(args: argparse.Namespace) -> int:
    base_url = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"missing environment variable: {args.api_key_env}")
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = httpx.Timeout(args.timeout, connect=10.0)
    limits = httpx.Limits(max_connections=max(args.concurrency) + 2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        before = await _snapshot(client, base_url)
        for index in range(args.warmups):
            nonce = (
                uuid.uuid5(uuid.NAMESPACE_URL, f"{args.nonce_seed}:warmup:{index}").hex
                if args.nonce_seed
                else uuid.uuid4().hex
            )
            await _run_sample(
                client,
                base_url=base_url,
                model=args.model,
                headers=headers,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                concurrency=1,
                round_index=-1,
                request_index=index,
                prompt_nonce=nonce,
            )

        samples: list[Sample] = []
        summaries: dict[str, Any] = {}
        for concurrency in args.concurrency:
            group_started = time.perf_counter()
            group: list[Sample] = []
            for round_index in range(args.rounds):
                nonces = [
                    (
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{args.nonce_seed}:c{concurrency}:r{round_index}:i{index}",
                        ).hex
                        if args.nonce_seed
                        else uuid.uuid4().hex
                    )
                    for index in range(concurrency)
                ]
                round_samples = await asyncio.gather(
                    *(
                        _run_sample(
                            client,
                            base_url=base_url,
                            model=args.model,
                            headers=headers,
                            reasoning_effort=args.reasoning_effort,
                            max_output_tokens=args.max_output_tokens,
                            concurrency=concurrency,
                            round_index=round_index,
                            request_index=request_index,
                            prompt_nonce=nonces[request_index],
                        )
                        for request_index in range(concurrency)
                    )
                )
                group.extend(round_samples)
            wall_s = time.perf_counter() - group_started
            samples.extend(group)
            summaries[str(concurrency)] = _summarize(group, wall_s)
            print(
                json.dumps(
                    {"concurrency": concurrency, **summaries[str(concurrency)]},
                    sort_keys=True,
                ),
                flush=True,
            )
        after = await _snapshot(client, base_url)

    tensor_before = _tensor_forward_counters(before)
    tensor_after = _tensor_forward_counters(after)
    tensor_requirements = _tensor_forward_requirements(
        tensor_before,
        tensor_after,
        target_rows=args.require_target_batch_rows,
        mtp_rows=args.require_mtp_batch_rows,
    )

    result = {
        "schema_version": "mlx-batch-server.live-responses-benchmark.v1",
        "created_at_unix": time.time(),
        "config": {
            "base_url": base_url,
            "model": args.model,
            "concurrency": args.concurrency,
            "rounds": args.rounds,
            "warmups": args.warmups,
            "nonce_seed": args.nonce_seed,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "usage_source": "terminal_responses_event",
            "require_target_batch_rows": args.require_target_batch_rows,
            "require_mtp_batch_rows": args.require_mtp_batch_rows,
        },
        "runtime_before": before,
        "runtime_after": after,
        "mtp_before": _mtp_counters(before),
        "mtp_after": _mtp_counters(after),
        "tensor_forward_before": tensor_before,
        "tensor_forward_after": tensor_after,
        "tensor_forward_requirements": tensor_requirements,
        "summaries": summaries,
        "samples": [asdict(sample) for sample in samples],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return (
        0
        if all(sample.status == "completed" and sample.exact for sample in samples)
        and tensor_requirements["passed"]
        else 1
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:10240")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--nonce-seed")
    parser.add_argument("--reasoning-effort", default="off")
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--require-target-batch-rows", type=_positive_int)
    parser.add_argument("--require-mtp-batch-rows", type=_positive_int)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_parser().parse_args())))
