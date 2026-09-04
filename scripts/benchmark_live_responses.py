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
        },
        "runtime_before": before,
        "runtime_after": after,
        "mtp_before": _mtp_counters(before),
        "mtp_after": _mtp_counters(after),
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
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_parser().parse_args())))
