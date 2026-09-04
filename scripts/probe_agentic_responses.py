#!/usr/bin/env python3
"""Run a deterministic client-tool loop against a live Responses endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

QUERY = "flash-main-port"
SOURCE = "https://fixture.local/articles/flash-42"
EXPECTED = f"PORT=8100\nSOURCE={SOURCE}"
TERMINAL_EVENTS = {
    "response.completed",
    "response.failed",
    "response.incomplete",
    "response.cancelled",
}


async def _response(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    event_types: Counter[str] = Counter()
    text_parts: list[str] = []
    terminal: dict[str, Any] | None = None
    response_headers: dict[str, str] = {}
    first_visible_s: float | None = None

    async with client.stream(
        "POST",
        f"{base_url}/v1/responses",
        headers=headers,
        json=payload,
    ) as response:
        response.raise_for_status()
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower().startswith("x-lbrx-")
        }
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            event = json.loads(data)
            event_type = str(event.get("type") or "")
            event_types[event_type] += 1
            delta = event.get("delta")
            if (
                first_visible_s is None
                and isinstance(delta, str)
                and delta
                and event_type.endswith(".delta")
            ):
                first_visible_s = time.perf_counter() - started
            if event_type == "response.output_text.delta" and isinstance(delta, str):
                text_parts.append(delta)
            if event_type in TERMINAL_EVENTS:
                terminal = event

    if terminal is None:
        raise RuntimeError("response stream ended without a terminal event")
    response = terminal.get("response")
    if not isinstance(response, dict):
        raise RuntimeError("terminal event does not contain a response")
    if response.get("status") != "completed":
        raise RuntimeError(
            "response did not complete: "
            + json.dumps(response.get("error"), sort_keys=True)
        )
    return {
        "response": response,
        "output_text": "".join(text_parts).strip(),
        "event_types": dict(sorted(event_types.items())),
        "headers": response_headers,
        "first_visible_s": first_visible_s,
        "latency_s": time.perf_counter() - started,
    }


def _function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        return []
    return [
        item
        for item in output
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


async def _main(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing environment variable: {args.api_key_env}")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    tool = {
        "type": "function",
        "name": "search_fixture",
        "description": "Return the authoritative local fixture for one exact query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    initial_payload = {
        "model": args.model,
        "input": (
            f"Call search_fixture with query {QUERY}. Do not answer from memory. "
            "After the client returns the result, answer with exactly two lines: "
            "PORT=<port> and SOURCE=<source>. Never call the tool twice. "
            "Question: which port serves the main Flash model?"
        ),
        "tools": [tool],
        "tool_choice": {"type": "function", "name": "search_fixture"},
        "reasoning": {"effort": "off"},
        "max_output_tokens": args.max_output_tokens,
        "stream": True,
        "store": True,
    }
    timeout = httpx.Timeout(args.timeout, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        initial = await _response(
            client,
            base_url=args.base_url.rstrip("/"),
            headers=headers,
            payload=initial_payload,
        )
        calls = _function_calls(initial["response"])
        if len(calls) != 1:
            raise RuntimeError(f"expected one function call, received {len(calls)}")
        call = calls[0]
        arguments = json.loads(str(call.get("arguments") or "{}"))
        if call.get("name") != "search_fixture" or arguments != {"query": QUERY}:
            raise RuntimeError(f"unexpected tool call: {call}")
        if initial["output_text"]:
            raise RuntimeError("model emitted an answer before the client tool receipt")

        receipt = {
            "query": QUERY,
            "port": 8100,
            "source": SOURCE,
            "title": "Flash main runtime fixture",
        }
        continuation_payload = {
            "model": args.model,
            "previous_response_id": initial["response"]["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(receipt, sort_keys=True),
                }
            ],
            "tools": [],
            "tool_choice": "none",
            "reasoning": {"effort": "off"},
            "max_output_tokens": args.max_output_tokens,
            "stream": True,
            "store": True,
        }
        continuation = await _response(
            client,
            base_url=args.base_url.rstrip("/"),
            headers=headers,
            payload=continuation_payload,
        )

    duplicate_calls = _function_calls(continuation["response"])
    result = {
        "schema_version": "mlx-batch-server.agentic-probe.v1",
        "base_url": args.base_url.rstrip("/"),
        "model": args.model,
        "rounds": 2,
        "initial": initial,
        "tool_call": call,
        "client_receipt": receipt,
        "continuation": continuation,
        "assertions": {
            "one_initial_call": len(calls) == 1,
            "stable_call_id": isinstance(call.get("call_id"), str),
            "no_answer_before_receipt": not initial["output_text"],
            "no_duplicate_call": not duplicate_calls,
            "exact_final_answer": continuation["output_text"] == EXPECTED,
            "source_only_from_receipt": SOURCE in continuation["output_text"],
        },
    }
    result["passed"] = all(result["assertions"].values())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0 if result["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8089")
    parser.add_argument("--model", default="buddy")
    parser.add_argument("--api-key-env", default="LBRX_API_KEY_MACIEJ")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_parser().parse_args())))
