#!/usr/bin/env python3
"""Probe native vision plus a forced function call on a live Responses endpoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

TERMINAL_EVENTS = {
    "response.completed",
    "response.failed",
    "response.incomplete",
    "response.cancelled",
}


def _image_data_url(path: Path) -> tuple[str, str]:
    payload = path.read_bytes()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{media_type};base64,{encoded}", hashlib.sha256(payload).hexdigest()


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
    image_url, image_sha256 = _image_data_url(args.image)
    payload = {
        "model": args.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Inspect the top-left product header in this screenshot. "
                            "Call record_ui_snapshot exactly once with the product name "
                            "and the adjacent model alias. Do not emit prose."
                        ),
                    },
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                ],
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "record_ui_snapshot",
                "description": "Record the product and model alias visible in the UI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "string"},
                        "model_alias": {"type": "string"},
                    },
                    "required": ["product", "model_alias"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
        "tool_choice": {"type": "function", "name": "record_ui_snapshot"},
        "reasoning": {"effort": "off"},
        "max_output_tokens": args.max_output_tokens,
        "stream": True,
        "store": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    event_types: Counter[str] = Counter()
    output_text: list[str] = []
    terminal: dict[str, Any] | None = None
    started = time.perf_counter()
    timeout = httpx.Timeout(args.timeout, connect=10.0)
    async with (
        httpx.AsyncClient(timeout=timeout) as client,
        client.stream(
            "POST",
            f"{args.base_url.rstrip('/')}/v1/responses",
            headers=headers,
            json=payload,
        ) as response,
    ):
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            event = json.loads(data)
            event_type = str(event.get("type") or "")
            event_types[event_type] += 1
            if event_type == "response.output_text.delta" and isinstance(
                event.get("delta"), str
            ):
                output_text.append(event["delta"])
            if event_type in TERMINAL_EVENTS:
                terminal = event

    if terminal is None or not isinstance(terminal.get("response"), dict):
        raise RuntimeError("response stream ended without a terminal response")
    terminal_response: dict[str, Any] = terminal["response"]
    if terminal_response.get("status") != "completed":
        raise RuntimeError(
            "response did not complete: "
            + json.dumps(terminal_response.get("error"), sort_keys=True)
        )
    calls = _function_calls(terminal_response)
    call = calls[0] if len(calls) == 1 else {}
    try:
        arguments = json.loads(str(call.get("arguments") or "{}"))
    except json.JSONDecodeError:
        arguments = {}
    assertions = {
        "one_function_call": len(calls) == 1,
        "correct_function": call.get("name") == "record_ui_snapshot",
        "stable_call_id": isinstance(call.get("call_id"), str),
        "product_is_mtplx": str(arguments.get("product", "")).upper() == "MTPLX",
        "alias_is_buddy": str(arguments.get("model_alias", "")).lower() == "buddy",
        "no_prose": not "".join(output_text).strip(),
    }
    result = {
        "schema_version": "mlx-batch-server.vision-tool-probe.v1",
        "base_url": args.base_url.rstrip("/"),
        "model": args.model,
        "image": {
            "path": str(args.image),
            "sha256": image_sha256,
            "bytes": args.image.stat().st_size,
        },
        "latency_s": time.perf_counter() - started,
        "event_types": dict(sorted(event_types.items())),
        "response_id": terminal_response.get("id"),
        "response_metadata": terminal_response.get("metadata"),
        "function_call": call,
        "arguments": arguments,
        "output_text": "".join(output_text),
        "assertions": assertions,
        "passed": all(assertions.values()),
    }
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
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_parser().parse_args())))
