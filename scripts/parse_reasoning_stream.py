#!/usr/bin/env python3
"""Parse Responses SSE JSON lines into separate Reasoning and Response blocks.

Expected input:
  curl ... | awk '/^data: /{... print json ...}' | python scripts/parse_reasoning_stream.py

This script supports both:
1. Native reasoning SSE events emitted by Harmony models
2. Models that stream everything through output_text.delta and wrap reasoning in
   <think>...</think> (or emit only the closing </think> because the prompt
   already ended with an opening <think>)

Output shape:
  RESPONSE_ID=...
  Reasoning:
  ... live reasoning deltas ...

  Response:
  ... one final response block ...
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

# Allow running directly from the repo without requiring editable install state.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional runtime enhancement
    AutoTokenizer = None


class ThinkTagStreamMux:
    """Split mixed output deltas into live reasoning and live response streams."""

    START_TAG = "<think>"
    END_TAG = "</think>"

    def __init__(self, assume_initial_thinking: bool = True) -> None:
        self.assume_initial_thinking = assume_initial_thinking
        self.mode = "thinking" if assume_initial_thinking else "response"
        self.buffer = ""
        self.saw_tags = False

    @staticmethod
    def _safe_prefix_len(buffer: str, tag: str) -> int:
        """Return length safe to emit without risking partial tag leakage."""
        max_overlap = min(len(buffer), len(tag) - 1)
        for overlap in range(max_overlap, 0, -1):
            if buffer.endswith(tag[:overlap]):
                return len(buffer) - overlap
        return len(buffer)

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []

        self.buffer += text
        out: list[tuple[str, str]] = []

        while self.buffer:
            if self.mode == "thinking":
                if self.buffer.startswith(self.START_TAG):
                    self.buffer = self.buffer[len(self.START_TAG) :]
                    self.saw_tags = True
                    continue

                end_idx = self.buffer.find(self.END_TAG)
                if end_idx != -1:
                    if end_idx > 0:
                        out.append(("thinking", self.buffer[:end_idx]))
                    self.buffer = self.buffer[end_idx + len(self.END_TAG) :]
                    self.mode = "response"
                    self.saw_tags = True
                    continue

                safe_len = self._safe_prefix_len(self.buffer, self.END_TAG)
                if safe_len > 0:
                    out.append(("thinking", self.buffer[:safe_len]))
                    self.buffer = self.buffer[safe_len:]
                break

            else:
                if self.buffer.startswith(self.END_TAG):
                    self.buffer = self.buffer[len(self.END_TAG) :]
                    self.mode = "response"
                    self.saw_tags = True
                    continue

                start_idx = self.buffer.find(self.START_TAG)
                if start_idx != -1:
                    if start_idx > 0:
                        out.append(("response", self.buffer[:start_idx]))
                    self.buffer = self.buffer[start_idx + len(self.START_TAG) :]
                    self.mode = "thinking"
                    self.saw_tags = True
                    continue

                safe_len = self._safe_prefix_len(self.buffer, self.START_TAG)
                if safe_len > 0:
                    out.append(("response", self.buffer[:safe_len]))
                    self.buffer = self.buffer[safe_len:]
                break

        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self.buffer:
            return []

        out: list[tuple[str, str]] = []
        out.append((self.mode, self.buffer))
        self.buffer = ""
        return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split Responses SSE JSON lines into Reasoning/Response blocks."
    )
    parser.add_argument(
        "--no-response-id",
        action="store_true",
        help="Do not print RESPONSE_ID in the final summary.",
    )
    parser.add_argument(
        "--response-id-at-start",
        action="store_true",
        help="Print RESPONSE_ID=... immediately on response.created as well.",
    )
    parser.add_argument(
        "--no-assume-initial-thinking",
        action="store_true",
        help=(
            "Treat output_text deltas as normal response text until an explicit "
            "<think> tag appears. By default, the parser assumes reasoning-first "
            "models that start inside <think> and only emit </think> later."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer path or HF repo used for exact token counting. "
            "If omitted, the parser falls back to TOKENIZER or MODEL env vars."
        ),
    )
    return parser


def _maybe_load_tokenizer(cli_tokenizer: str | None):
    if AutoTokenizer is None:
        return None, None

    tokenizer_id = cli_tokenizer or os.environ.get("TOKENIZER")
    model_id = os.environ.get("MODEL")
    candidate = tokenizer_id or model_id
    if not candidate:
        return None, None

    candidate_path = Path(candidate).expanduser()
    looks_like_repo = "/" in candidate and not candidate.startswith("http")

    if not tokenizer_id and not (candidate_path.exists() or looks_like_repo):
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
        return tokenizer, candidate
    except Exception:
        return None, candidate


def _count_tokens(tokenizer, text: str) -> int | None:
    if not text:
        return 0
    if tokenizer is None:
        return None
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return len(encoded)
    except Exception:
        return None


def _format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds * 1000:.1f} ms"


def _format_tps(tokens: int | None, seconds: float | None) -> str:
    if tokens is None or seconds is None or seconds <= 0:
        return "n/a"
    return f"{tokens / seconds:.2f}"


def _get_thinking_decoder():
    module = importlib.import_module("mlx_batch_server.chat.mlx.tools.thinking_decoder")
    return module.ThinkingDecoder


def main() -> int:  # noqa: PLR0912, PLR0915
    args = build_arg_parser().parse_args()
    ThinkingDecoder = _get_thinking_decoder()

    mux = ThinkTagStreamMux(assume_initial_thinking=not args.no_assume_initial_thinking)
    final_decoder = ThinkingDecoder()

    printed_reasoning_header = False
    printed_response_header = False
    saw_explicit_reasoning_stream = False
    live_reasoning_parts: list[str] = []
    full_output_parts: list[str] = []
    response_id: str | None = None

    started_at = time.perf_counter()
    first_visible_at: float | None = None
    first_reasoning_at: float | None = None
    first_response_at: float | None = None
    completed_at: float | None = None

    def emit(kind: str, text: str) -> None:
        nonlocal printed_reasoning_header
        nonlocal printed_response_header
        nonlocal first_visible_at
        nonlocal first_reasoning_at
        nonlocal first_response_at

        if not text:
            return

        now = time.perf_counter()
        if first_visible_at is None:
            first_visible_at = now
        if kind == "thinking" and first_reasoning_at is None:
            first_reasoning_at = now
        if kind == "response" and first_response_at is None:
            first_response_at = now

        if kind == "thinking":
            if not printed_reasoning_header:
                print("Reasoning:")
                printed_reasoning_header = True
            print(text, end="", flush=True)
            live_reasoning_parts.append(text)
            return

        if not printed_response_header:
            text = text.lstrip("\r\n")
            if printed_reasoning_header:
                print()
            print("Response:")
            printed_response_header = True
        print(text, end="", flush=True)

    for raw_line in sys.stdin:
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue

        try:
            event = json.loads(stripped_line)
        except json.JSONDecodeError as exc:
            print(f"\n[parser] skipping invalid json line: {exc}", file=sys.stderr)
            continue

        event_type = event.get("type")

        if event_type == "response.created" and not args.no_response_id:
            response = event.get("response") or {}
            response_id = response.get("id") or event.get("id")
            if response_id and args.response_id_at_start:
                print(f"RESPONSE_ID={response_id}")
            continue

        if event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            saw_explicit_reasoning_stream = True
            emit("thinking", event.get("delta") or "")
            continue

        if event_type in {
            "response.reasoning_summary_text.done",
            "response.reasoning_text.done",
        }:
            if printed_reasoning_header:
                print()
            continue

        if event_type == "response.output_text.delta":
            delta = event.get("delta") or ""
            if not delta:
                continue

            full_output_parts.append(delta)

            if not saw_explicit_reasoning_stream:
                for kind, piece in mux.feed(delta):
                    emit(kind, piece)
            continue

        if event_type == "response.output_text.done":
            completed_at = time.perf_counter()
            full_text = (
                "".join(full_output_parts)
                if full_output_parts
                else (event.get("text") or "")
            )

            if not saw_explicit_reasoning_stream:
                for kind, piece in mux.flush():
                    emit(kind, piece)

                parsed = final_decoder.decode(full_text)
                reasoning_text = parsed.get("thinking")
                response_text = parsed.get("content") or ""

                if reasoning_text and not printed_reasoning_header:
                    emit("thinking", reasoning_text)
                    print()
                if response_text and not printed_response_header:
                    emit("response", response_text)
            else:
                emit("response", full_text)

            print()
            continue

        if event_type == "response.completed":
            if response_id is None:
                response = event.get("response") or {}
                response_id = response.get("id") or event.get("id")
            if completed_at is None:
                completed_at = time.perf_counter()
            continue

    if completed_at is None:
        completed_at = time.perf_counter()

    full_text = "".join(full_output_parts)
    parsed = (
        final_decoder.decode(full_text)
        if full_text
        else {"thinking": None, "content": ""}
    )
    final_reasoning = "".join(live_reasoning_parts) or parsed.get("thinking") or ""
    final_response = parsed.get("content") or full_text

    tokenizer, tokenizer_source = _maybe_load_tokenizer(args.tokenizer)
    reasoning_tokens = _count_tokens(tokenizer, final_reasoning)
    response_tokens = _count_tokens(tokenizer, final_response)
    total_tokens = None
    if reasoning_tokens is not None and response_tokens is not None:
        total_tokens = reasoning_tokens + response_tokens

    if tokenizer is None:
        approx_reasoning_tokens = max(0, round(len(final_reasoning) / 4))
        approx_response_tokens = max(0, round(len(final_response) / 4))
        approx_total_tokens = approx_reasoning_tokens + approx_response_tokens
        reasoning_tokens_display = f"~{approx_reasoning_tokens} (approx)"
        response_tokens_display = f"~{approx_response_tokens} (approx)"
        total_tokens_display = f"~{approx_total_tokens} (approx)"
        response_tps_display = _format_tps(
            approx_response_tokens,
            (completed_at - first_response_at) if first_response_at else None,
        )
        total_tps_display = _format_tps(
            approx_total_tokens,
            (completed_at - first_visible_at) if first_visible_at else None,
        )
        token_source_display = (
            "approx chars/4"
            if tokenizer_source is None
            else f"fallback approx chars/4 (tokenizer unavailable: {tokenizer_source})"
        )
    else:
        reasoning_tokens_display = str(reasoning_tokens)
        response_tokens_display = str(response_tokens)
        total_tokens_display = str(total_tokens)
        response_tps_display = _format_tps(
            response_tokens,
            (completed_at - first_response_at) if first_response_at else None,
        )
        total_tps_display = _format_tps(
            total_tokens,
            (completed_at - first_visible_at) if first_visible_at else None,
        )
        token_source_display = f"tokenizer={tokenizer_source}"

    print("Metrics:")
    print(
        f"  TTFT: {_format_ms((first_visible_at - started_at) if first_visible_at else None)}"
    )
    print(
        f"  TTFR: {_format_ms((first_reasoning_at - started_at) if first_reasoning_at else None)}"
    )
    print(
        f"  TTFO: {_format_ms((first_response_at - started_at) if first_response_at else None)}"
    )
    print(f"  Reasoning tokens: {reasoning_tokens_display}")
    print(f"  Response tokens: {response_tokens_display}")
    print(f"  Total tokens: {total_tokens_display}")
    print(f"  Response tok/s: {response_tps_display}")
    print(f"  Total tok/s: {total_tps_display}")
    print(f"  Token source: {token_source_display}")
    if response_id and not args.no_response_id:
        print(f"  RESPONSE_ID: {response_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
