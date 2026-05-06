#!/usr/bin/env python3
"""Probe the Responses API input_video contract without embedding base64 in logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe /v1/responses input_video")
    parser.add_argument("video", type=Path, help="Path or URL to a .mov/.mp4/.avi file")
    parser.add_argument("--model", default="chat")
    parser.add_argument("--base-url", default="http://localhost:10240")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    video_source = str(args.video)
    if args.video.exists():
        video_source = str(args.video.resolve())

    payload = {
        "model": args.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_video", "video_url": video_source},
                    {"type": "input_text", "text": "Describe this video briefly."},
                ],
            }
        ],
        "max_output_tokens": 128,
        "stream": False,
    }

    with httpx.Client(timeout=args.timeout) as client:
        response = client.post(f"{args.base_url}/v1/responses", json=payload)

    try:
        body = response.json()
    except json.JSONDecodeError:
        print(response.text[:2000], file=sys.stderr)
        return 1

    print(json.dumps(body, indent=2, ensure_ascii=False)[:4000])
    if response.status_code >= 400 or body.get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
