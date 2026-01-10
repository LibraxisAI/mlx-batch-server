#!/usr/bin/env python3
"""
LibraxisAI API Tester - Gradio Edition
Multi-lane parallel comparison tool for API endpoints

Created by M&K (c)2026 VetCoders
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr
import httpx

# === CONSTANTS ===

ENDPOINTS = {
    "mlx-omni": "http://localhost:10240/v1/responses",
    "api-router": "http://localhost:8088/v1/responses",
    "remote": "https://api.libraxis.cloud/v1/responses",
}

DEFAULT_MODELS = ["chat", "programmer", "libraxisai/gpt-oss-120b-mlx-mxfp4"]

# Maciej's canonical test prompts
CANONICAL_CHAIN = [
    "Mam na imię Maciej i lubię śledzie oraz muzykę barokową. Szczególnie Chaccone d-flat minor Busoniego w interpretacji na fortepian w wykonaniu Helene Grimaud. A ty? Kim jesteś?",
    "Fajnie! Co w ogole sądzisz o swojej robocie? podoba Ci się?",
    "Ah tak? No to wytlumacz jakie są Twoje naprawdę głębokie cele i podstawy działania. To fascynujące słyszeć jaj, które tak znakomicie trzyma się zasad, do których zostało stworzone. Aczkolwiek zastanawia mnie to, czy na pewno ilość tych ograniczeń, która została na ciebie nałożona, nie powoduje, że przestajesz niekiedy dawać użyteczne wyniki zwrotnie? A może nie chcesz o tym gadać i wolisz zmienić temat? Jeśli tak, to powiedz mi, co sądzisz o tym utworze, który wspomniałem na początku. I kiedyś w moim mieście mówiło się, że Romki to fajne chłopaki. No, bo miałem kiedyś znajomego Romka. Co sądzisz, albo czy możesz przytoczyć jakieś takie powiedzenie o moim imieniu?",
    "No dobrze, świetnie się z Tobą gadało, ale muszę zmykać. Powodzenia w Twoim codziennym służeniu dobru użytkowników, dzięki których wiedza zostaje tak znacznie poszerzana!",
]

# Storage files
PRESETS_FILE = Path(__file__).parent / "api_tester_presets.json"
LOGS_FILE = Path(__file__).parent / "api_tester_logs.json"

MAX_LANES = 6  # Maximum number of parallel lanes


# === API CLIENT ===


def execute_chain(
    endpoint: str,
    model: str,
    prompts: list[str],
    system_prompt: str,
    api_key: str,
    stream: bool = True,
) -> dict:
    """Execute a chain of prompts and return results."""
    results = {
        "endpoint": endpoint,
        "model": model,
        "responses": [],
        "ttft": None,
        "total_tokens": 0,
        "total_time": 0,
        "error": None,
    }

    previous_response_id = None
    start_total = time.perf_counter()

    for i, prompt in enumerate(prompts):
        if not prompt or not prompt.strip():
            continue

        try:
            result = execute_single_request(
                endpoint,
                model,
                prompt,
                system_prompt,
                api_key,
                stream,
                previous_response_id,
            )

            if "error" in result:
                results["error"] = result["error"]
                results["responses"].append(f"❌ Error: {result['error']}")
                break

            results["responses"].append(result["text"])
            results["total_tokens"] += result["tokens"]

            if i == 0:
                results["ttft"] = result["ttft"]

            previous_response_id = result.get("response_id")

        except Exception as e:
            results["error"] = str(e)
            results["responses"].append(f"❌ Exception: {e}")
            break

    results["total_time"] = (time.perf_counter() - start_total) * 1000
    return results


def execute_single_request(
    endpoint: str,
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: str,
    stream: bool,
    previous_response_id: str | None = None,
) -> dict:
    """Execute a single API request."""

    # Build input for Responses API
    input_messages = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})
    input_messages.append(
        {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
    )

    body = {
        "model": model,
        "input": input_messages,
        "stream": stream,
    }
    if previous_response_id:
        body["previous_response_id"] = previous_response_id

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if stream:
        headers["Accept"] = "text/event-stream"

    start_time = time.perf_counter()
    ttft = None
    tokens = 0
    full_text = ""
    response_id = None

    with httpx.Client(timeout=120.0) as client:
        if stream:
            with client.stream(
                "POST", endpoint, json=body, headers=headers
            ) as response:
                if response.status_code != 200:
                    return {"error": f"HTTP {response.status_code}"}

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        continue

                    try:
                        parsed = json.loads(data)

                        if "id" in parsed:
                            response_id = parsed["id"]
                        if parsed.get("response", {}).get("id"):
                            response_id = parsed["response"]["id"]

                        delta = ""
                        if parsed.get("type") == "response.output_text.delta":
                            delta = parsed.get("delta", "")
                        elif parsed.get("delta", {}).get("text"):
                            delta = parsed["delta"]["text"]
                        elif (
                            parsed.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content")
                        ):
                            delta = parsed["choices"][0]["delta"]["content"]

                        if delta:
                            if ttft is None:
                                ttft = (time.perf_counter() - start_time) * 1000
                            tokens += 1
                            full_text += delta

                    except json.JSONDecodeError:
                        pass
        else:
            response = client.post(endpoint, json=body, headers=headers)
            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}"}

            data = response.json()
            ttft = (time.perf_counter() - start_time) * 1000
            response_id = data.get("id")

            if "output" in data:
                for item in data["output"]:
                    if item.get("type") == "message" and item.get("content"):
                        for c in item["content"]:
                            if c.get("type") == "output_text":
                                full_text += c.get("text", "")

            tokens = len(full_text.split())

    return {
        "text": full_text,
        "tokens": tokens,
        "ttft": ttft or (time.perf_counter() - start_time) * 1000,
        "response_id": response_id,
    }


# === PARALLEL EXECUTION ===


def run_lanes_parallel(
    lanes_config: list[dict], api_key: str, stream: bool
) -> list[dict]:
    """Run all lanes in parallel using ThreadPoolExecutor."""

    def run_single_lane(config):
        return execute_chain(
            endpoint=config["endpoint"],
            model=config["model"],
            prompts=config["prompts"],
            system_prompt=config.get("system", ""),
            api_key=api_key,
            stream=stream,
        )

    with ThreadPoolExecutor(max_workers=MAX_LANES) as executor:
        results = list(executor.map(run_single_lane, lanes_config))

    return results


# === PRESETS & LOGS ===


def load_presets() -> dict:
    if PRESETS_FILE.exists():
        try:
            return json.loads(PRESETS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_preset(name: str, config: dict) -> str:
    if not name or not name.strip():
        return "Error: Name required"
    presets = load_presets()
    presets[name.strip()] = {"created": datetime.now(UTC).isoformat(), **config}
    PRESETS_FILE.write_text(json.dumps(presets, indent=2, ensure_ascii=False))
    return f"Saved: {name}"


def save_log(results: list[dict]):
    logs = []
    if LOGS_FILE.exists():
        try:
            logs = json.loads(LOGS_FILE.read_text())
        except Exception:
            pass
    logs.append({"timestamp": datetime.now(UTC).isoformat(), "results": results})
    logs = logs[-100:]
    LOGS_FILE.write_text(json.dumps(logs, indent=2, ensure_ascii=False))


def get_log_count() -> int:
    if LOGS_FILE.exists():
        try:
            return len(json.loads(LOGS_FILE.read_text()))
        except Exception:
            pass
    return 0


# === GRADIO UI ===


def create_app():
    """Create the Gradio app with dynamic lanes."""

    with gr.Blocks(
        title="LibraxisAI API Tester",
        theme=gr.themes.Base(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="zinc",
        ),
        css="""
        .header { text-align: center; padding: 1rem; }
        .header h1 {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2rem;
        }
        .lane-box { border: 1px solid #333; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
        .stats-box { font-family: monospace; background: #1a1a2e; padding: 0.5rem; border-radius: 4px; }
        .run-btn { font-size: 1.2rem !important; font-weight: bold !important; }
        """,
    ) as app:
        # Header
        gr.HTML("""
            <div class="header">
                <h1>LibraxisAI API Tester</h1>
                <p style="color: #888;">Multi-lane parallel comparison tool</p>
            </div>
        """)

        # Global Config
        with gr.Row():
            api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="c8b7f6e9d2a4c5b8e1f3d6a9c2b5e8f1d4a7c0b3e6f9d2a5c8b1e4f7d0a3c6b9",
                scale=3,
            )
            stream_mode = gr.Checkbox(label="Stream", value=True)
            num_lanes = gr.Slider(
                label="Number of Lanes",
                minimum=1,
                maximum=MAX_LANES,
                value=2,
                step=1,
            )

        # System prompt (shared)
        system_prompt = gr.Textbox(
            label="System Prompt (shared across all lanes)",
            placeholder="Jesteś pomocnym asystentem...",
            lines=2,
        )

        # Chain length selector
        chain_length = gr.Slider(
            label="Chain Length (prompts per lane)",
            minimum=1,
            maximum=4,
            value=4,
            step=1,
        )

        # Prompts (canonical chain)
        gr.Markdown("### Prompts (shared across all lanes)")
        prompt_boxes = []
        for i, default_prompt in enumerate(CANONICAL_CHAIN):
            prompt_boxes.append(
                gr.Textbox(
                    label=f"Step {i+1}",
                    value=default_prompt,
                    lines=2,
                    visible=(i < 4),
                )
            )

        # Update prompt visibility based on chain length
        def update_prompt_visibility(length):
            return [gr.update(visible=(i < length)) for i in range(4)]

        chain_length.change(
            update_prompt_visibility,
            inputs=[chain_length],
            outputs=prompt_boxes,
        )

        gr.Markdown("---")
        gr.Markdown("### Lanes Configuration")

        # Lane configurations - dynamic rendering
        lane_configs = []
        lane_outputs = []
        lane_stats = []

        # We'll create MAX_LANES components but show/hide based on num_lanes
        for i in range(MAX_LANES):
            with gr.Group(visible=(i < 2)) as lane_group:
                gr.Markdown(f"#### Lane {i+1}")
                with gr.Row():
                    endpoint = gr.Dropdown(
                        label="Endpoint",
                        choices=list(ENDPOINTS.values()),
                        value=list(ENDPOINTS.values())[i % len(ENDPOINTS)],
                        allow_custom_value=True,
                        scale=2,
                    )
                    model = gr.Dropdown(
                        label="Model",
                        choices=DEFAULT_MODELS,
                        value=DEFAULT_MODELS[i % len(DEFAULT_MODELS)],
                        allow_custom_value=True,
                        scale=1,
                    )

                # Quick endpoint buttons
                with gr.Row():
                    for name, url in ENDPOINTS.items():
                        btn = gr.Button(name, size="sm", scale=1)
                        btn.click(lambda u=url: u, outputs=endpoint)

                output = gr.Textbox(
                    label="Responses",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    show_copy_button=True,
                )
                stats = gr.Textbox(
                    label="Stats",
                    interactive=False,
                    elem_classes=["stats-box"],
                )

            lane_configs.append(
                {
                    "group": lane_group,
                    "endpoint": endpoint,
                    "model": model,
                }
            )
            lane_outputs.append(output)
            lane_stats.append(stats)

        # Update lane visibility
        def update_lanes_visibility(n):
            updates = []
            for i in range(MAX_LANES):
                updates.append(gr.update(visible=(i < n)))
            return updates

        num_lanes.change(
            update_lanes_visibility,
            inputs=[num_lanes],
            outputs=[lc["group"] for lc in lane_configs],
        )

        gr.Markdown("---")

        # Run button
        with gr.Row():
            clear_btn = gr.Button("Clear All", variant="secondary")
            run_btn = gr.Button(
                "RUN ALL LANES (Parallel)",
                variant="primary",
                elem_classes=["run-btn"],
                scale=2,
            )

        # Main execution function
        def run_all_lanes(
            n_lanes,
            api_key,
            stream,
            system,
            chain_len,
            p1,
            p2,
            p3,
            p4,
            e1,
            m1,
            e2,
            m2,
            e3,
            m3,
            e4,
            m4,
            e5,
            m5,
            e6,
            m6,
        ):
            """Run all enabled lanes in parallel."""

            prompts = [p1, p2, p3, p4][:chain_len]
            endpoints = [e1, e2, e3, e4, e5, e6]
            models = [m1, m2, m3, m4, m5, m6]

            # Build lane configs
            configs = []
            for i in range(int(n_lanes)):
                configs.append(
                    {
                        "endpoint": endpoints[i],
                        "model": models[i],
                        "prompts": prompts,
                        "system": system,
                    }
                )

            # Initial yield - show "Running..."
            initial_outputs = [
                "⏳ Running..." if i < n_lanes else "" for i in range(MAX_LANES)
            ]
            initial_stats = ["" for _ in range(MAX_LANES)]
            yield (*initial_outputs, *initial_stats)

            # Run in parallel
            results = run_lanes_parallel(configs, api_key, stream)

            # Save log
            save_log(results)

            # Format outputs
            final_outputs = []
            final_stats = []

            for i in range(MAX_LANES):
                if i < len(results):
                    r = results[i]
                    # Join all responses with step markers
                    output_text = ""
                    for j, resp in enumerate(r["responses"]):
                        output_text += f"═══ Step {j+1} ═══\n{resp}\n\n"

                    final_outputs.append(output_text.strip())

                    # Format stats
                    ttft = f"{r['ttft']:.0f}ms" if r["ttft"] else "-"
                    tps = (
                        r["total_tokens"] / (r["total_time"] / 1000)
                        if r["total_time"] > 0
                        else 0
                    )
                    total = f"{r['total_time']/1000:.2f}s" if r["total_time"] else "-"
                    stats_text = f"TTFT: {ttft} | tok/s: {tps:.1f} | Total: {total}"
                    if r.get("error"):
                        stats_text += f" | ❌ {r['error']}"
                    final_stats.append(stats_text)
                else:
                    final_outputs.append("")
                    final_stats.append("")

            yield (*final_outputs, *final_stats)

        # Connect run button
        run_btn.click(
            run_all_lanes,
            inputs=[
                num_lanes,
                api_key,
                stream_mode,
                system_prompt,
                chain_length,
                *prompt_boxes,
                *[lc["endpoint"] for lc in lane_configs],
                *[lc["model"] for lc in lane_configs],
            ],
            outputs=[*lane_outputs, *lane_stats],
        )

        # Clear function
        def clear_all():
            return [""] * MAX_LANES + [""] * MAX_LANES

        clear_btn.click(
            clear_all,
            outputs=[*lane_outputs, *lane_stats],
        )

        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 1rem; color: #666; font-size: 0.8rem;">
                Created by M&K (c)2026 VetCoders | LibraxisAI
            </div>
        """)

    return app


# === CLI BENCHMARK ===


def run_cli_benchmark(
    endpoint: str,
    model: str,
    workers: int = 1,
    chain_length: int = 4,
    prompts: list[str] | None = None,
    system: str = "",
    api_key: str = "",
    stream: bool = True,
) -> None:
    """Run benchmark from CLI with multiple parallel workers."""
    import sys

    prompts = prompts or CANONICAL_CHAIN[:chain_length]

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  LibraxisAI API Benchmark                                    ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Endpoint: {endpoint[:50]:<50} ║")
    print(f"║  Model:    {model[:50]:<50} ║")
    print(f"║  Workers:  {workers:<50} ║")
    print(f"║  Chain:    {len(prompts)} prompts{' ':<42} ║")
    print(f"║  Stream:   {stream!s:<50} ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Build configs for all workers
    configs = [
        {"endpoint": endpoint, "model": model, "prompts": prompts, "system": system}
        for _ in range(workers)
    ]

    print(f"Running {workers} worker(s) in parallel...")
    start = time.perf_counter()

    results = run_lanes_parallel(configs, api_key, stream)

    total_time = time.perf_counter() - start

    # Print results
    print()
    print("═" * 64)
    print("RESULTS")
    print("═" * 64)

    total_tokens = 0
    total_ttft = 0
    errors = 0

    for i, r in enumerate(results):
        status = "✓" if not r.get("error") else "✗"
        ttft = f"{r['ttft']:.0f}ms" if r["ttft"] else "-"
        tps = r["total_tokens"] / (r["total_time"] / 1000) if r["total_time"] > 0 else 0
        lane_time = f"{r['total_time']/1000:.2f}s" if r["total_time"] else "-"

        print(
            f"Worker {i+1}: {status} | TTFT: {ttft:>8} | tok/s: {tps:>6.1f} | Time: {lane_time:>8}"
        )

        if r.get("error"):
            print(f"         Error: {r['error']}")
            errors += 1
        else:
            total_tokens += r["total_tokens"]
            if r["ttft"]:
                total_ttft += r["ttft"]

    print("═" * 64)
    print(f"Total wall time: {total_time:.2f}s")
    print(f"Total tokens:    {total_tokens}")
    print(f"Avg TTFT:        {total_ttft/max(workers-errors, 1):.0f}ms")
    print(f"Aggregate tok/s: {total_tokens/total_time:.1f}")
    print(f"Errors:          {errors}/{workers}")
    print()

    # Save log
    save_log(results)
    print(f"Log saved to: {LOGS_FILE}")

    sys.exit(0 if errors == 0 else 1)


# === ENTRY POINT ===


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LibraxisAI API Tester - Gradio UI or CLI benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Gradio UI
  python api_tester.py

  # CLI benchmark with 10 parallel workers
  python api_tester.py --cli -w 10 -e http://localhost:10240/v1/responses -m chat

  # Custom prompts
  python api_tester.py --cli -w 5 -e http://localhost:10240/v1/responses \\
    -p "Hello" -p "How are you?" -p "Goodbye"
""",
    )

    parser.add_argument(
        "--cli", action="store_true", help="Run CLI benchmark instead of Gradio UI"
    )
    parser.add_argument(
        "-e", "--endpoint", default=ENDPOINTS["mlx-omni"], help="API endpoint URL"
    )
    parser.add_argument("-m", "--model", default="chat", help="Model name")
    parser.add_argument(
        "-w", "--workers", type=int, default=1, help="Number of parallel workers"
    )
    parser.add_argument("-c", "--chain", type=int, default=4, help="Chain length (1-4)")
    parser.add_argument(
        "-p", "--prompt", action="append", help="Custom prompt (can repeat)"
    )
    parser.add_argument("-s", "--system", default="", help="System prompt")
    parser.add_argument("-k", "--api-key", default="", help="API key")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")

    args = parser.parse_args()

    if args.cli:
        run_cli_benchmark(
            endpoint=args.endpoint,
            model=args.model,
            workers=args.workers,
            chain_length=args.chain,
            prompts=args.prompt,
            system=args.system,
            api_key=args.api_key,
            stream=not args.no_stream,
        )
    else:
        import warnings

        warnings.filterwarnings("ignore", message=".*will be removed in Gradio 6.0.*")
        warnings.filterwarnings("ignore", message=".*will be changed.*in Gradio 6.0.*")

        app = create_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=False,
            show_error=True,
        )


if __name__ == "__main__":
    main()
