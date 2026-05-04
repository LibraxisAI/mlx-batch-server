"""Command line interface for the standalone operator backend."""

from __future__ import annotations

from pathlib import Path

import click
import httpx
import uvicorn

from mlx_batch_server.operator.config import get_settings


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Run and inspect the MLX Batch Server operator."""


@main.command()
@click.option("--host", default=None, help="Host to bind, default from settings.")
@click.option(
    "--port", type=int, default=None, help="Port to bind, default from settings."
)
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
)
def serve(host: str | None, port: int | None, log_level: str) -> None:
    """Serve the operator FastAPI app."""
    settings = get_settings()
    uvicorn.run(
        "mlx_batch_server.operator.main:app",
        host=host or settings.host,
        port=port or settings.port,
        log_level=log_level,
    )


@main.command()
@click.option("--base-url", default=None, help="Operator base URL.")
def status(base_url: str | None) -> None:
    """Probe /api/lifecycle/status."""
    settings = get_settings()
    url = (base_url or f"http://{settings.host}:{settings.port}").rstrip("/")
    response = httpx.get(f"{url}/api/lifecycle/status", timeout=5)
    response.raise_for_status()
    click.echo(response.text)


@main.command()
@click.option("--base-url", default=None, help="Operator base URL.")
@click.option("--service", default=None, help="Log service name; defaults to server.")
@click.option("--lines", default=200, type=int, help="Number of lines to return.")
def logs(base_url: str | None, service: str | None, lines: int) -> None:
    """Tail operator logs via the API, falling back to the local log path."""
    settings = get_settings()
    url = (base_url or f"http://{settings.host}:{settings.port}").rstrip("/")
    try:
        response = httpx.get(
            f"{url}/api/logs/tail",
            params={"service": service, "lines": lines},
            timeout=5,
        )
        response.raise_for_status()
        for line in response.json().get("lines", []):
            click.echo(line)
        return
    except httpx.HTTPError:
        path = (
            settings.log_path
            if service in {None, "", "server"}
            else Path(settings.log_path.parent / f"{service}.log")
        )
        if not path.exists():
            raise click.ClickException(f"Log file not found: {path}") from None
        click.echo(
            "\n".join(
                path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
            )
        )


if __name__ == "__main__":
    main()
