# MLX Batch Server - Development Makefile
# Extended by LibraxisAI
#
# Usage:
#   make install    - Install dependencies
#   make dev        - Run development server
#   make test       - Run tests
#   make lint       - Run linters
#   make format     - Format code
#   make check      - Run all checks (lint + test)
#   make clean      - Clean build artifacts

.PHONY: install dev run stop restart logs test lint format check clean help benchmark \
	benchmark-cli benchmark-quick benchmark-build setup install-dev install-hooks lint-fix format-check \
	security pre-commit pre-push test-fast test-cov test-responses loctree twins build \
	docker-build docker-run load unload list ps status batch-stats embeddings reranker \
	vision stt tts
.DEFAULT_GOAL := help

# === Configuration ===
PYTHON := uv run python
PORT ?= 10240
HOST ?= 0.0.0.0
LOG_LEVEL ?= info
CORS ?= http://localhost:*

# === Installation ===
install: ## Install as global CLI tool (mlx-batch-server command)
	uv tool install . --force --prerelease=allow
	@echo "✓ Installed: mlx-batch-server --help"

setup: ## Full dev setup (deps + hooks)
	uv sync --all-groups
	@$(MAKE) install-hooks
	@echo "✓ Dev environment ready"

install-dev: ## Install dev dependencies only
	uv sync --group dev

install-hooks: ## Install pre-commit hooks
	@if [ -d .git ]; then \
		uv run pre-commit install && \
		uv run pre-commit install --hook-type pre-push && \
		echo "✓ Git hooks installed"; \
	else \
		echo "⚠ Not a git repo - skipping hooks"; \
	fi

# === Development ===
dev: ## Run development server (foreground, default port: 10240)
	$(PYTHON) -m mlx_batch_server.main --port $(PORT) --host $(HOST) --log-level $(LOG_LEVEL) --cors-allow-origins="$(CORS)"

dev-10240: ## Run on port 10240 (LibraxisAI integration)
	$(PYTHON) -m mlx_batch_server.main --port 10240 --host 0.0.0.0 --log-level info --cors-allow-origins="$(CORS)"

LOG_FILE ?= mlx-batch-server.log
PID_FILE ?= .mlx-batch-server.pid

run: ## Run server as background daemon (logs to $(LOG_FILE))
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Server already running (PID: $$(cat $(PID_FILE)))"; \
	else \
		nohup $(PYTHON) -m mlx_batch_server.main --port $(PORT) --host $(HOST) --log-level $(LOG_LEVEL) --cors-allow-origins="$(CORS)" > $(LOG_FILE) 2>&1 & \
		echo $$! > $(PID_FILE); \
		sleep 1; \
		if kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
			echo "Server started (PID: $$(cat $(PID_FILE)), log: $(LOG_FILE))"; \
		else \
			echo "Failed to start server - check $(LOG_FILE)"; \
			rm -f $(PID_FILE); \
		fi \
	fi

stop: ## Stop background server
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			kill $$PID && echo "Server stopped (PID: $$PID)"; \
		else \
			echo "Server not running (stale PID file)"; \
		fi; \
		rm -f $(PID_FILE); \
	else \
		echo "No PID file found"; \
	fi

restart: stop run ## Restart background server

logs: ## Tail server logs
	@if [ -f $(LOG_FILE) ]; then tail -f $(LOG_FILE); else echo "No log file found"; fi

# === Playground / Benchmarking ===
BENCH_PORT ?= 7860
BENCH_WORKERS ?= 10
BENCH_ENDPOINT ?= http://localhost:$(PORT)/v1/responses
BENCH_MODEL ?= chat

benchmark: benchmark-build ## Run API tester (Gradio UI on http://localhost:7860)
	$(PYTHON) playground/api_tester.py --port $(BENCH_PORT)

benchmark-build: ## Build HTML tester static assets
	$(PYTHON) playground/build_static.py

benchmark-cli: ## Run CLI benchmark (BENCH_WORKERS=10 BENCH_MODEL=chat)
	$(PYTHON) playground/api_tester.py --cli -w $(BENCH_WORKERS) -e $(BENCH_ENDPOINT) -m $(BENCH_MODEL)

benchmark-quick: ## Quick CLI benchmark (1 worker, 1 prompt)
	$(PYTHON) playground/api_tester.py --cli -w 1 -c 1 -e $(BENCH_ENDPOINT) -m $(BENCH_MODEL)

# === Testing ===
test: ## Run all tests
	uv run pytest tests/ -v

test-fast: ## Run fast tests only (skip slow)
	uv run pytest tests/ -v -m "not slow"

test-cov: ## Run tests with coverage
	uv run pytest tests/ -v --cov=src/mlx_batch_server --cov-report=term-missing

test-responses: ## Run responses API tests
	uv run pytest tests/test_responses.py tests/test_circuit_breaker.py -v

# === Linting & Formatting ===
lint: ## Run all linters
	uv run ruff check src/ tests/
	uv run mypy src/

lint-fix: ## Run linters and fix issues
	uv run ruff check --fix src/ tests/

format: ## Format code with ruff
	uv run ruff format src/ tests/

format-check: ## Check formatting without changes
	uv run ruff format --check src/ tests/

security: ## Run security checks (bandit + semgrep)
	uv run bandit -c pyproject.toml -r src/
	@echo "Run 'semgrep --config auto src/' for full security scan"

# === Quality Gates ===
check: lint format-check test-fast ## Run all checks (CI simulation)
	@echo "All checks passed!"

pre-commit: ## Run pre-commit on all files
	uv run pre-commit run --all-files

pre-push: ## Run pre-push hooks
	uv run pre-commit run --hook-stage pre-push --all-files

# === Code Analysis ===
loctree: ## Run loctree analysis
	@if command -v loct &>/dev/null; then \
		loct auto; \
		cat .loctree/agent.json | python -c "import sys,json; d=json.load(sys.stdin); print(f'Health: {d[\"summary\"][\"health_score\"]}/100')"; \
	else \
		echo "loctree not installed"; \
	fi

twins: ## Check for duplicate code
	@if command -v loct &>/dev/null; then loct twins; else echo "loctree not installed"; fi

# === Model Management (LMS-style) ===
MODEL ?= mlx-community/Qwen2.5-7B-Instruct-4bit
SERVER_URL ?= http://localhost:$(PORT)
TASK ?=
UNLOAD_MODEL := $(if $(filter-out file,$(origin MODEL)),$(MODEL),)
EMBEDDINGS_INPUT ?= hello from mlx
RERANKER_INPUT ?= query: hello | passage: hello
VISION_PROMPT ?= a simple test image
VISION_RESPONSE_FORMAT ?= url
STT_AUDIO ?= tests/test_audio.wav
STT_RESPONSE_FORMAT ?= json
TTS_INPUT ?= hello from mlx
TTS_VOICE ?= af_sky
TTS_FORMAT ?= wav
TTS_OUTPUT ?= logs/tts-output.wav

load: ## Load a model (MODEL=<model-id> [TASK=llm|embeddings|visual|images|stt|tts])
	@echo "Loading model: $(MODEL)"
	@payload='{"model": "$(MODEL)"}'; \
	if [ -n "$(TASK)" ]; then payload=$$(printf '{"model": "%s", "task": "%s"}' "$(MODEL)" "$(TASK)"); fi; \
	curl -s -X POST $(SERVER_URL)/v1/models/load \
		-H "Content-Type: application/json" \
		-d "$$payload" | $(PYTHON) -m json.tool 2>/dev/null || \
		echo "Server not running or endpoint not available"

unload: ## Unload model(s) (MODEL=<model-id> [TASK=llm|embeddings|visual|images|stt|tts])
	@echo "Unloading model..."
	@payload='{}'; \
	if [ -n "$(UNLOAD_MODEL)" ]; then payload=$$(printf '{"model": "%s"}' "$(UNLOAD_MODEL)"); fi; \
	if [ -n "$(TASK)" ]; then \
		if [ -n "$(UNLOAD_MODEL)" ]; then \
			payload=$$(printf '{"model": "%s", "task": "%s"}' "$(UNLOAD_MODEL)" "$(TASK)"); \
		else \
			payload=$$(printf '{"task": "%s"}' "$(TASK)"); \
		fi; \
	fi; \
	curl -s -X POST $(SERVER_URL)/v1/models/unload \
		-H "Content-Type: application/json" \
		-d "$$payload" | $(PYTHON) -m json.tool 2>/dev/null || \
		echo "Server not running or endpoint not available"

ps: ## List loaded models (in memory)
	@curl -s $(SERVER_URL)/v1/models/loaded 2>/dev/null | \
		$(PYTHON) -c "import sys,json; d=json.load(sys.stdin); models=d.get('data',[]); \
		print('\n'.join(f'  \033[32m●\033[0m {m[\"id\"]}' for m in models)) if models else print('  (none loaded)')" 2>/dev/null || \
		echo "Server not running"

list: ## List all models in HuggingFace cache (local disk)
	@echo "=== HuggingFace Cache (local models) ==="
	@LOADED=$$(curl -s $(SERVER_URL)/v1/models/loaded 2>/dev/null | $(PYTHON) -c "import sys,json; d=json.load(sys.stdin); print(' '.join(m['id'] for m in d.get('data',[])))" 2>/dev/null || echo ""); \
	hf cache ls 2>/dev/null | grep "^model/" | \
		$(PYTHON) -c "import sys; loaded='$$LOADED'.split(); \
[print(f'  \033[32m● loaded\033[0m  {p[6:]:50} {s:>8}') if p[6:] in loaded else print(f'  \033[90m○ cached\033[0m  {p[6:]:50} {s:>8}') \
for line in sys.stdin if (parts := line.strip().split()) and len(parts) >= 2 and (p := parts[0]) and (s := parts[1])]" 2>/dev/null || \
		echo "  (hf CLI not available - install: cargo install hf-cli)"

status: ## Server status with loaded models
	@echo "=== MLX Batch Server Status ==="
	@curl -s $(SERVER_URL)/health 2>/dev/null | \
		$(PYTHON) -c "import sys,json; h=json.load(sys.stdin); \
		print(f'Server: \033[32mUP\033[0m ({h[\"loaded_models_count\"]} models loaded)')" 2>/dev/null || \
		echo "Server: \033[31mDOWN\033[0m"
	@echo ""
	@echo "Available models (disk cache):"
	@LOADED=$$(curl -s $(SERVER_URL)/v1/models/loaded 2>/dev/null | $(PYTHON) -c "import sys,json; d=json.load(sys.stdin); print(' '.join(m['id'] for m in d.get('data',[])))" 2>/dev/null); \
	curl -s $(SERVER_URL)/v1/models 2>/dev/null | \
		$(PYTHON) -c "import sys,json; loaded='$$LOADED'.split(); d=json.load(sys.stdin); \
		[print(f'  \033[32m● loaded\033[0m  {m[\"id\"]}') if m['id'] in loaded else print(f'  \033[90m○ cached\033[0m  {m[\"id\"]}') for m in d.get('data',[])]" 2>/dev/null || \
		echo "  (server not running)"

batch-stats: ## Show batch coordinator stats
	@curl -s $(SERVER_URL)/v1/batch/stats | $(PYTHON) -m json.tool 2>/dev/null || \
		echo "Server not running or endpoint not available"

# === Task Model Helpers ===
embeddings: ## Run embeddings model (MODEL=<model-id> PORT=<port>)
	@echo "Embedding request: $(MODEL)"
	@curl -s -X POST $(SERVER_URL)/v1/embeddings \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL)", "input": "$(EMBEDDINGS_INPUT)"}' | \
		$(PYTHON) -c "import sys,json; d=json.load(sys.stdin); dim=len(d.get('data',[{}])[0].get('embedding', [])); print('OK embeddings model: %s (dim=%s)' % (d.get('model'), dim))" 2>/dev/null || \
		echo "Server not running or endpoint not available"

reranker: ## Run reranker model (MODEL=<model-id> PORT=<port>)
	@echo "Reranker request: $(MODEL)"
	@curl -s -X POST $(SERVER_URL)/v1/embeddings \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL)", "input": "$(RERANKER_INPUT)"}' | \
		$(PYTHON) -c "import sys,json; d=json.load(sys.stdin); dim=len(d.get('data',[{}])[0].get('embedding', [])); print('OK reranker model: %s (dim=%s)' % (d.get('model'), dim))" 2>/dev/null || \
		echo "Server not running or endpoint not available"

vision: ## Run vision model (MODEL=<model-id> PORT=<port>)
	@echo "Vision request: $(MODEL)"
	@curl -s -X POST $(SERVER_URL)/v1/images/generations \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL)", "prompt": "$(VISION_PROMPT)", "response_format": "$(VISION_RESPONSE_FORMAT)"}' | \
		$(PYTHON) -c "import sys,json; d=json.load(sys.stdin); data=d.get('data', []); url=data[0].get('url') if data else None; print('OK vision response: %s' % (url or 'no url'))" 2>/dev/null || \
		echo "Server not running or endpoint not available"

stt: ## Run stt model (MODEL=<model-id> PORT=<port>)
	@echo "STT request: $(MODEL) (audio: $(STT_AUDIO))"
	@curl -s -X POST $(SERVER_URL)/v1/audio/transcriptions \
		-H "Content-Type: multipart/form-data" \
		-F "file=@$(STT_AUDIO)" \
		-F "model=$(MODEL)" \
		-F "response_format=$(STT_RESPONSE_FORMAT)" || \
		echo "Server not running or endpoint not available"

tts: ## Run tts model (MODEL=<model-id> PORT=<port>)
	@mkdir -p $(dir $(TTS_OUTPUT))
	@echo "TTS request: $(MODEL) -> $(TTS_OUTPUT)"
	@curl -s -X POST $(SERVER_URL)/v1/audio/speech \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL)", "input": "$(TTS_INPUT)", "voice": "$(TTS_VOICE)", "response_format": "$(TTS_FORMAT)"}' \
		--output $(TTS_OUTPUT) && \
		echo "Saved audio to $(TTS_OUTPUT)" || \
		echo "Server not running or endpoint not available"

# === Build & Release ===
build: ## Build package
	uv build

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# === Docker (optional) ===
docker-build: ## Build Docker image
	docker build -t mlx-batch-server:latest .

docker-run: ## Run in Docker
	docker run -p $(PORT):$(PORT) mlx-batch-server:latest

# === Help ===
help: ## Show this help
	@echo "MLX Batch Server - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Created by M&K (c)2026 The LibraxisAI Team
