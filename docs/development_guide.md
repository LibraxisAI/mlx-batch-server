# Development Guide

This guide is intended for developers who want to contribute to MLX Batch Server or create their own extensions.

## Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/madroidmaq/mlx-batch-server.git
cd mlx-batch-server
```

2. Install dependencies using uv:
```bash
uv pip install -e .
```

## Running the Server in Development Mode

There are two ways to run the server during development:

### 1. Using uvicorn (Recommended for development)

```bash
uvicorn mlx_batch_server.main:app --reload --host 0.0.0.0 --port 8100
```

The `--reload` flag enables hot-reload, which automatically restarts the server when code changes are detected. This is particularly useful during development.

### 2. Using the standard entry point

```bash
mlx-batch-server
```


## Contributing Guidelines

1. Fork the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Follow the code style:
   - Use [Black](https://black.readthedocs.io/) for Python code formatting
   - Use [isort](https://pycqa.github.io/isort/) for import sorting
   - Run pre-commit hooks before committing:
     ```bash
     pre-commit install
     pre-commit run --all-files
     ```
4. Write clear commit messages
5. Push to your branch:
   ```bash
   git push origin feature/amazing-feature
   ```
6. Open a Pull Request with:
   - Clear description of the changes
   - Any relevant issue numbers
   - Screenshots for UI changes (if applicable)

## Testing

Run the test suite:
```bash
pytest
```

## Building Documentation

The documentation is written in Markdown and stored in the `docs/` directory.

## Getting Help

- Open an issue for bugs or feature requests
- Join discussions in the GitHub Discussions section
- Check existing issues and pull requests before creating new ones
