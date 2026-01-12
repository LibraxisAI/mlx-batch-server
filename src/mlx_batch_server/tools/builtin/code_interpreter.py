"""
Code Interpreter Tool - Execute Python code in a restricted environment.

Provides basic code execution as a hosted tool for the Responses API.
Uses restricted builtins for safety.

WARNING: This is a simplified implementation. For production use,
consider using a proper sandbox like Docker or firecracker.

Created by M&K (c)2026 The LibraxisAI Team
"""

from __future__ import annotations

import io
import logging
import math
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

logger = logging.getLogger(__name__)

# Allowed builtins for code execution (safe subset)
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    # Math functions
    "math": math,
}

# Maximum execution time (seconds)
MAX_EXECUTION_TIME = 5

# Maximum output length (characters)
MAX_OUTPUT_LENGTH = 10000


async def execute_code(
    code: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Execute Python code in a restricted environment.

    Args:
        code: Python code to execute
        **kwargs: Additional parameters (ignored)

    Returns:
        Execution result dict with 'output', 'result', or 'error'
    """
    if not code or not code.strip():
        return {"error": "No code provided"}

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Create restricted globals
    restricted_globals = {
        "__builtins__": SAFE_BUILTINS,
        "__name__": "__main__",
    }

    result = None
    error = None

    try:
        # Redirect output
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Try to evaluate as expression first
            try:
                result = eval(code, restricted_globals)  # nosec B307
            except SyntaxError:
                # Fall back to exec for statements
                exec(code, restricted_globals)  # nosec B102
                result = restricted_globals.get("result")

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.warning(f"Code execution error: {error}")

    # Get output
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()

    # Truncate if needed
    if len(stdout_output) > MAX_OUTPUT_LENGTH:
        stdout_output = stdout_output[:MAX_OUTPUT_LENGTH] + "\n... (truncated)"

    response: dict[str, Any] = {}

    if stdout_output:
        response["output"] = stdout_output

    if stderr_output:
        response["stderr"] = stderr_output

    if result is not None:
        # Convert result to string representation
        try:
            response["result"] = repr(result)
        except Exception:
            response["result"] = str(result)

    if error:
        response["error"] = error

    if not response:
        response["output"] = "(no output)"

    logger.info(
        f"Code execution completed: {len(code)} chars, error={error is not None}"
    )

    return response
