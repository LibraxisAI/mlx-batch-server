"""Batch Processing Module - Parallel inference for concurrent requests.

This module provides transparent batching for concurrent requests using mlx-lm's
BatchGenerator. Multiple requests are automatically collected and processed
together for improved throughput.

Key components:
- BatchChatGenerator: Wrapper around mlx-lm BatchGenerator
- BatchRequestCoordinator: Orchestrates request collection and dispatch
- get_batch_coordinator: Factory function for singleton coordinators

Target metrics:
- 10+ concurrent streaming requests
- 500+ tok/s total throughput for 70B model
- <500ms time-to-first-token per user
- <150MB overhead per concurrent request

Vibecrafted. with AI Agents by VetCoders (c)2024-2026 The LibraxisAI Team
"""

from . import router
from .coordinator import (
    BatchRequestCoordinator,
    get_batch_coordinator,
    get_loaded_batch_models,
    shutdown_all_coordinators,
    shutdown_batch_coordinator,
)
from .generator import (
    BatchChatGenerator,
    BatchGenerationStats,
    BatchRequest,
    BatchStreamChunk,
)

__all__ = [
    "BatchChatGenerator",
    "BatchGenerationStats",
    "BatchRequest",
    "BatchRequestCoordinator",
    "BatchStreamChunk",
    "get_batch_coordinator",
    "get_loaded_batch_models",
    "router",
    "shutdown_all_coordinators",
    "shutdown_batch_coordinator",
]
