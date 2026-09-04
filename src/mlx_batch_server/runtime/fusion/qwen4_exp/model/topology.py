# SPDX-License-Identifier: Apache-2.0
"""Execution topology derived without importing the Qwen4Exp tensor stack."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .config import Qwen4ExpCheckpointConfig, Qwen4ExpConfigError


class Qwen4ExpTensorBatchMode(StrEnum):
    ROW_SERIAL = "row_serial"


@dataclass(frozen=True, slots=True)
class Qwen4ExpTopology:
    layer_types: tuple[str, ...]
    full_attention_layers: tuple[int, ...]
    linear_attention_layers: tuple[int, ...]
    ple_layers: tuple[int, ...]
    mtp_layers: int
    has_vision: bool
    tensor_batch_mode: Qwen4ExpTensorBatchMode
    max_qsa_batch_rows: int
    max_verified_mtp_rows: int

    def __post_init__(self) -> None:
        all_layers = self.full_attention_layers + self.linear_attention_layers
        if tuple(sorted(all_layers)) != tuple(range(len(self.layer_types))):
            raise Qwen4ExpConfigError("topology does not cover every text layer")
        if any(
            self.layer_types[index] != "full_attention"
            for index in self.full_attention_layers
        ):
            raise Qwen4ExpConfigError("full-attention topology is inconsistent")
        if any(
            self.layer_types[index] != "linear_attention"
            for index in self.linear_attention_layers
        ):
            raise Qwen4ExpConfigError("linear-attention topology is inconsistent")
        if self.max_qsa_batch_rows != 1:
            raise Qwen4ExpConfigError("frozen MTPLX QSA supports exactly one row")
        if self.max_verified_mtp_rows != 1:
            raise Qwen4ExpConfigError("multi-row MTP remains unproven")


def build_qwen4_exp_topology(
    config: Qwen4ExpCheckpointConfig,
) -> Qwen4ExpTopology:
    """Build the first honest execution envelope for the frozen donor trunk.

    Scheduler admission and late joining may still cover several requests.
    ``ROW_SERIAL`` means the initial tensor implementation must execute each
    scheduled row independently on the one owner thread because MTPLX QSA
    rejects batch sizes other than one.
    """

    layers = config.text.layer_types
    full = tuple(index for index, kind in enumerate(layers) if kind == "full_attention")
    linear = tuple(
        index for index, kind in enumerate(layers) if kind == "linear_attention"
    )
    expected_full = tuple(
        index
        for index in range(config.text.num_hidden_layers)
        if (index + 1) % config.text.full_attention_interval == 0
    )
    if full != expected_full:
        raise Qwen4ExpConfigError(
            "full-attention layers do not match full_attention_interval"
        )
    return Qwen4ExpTopology(
        layer_types=layers,
        full_attention_layers=full,
        linear_attention_layers=linear,
        ple_layers=tuple(item - 1 for item in config.text.ple_layer_ids),
        mtp_layers=config.text.mtp.num_hidden_layers,
        has_vision=True,
        tensor_batch_mode=Qwen4ExpTensorBatchMode.ROW_SERIAL,
        max_qsa_batch_rows=1,
        max_verified_mtp_rows=1,
    )


__all__ = [
    "Qwen4ExpTensorBatchMode",
    "Qwen4ExpTopology",
    "build_qwen4_exp_topology",
]
