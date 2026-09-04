from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mlx_batch_server.runtime.fusion.qwen4_exp.model.tensor import NGramTable


def test_resident_ple_fuses_packed_shards_exactly() -> None:
    mx.random.seed(43)
    table = NGramTable(rows=32, dim=64, shard_count=4)
    table.shards = [
        nn.QuantizedEmbedding.from_embedding(
            shard,
            group_size=32,
            bits=4,
            mode="affine",
        )
        for shard in table.shards
    ]
    indices = mx.array([[0, 9, 17, 31, 9]], dtype=mx.int32)
    expected = table(indices)
    mx.eval(expected)

    assert table.fuse_quantized_shards() is True
    assert table.fuse_quantized_shards() is False
    assert table.shards == []
    actual = table(indices)
    mx.eval(actual)

    assert mx.array_equal(actual, expected).item()


def test_resident_ple_fusion_rejects_unquantized_shards() -> None:
    table = NGramTable(rows=8, dim=32, shard_count=2)

    assert table.fuse_quantized_shards() is False
