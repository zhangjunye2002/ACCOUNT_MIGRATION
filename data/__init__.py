"""
通用数据层：
- 流式读取 BlockTransaction CSV
- 聚合为 shard/prefix 统计

该层不依赖 AERO，可被其他后端（如 BlockEmulator）替换。
"""

from .schema import EpochBatch, RawEpochStats
from .csv_stream import stream_epochs
from .aggregator import aggregate_epoch, default_addr_to_prefix

__all__ = [
    "EpochBatch",
    "RawEpochStats",
    "stream_epochs",
    "aggregate_epoch",
    "default_addr_to_prefix",
]

