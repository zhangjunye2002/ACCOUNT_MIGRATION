"""
通用数据层：按块流式读 CSV，产出 EpochBatch / RawEpochStats。
与 AERO 解耦，后续可替换为从 BlockEmulator 拉同一接口。
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
