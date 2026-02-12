from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EpochBatch:
    """一个 epoch 的原始交易批次。"""

    block_start: int
    block_end: int
    transactions: List[Tuple[str, str]]  # (from_addr, to_addr)


@dataclass
class RawEpochStats:
    """按 shard/prefix 聚合后的统计（与 AERO 解耦）。"""

    block_start: int
    block_end: int
    cst_per_shard: List[int]
    ist_per_shard: List[int]
    txc: List[List[float]]  # shape: (num_prefixes, num_shards)
    txi: List[List[float]]  # shape: (num_prefixes, num_shards)

