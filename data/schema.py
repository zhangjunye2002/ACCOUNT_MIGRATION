"""
通用数据结构：与 AERO / BlockEmulator 解耦，仅表示「按块的交易流」与「聚合统计」。
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class EpochBatch:
    """
    一个 epoch 内的原始交易批次。
    由 CSV 流式加载或由 BlockEmulator 提供，上层只依赖本类型。
    """
    block_start: int
    block_end: int
    transactions: List[Tuple[str, str]]  # [(from_address, to_address), ...]


@dataclass
class RawEpochStats:
    """
    按 shard/prefix 聚合后的 epoch 统计（无 AERO 类型依赖）。
    - 可由 data.aggregator 从 EpochBatch 计算得到；
    - 也可由 BlockEmulator 直接产出同结构数据，再交给 aero 的 adapter 转成 AEROState。
    """
    block_start: int
    block_end: int
    cst_per_shard: List[int]   # 各 shard 的跨片交易数，长度 N
    ist_per_shard: List[int]   # 各 shard 的片内交易数，长度 N
    txc: List[List[float]]   # TX^c: (num_prefixes, num_shards)，CSTX 按 prefix×shard
    txi: List[List[float]]   # TX^i: (num_prefixes, num_shards)，片内交易按 prefix×shard
