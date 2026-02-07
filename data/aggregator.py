"""
从 (from_addr, to_addr) 列表按 prefix/shard 聚合成 RawEpochStats。
不依赖 AERO，仅输出通用统计；后续可由 aero 的 adapter 转成 AEROState，或由 BlockEmulator 产出同结构。
"""

from typing import Callable, List, Tuple

from .schema import EpochBatch, RawEpochStats


def default_addr_to_prefix(addr: str, prefix_bits: int = 8) -> int:
    """
    地址 -> prefix_id。默认取 0x 后前 prefix_bits 位对应的整数值。
    例如 prefix_bits=8：addr "0x735854c5..." -> 0x73 -> 115。
    """
    s = (addr or "").strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if not s:
        return 0
    # 取前若干字符，每字符 4 bit；prefix_bits=8 -> 2 字符
    n_chars = (prefix_bits + 3) // 4
    chunk = s[:n_chars].ljust(n_chars, "0")
    return int(chunk, 16)


def aggregate_epoch(
    batch: EpochBatch,
    prefix_to_shard: List[int],
    num_shards: int,
    num_prefixes: int,
    *,
    addr_to_prefix: Callable[[str], int] | None = None,
    prefix_bits: int = 8,
) -> RawEpochStats:
    """
    将 EpochBatch 聚合成 RawEpochStats。
    - prefix_to_shard: 长度 num_prefixes，prefix_to_shard[p] = 该 prefix 所在 shard
    - addr_to_prefix: 若为 None，使用 default_addr_to_prefix(addr, prefix_bits)
    """
    if addr_to_prefix is None:
        def _ap(a: str) -> int:
            return default_addr_to_prefix(a, prefix_bits)
        addr_to_prefix = _ap

    P, N = num_prefixes, num_shards
    cst = [0] * N
    ist = [0] * N
    txc = [[0.0] * N for _ in range(P)]
    txi = [[0.0] * N for _ in range(P)]

    for (f, t) in batch.transactions:
        pf = addr_to_prefix(f) % P
        pt = addr_to_prefix(t) % P
        sf = prefix_to_shard[pf] % N
        st = prefix_to_shard[pt] % N

        if sf == st:
            ist[sf] += 1
            txi[pf][sf] += 1
            txi[pt][st] += 1
        else:
            cst[sf] += 1
            cst[st] += 1
            txc[pf][sf] += 1
            txc[pf][st] += 1
            txc[pt][sf] += 1
            txc[pt][st] += 1

    return RawEpochStats(
        block_start=batch.block_start,
        block_end=batch.block_end,
        cst_per_shard=cst,
        ist_per_shard=ist,
        txc=txc,
        txi=txi,
    )
