from typing import Callable, List, Optional

from .schema import EpochBatch, RawEpochStats


def default_addr_to_prefix(addr: str, prefix_bits: int = 8) -> int:
    """
    地址 -> prefix_id。
    默认 prefix_bits=8，即取 0x 后前 2 个 hex 字符。
    """
    s = (addr or "").strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if not s:
        return 0
    n_chars = (prefix_bits + 3) // 4
    chunk = s[:n_chars].ljust(n_chars, "0")
    return int(chunk, 16)


def aggregate_epoch(
    batch: EpochBatch,
    prefix_to_shard: List[int],
    num_shards: int,
    num_prefixes: int,
    *,
    addr_to_prefix: Optional[Callable[[str], int]] = None,
    prefix_bits: int = 8,
) -> RawEpochStats:
    """
    将一个 epoch 的 (from,to) 交易列表聚合为：
    - 每个 shard 的 CST/IST 数量
    - 每个 prefix×shard 的 txc/txi 统计
    """
    if addr_to_prefix is None:
        def addr_to_prefix(a: str) -> int:  # type: ignore[redefinition]
            return default_addr_to_prefix(a, prefix_bits)

    N, P = num_shards, num_prefixes
    cst = [0] * N
    ist = [0] * N
    txc = [[0.0] * N for _ in range(P)]
    txi = [[0.0] * N for _ in range(P)]

    for f, t in batch.transactions:
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

