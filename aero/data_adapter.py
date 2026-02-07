"""
薄适配层：把「通用聚合统计」转换成 AEROState。

设计目标：
- data/ 层只关心区块与交易（EpochBatch / RawEpochStats），完全不知道 DRL 和 AERO；
- aero/ 层只关心自己的 AEROState；
- 两者之间只在这里发生「格式转换」，方便以后用 BlockEmulator 替换数据来源。
"""

import numpy as np
from typing import TYPE_CHECKING, Optional

from .state import AEROState

if TYPE_CHECKING:
    from data.schema import RawEpochStats


def _ensure_shape(arr: np.ndarray, P: int, N: int) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.shape[0] < P or out.shape[1] < N:
        pad = np.zeros((P, N), dtype=np.float32)
        pad[: out.shape[0], : out.shape[1]] = out
        return pad
    return out[:P, :N].copy()


def raw_stats_to_aero_state(
    current: "RawEpochStats",
    num_shards: int,
    num_prefixes: int,
    prev: Optional["RawEpochStats"] = None,
) -> AEROState:
    """
    将 data.aggregator 产出的 RawEpochStats 转为 AEROState。
    - current: 当前 epoch 的 RawEpochStats（来自 stream_epochs + aggregate_epoch 或 BlockEmulator）
    - prev: 上一 epoch 的 RawEpochStats，若为 None 则 TX^c_{t-1}, TX^i_{t-1} 置零
    """
    N, P = num_shards, num_prefixes
    cst = np.asarray(current.cst_per_shard, dtype=np.float64)[:N]
    ist = np.asarray(current.ist_per_shard, dtype=np.float64)[:N]
    total = np.maximum(cst + ist, 1.0)
    T_t = (ist / total).tolist()
    C_t = (cst / total).tolist()
    var_cst = float(np.var(cst)) if cst.size else 0.0
    V_t = [var_cst] * N

    txc = _ensure_shape(np.asarray(current.txc), P, N)
    txi = _ensure_shape(np.asarray(current.txi), P, N)
    txp_c = _ensure_shape(np.asarray(prev.txc), P, N) if prev is not None else np.zeros((P, N), dtype=np.float32)
    txp_i = _ensure_shape(np.asarray(prev.txi), P, N) if prev is not None else np.zeros((P, N), dtype=np.float32)

    return AEROState(
        num_shards=N,
        num_prefixes=P,
        T_t=T_t,
        C_t=C_t,
        V_t=V_t,
        CST_per_shard=[int(x) for x in cst],
        IST_per_shard=[int(x) for x in ist],
        TXc_t=txc,
        TXi_t=txi,
        TXc_prev=txp_c,
        TXi_prev=txp_i,
    )
