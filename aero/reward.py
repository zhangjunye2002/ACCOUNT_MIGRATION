"""
AERO 的奖励函数（对应论文 Section 4.3, 式 (8)–(12)）。

直观理解：
- u_t 越小越好：代表「跨片交易占比」越小，更多交易发生在同一个 shard 里；
- v_t 越大越好：这里 v_t = -Var(CST_i)，也就是「负的方差」，
  所以 variance 越小，v_t 越接近 0，表示不同 shard 的负载越均衡。

最终奖励：

    R_t = w1 * u_t + w2 * v_t

通过调节 w1 / w2，可以在「降低 CSTX」和「负载均衡」之间做权衡。
"""

from typing import List, Union
import numpy as np

from .state import AEROState


def compute_reward(
    CST_per_shard: Union[List[int], np.ndarray],
    IST_per_shard: Union[List[int], np.ndarray],
    w1: float = 1.0,
    w2: float = 1.0,
    v_scale: float = 1e-6,
    N: int = None,
) -> float:
    """
    R_t = w1 * (1 - u_t) + w2 * v_t

    - u_t = c_t / (b_t + c_t): 跨片交易占比（CSTX ratio），越低越好
    - v_t = -(1/N) * sum_i (CST_i - c_t)^2: 负方差，越接近 0 越好

    最大化 R → 降低 u_t（跨片比例） + 最小化方差 → 分片更均衡。
    注意：u_t 是 CSTX ratio，我们用 (1-u_t) 使方向一致，
    即"最大化 R ↔ 降低跨片比例"。
    """
    CST = np.asarray(CST_per_shard, dtype=np.float64)
    IST = np.asarray(IST_per_shard, dtype=np.float64)
    if N is None:
        N = len(CST)
    if N == 0:
        return 0.0

    c_t = CST.mean()
    b_t = IST.mean()
    # Avoid div by zero
    denom = b_t + c_t
    if denom <= 0:
        u_t = 0.0
    else:
        u_t = c_t / denom

    # v_t = negative variance of CST_i
    var_cst = np.mean((CST - c_t) ** 2)
    v_t = -var_cst

    # 修正奖励方向：用 (1-u_t) 使「最大化 R ↔ 降低跨片比例」
    # v_t 缩放避免量级差异过大；可按数据集调整 v_scale: 1e-6 ~ 1e-4
    return float(w1 * (1.0 - u_t) + w2 * (v_scale * v_t))


def reward_from_state(
    state: AEROState,
    w1: float = 1.0,
    w2: float = 1.0,
    v_scale: float = 1e-6,
) -> float:
    """Compute reward from AEROState using CST_per_shard and IST_per_shard."""
    return compute_reward(
        state.CST_per_shard,
        state.IST_per_shard,
        w1=w1,
        w2=w2,
        v_scale=v_scale,
        N=state.num_shards,
    )
