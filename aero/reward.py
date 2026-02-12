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
    R_t = w1 * u_t + w2 * v_t

    - u_t = c_t / (b_t + c_t)
      c_t = (1/N) * sum_i CST_i
      b_t = (1/N) * sum_i IST_i
    - v_t = - (1/N) * sum_i (CST_i - c_t)^2  [negative variance of CST per shard]

    Maximizing R encourages: lower CSTX ratio (smaller u_t) and more balanced CST (smaller variance).
    So we want u_t small and variance small -> u_t is "CSTX ratio" so lower is better.
    Paper says "By maximizing R_t, the agent is encouraged to reduce u_t and minimize v_t".
    Here v_t is already negative variance, so maximizing v_t means minimizing variance.
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

    # 对 v_t 做缩放，避免其绝对值远大于 u_t，导致学习信号被完全淹没
    return float(w1 * u_t + w2 * (v_scale * v_t))


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
