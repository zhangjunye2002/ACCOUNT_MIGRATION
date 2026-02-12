"""
AERO 中 MDP 的「状态」表示（对应论文 Section 4.1）。

在强化学习里，我们需要把「当前链的运行情况」编码成一个向量 s_t，
喂给神经网络做决策。论文中给出的状态是：

    s_t = {T_t, C_t, V_t, TX^c_t, TX^i_t, TX^c_{t-1}, TX^i_{t-1}}

本文件就是把这些符号变成实际的数据结构：
- T_t: 每个 shard 的吞吐情况；
- C_t: 每个 shard 的跨 shard 交易比例；
- V_t: 用于反映负载方差的指标；
- TX^c / TX^i: prefix × shard 维度的交易量矩阵（跨片 / 片内）；
- 上一轮(epoch t-1) 的 TX^c / TX^i：让策略能看到时间上的变化趋势。
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ShardState:
    """
    单个 shard 在某个 epoch 的简单统计。

    实际训练时主要用 AEROState；ShardState 更多是用来帮助理解/调试。
    """

    T: float  # throughput of this physical shard
    C: float  # CSTX ratio of this shard (or raw CST count for reward)
    V: float  # variance-related metric for this shard
    # For reward we need CST_i and IST_i per shard; can be held here or in AEROState
    CST: int = 0  # number of cross-shard transactions in this shard
    IST: int = 0  # number of intra-shard transactions in this shard


@dataclass
class AEROState:
    """
    强化学习中的「环境状态」 s_t，对应某一 epoch 结束时链的整体情况。

    - T_t: list of throughput per physical shard (length N)
    - C_t: list of CSTX ratio per shard (or we use global); paper uses "overall CSTX ratio"
    - V_t: list of variance per shard
    - TX^c_t: CSTX volumes per (prefix, shard) or per shard — paper says "CSTX volumes for each
      account prefix p within each physical shard", so shape (num_prefixes, num_shards) or
      aggregated per shard (num_shards,) for simpler version
    - TX^i_t: intra-shard transaction volumes (per shard or per prefix per shard)
    - Same for t-1: TX^c_{t-1}, TX^i_{t-1}

    For reward we need CST_i and IST_i per shard; we store them in shard_stats.
    """

    num_shards: int
    num_prefixes: int

    # Length-N lists (N = num_shards)
    T_t: List[float]  # throughput per shard
    C_t: List[float]  # CSTX ratio per shard (or scalar for global)
    V_t: List[float]  # variance per shard

    # CST_i, IST_i per shard (for reward: u_t, v_t)
    CST_per_shard: List[int]  # length N
    IST_per_shard: List[int]  # length N

    # TX^c_t: (num_prefixes, num_shards) — CSTX count for prefix p in shard i
    TXc_t: np.ndarray  # shape (num_prefixes, num_shards)
    # TX^i_t: intra-shard volume per prefix per shard or per shard only
    TXi_t: np.ndarray  # shape (num_prefixes, num_shards) or (num_shards,)

    # Previous epoch
    TXc_prev: np.ndarray
    TXi_prev: np.ndarray

    def to_vector(self) -> np.ndarray:
        """
        把结构化的状态展开成一条向量，供神经网络输入。

        对应论文里的 d_s 维状态向量：
        - 前 3*N 维：每个 shard 的 T / C / V；
        - 后面 4 * P * N 维：当前和上一轮的 TX^c / TX^i 四个矩阵按行展平。
        """
        N = self.num_shards
        P = self.num_prefixes
        # T_t, C_t, V_t: 3*N
        part1 = np.concatenate([
            np.asarray(self.T_t, dtype=np.float32),
            np.asarray(self.C_t, dtype=np.float32),
            np.asarray(self.V_t, dtype=np.float32),
        ])
        # TXc_t, TXi_t: P*N + P*N
        part2 = self.TXc_t.ravel().astype(np.float32)
        part3 = self.TXi_t.ravel().astype(np.float32)
        # prev
        part4 = self.TXc_prev.ravel().astype(np.float32)
        part5 = self.TXi_prev.ravel().astype(np.float32)
        return np.concatenate([part1, part2, part3, part4, part5])

    @property
    def state_dim(self) -> int:
        N = self.num_shards
        P = self.num_prefixes
        return 3 * N + 4 * P * N

    @classmethod
    def dummy(cls, num_shards: int = 16, num_prefixes: int = 256) -> "AEROState":
        """Build a dummy state for dimension checks."""
        return cls(
            num_shards=num_shards,
            num_prefixes=num_prefixes,
            T_t=[0.0] * num_shards,
            C_t=[0.0] * num_shards,
            V_t=[0.0] * num_shards,
            CST_per_shard=[0] * num_shards,
            IST_per_shard=[0] * num_shards,
            TXc_t=np.zeros((num_prefixes, num_shards), dtype=np.float32),
            TXi_t=np.zeros((num_prefixes, num_shards), dtype=np.float32),
            TXc_prev=np.zeros((num_prefixes, num_shards), dtype=np.float32),
            TXi_prev=np.zeros((num_prefixes, num_shards), dtype=np.float32),
        )

    @classmethod
    def from_vector(
        cls,
        vec: np.ndarray,
        CST_per_shard: np.ndarray,
        IST_per_shard: np.ndarray,
        num_shards: int,
        num_prefixes: int,
    ) -> "AEROState":
        """Reconstruct AEROState from state vector and CST/IST (e.g. for checkpoint resume)."""
        N, P = num_shards, num_prefixes
        ofs = 0
        part1 = vec[ofs : ofs + 3 * N]
        ofs += 3 * N
        part2 = vec[ofs : ofs + P * N].reshape(P, N)
        ofs += P * N
        part3 = vec[ofs : ofs + P * N].reshape(P, N)
        ofs += P * N
        part4 = vec[ofs : ofs + P * N].reshape(P, N)
        ofs += P * N
        part5 = vec[ofs : ofs + P * N].reshape(P, N)
        return cls(
            num_shards=N,
            num_prefixes=P,
            T_t=part1[:N].tolist(),
            C_t=part1[N : 2 * N].tolist(),
            V_t=part1[2 * N : 3 * N].tolist(),
            CST_per_shard=np.asarray(CST_per_shard).tolist(),
            IST_per_shard=np.asarray(IST_per_shard).tolist(),
            TXc_t=part2.astype(np.float32),
            TXi_t=part3.astype(np.float32),
            TXc_prev=part4.astype(np.float32),
            TXi_prev=part5.astype(np.float32),
        )
