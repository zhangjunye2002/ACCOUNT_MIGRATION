"""
AERO 的 MDP 环境实现（对应论文 Section 4, Fig. 3）。

在强化学习里，「环境」要提供两个核心接口：
- reset(): 重置环境，返回初始状态 s_0；
- step(action): 接受一次动作 a_t，返回 (s_{t+1}, r_t, done, ...)。

这里我们先实现一个 **独立可运行的仿真环境**：
- 用随机/启发式方式模拟「迁移前后」 shard 负载和 CST/IST 的变化，
  方便你在没有 BlockEmulator 的情况下先把 DRL 流程跑通、调试网络结构；
- 将来真正接上 BlockEmulator 时，只需要把 `_step_transition()` 换成：
  「把 (state, action) 丢给 BlockEmulator，拿回新的统计数据」即可。
"""

import numpy as np
from typing import Any, Dict, List, Tuple
from .config import AEROConfig
from .state import AEROState
from .action import MigrationTransaction, action_to_migrations
from .reward import reward_from_state


class AEROEnv:
    """
    把「一次重配置周期 (epoch)」看成 RL 里的一个 step：

    - 状态 s_t: 链在第 t 个 epoch 结束时的整体统计（见 AEROState）；
    - 动作 a_t: 本轮要执行的迁移计划（若干个 MigrationTransaction 的序列）；
    - 奖励 R_t: 根据本轮之后的 CST 比例 / 负载方差计算出来的标量；
    - 下一个状态 s_{t+1}: 执行迁移 + 新一轮交易之后的链状态。
    """

    def __init__(self, config: AEROConfig, seed: int = 0):
        self.config = config
        self.num_shards = config.num_shards
        self.num_prefixes = config.num_prefixes
        self.max_migrations = config.max_migrations_per_epoch
        self.rng = np.random.default_rng(seed)
        # prefix_to_shard: current assignment (prefix index -> shard index)
        self._prefix_to_shard: np.ndarray = np.zeros(config.num_prefixes, dtype=np.int32)
        self._epoch = 0
        self._action_history: List[Tuple[int, int, int]] = []

    def reset(
        self,
        *,
        prefix_to_shard: np.ndarray = None,
        CST_per_shard: np.ndarray = None,
        IST_per_shard: np.ndarray = None,
        TXc: np.ndarray = None,
        TXi: np.ndarray = None,
    ) -> Tuple[AEROState, Dict[str, Any]]:
        """
        重置环境，返回一个「初始 epoch」的状态。

        - 如果你传入 prefix_to_shard / CST/IST 等，就使用你给定的初始条件；
        - 如果不传，就随机生成一组 shard 负载和前缀分布。
        """
        self._epoch = 0
        self._action_history = []
        if prefix_to_shard is not None:
            self._prefix_to_shard = np.asarray(prefix_to_shard, dtype=np.int32)
        else:
            # Random initial assignment
            self._prefix_to_shard = self.rng.integers(
                0, self.num_shards, size=self.num_prefixes, dtype=np.int32
            )
        if CST_per_shard is not None and IST_per_shard is not None:
            cst = np.asarray(CST_per_shard)
            ist = np.asarray(IST_per_shard)
        else:
            cst, ist, TXc, TXi = self._random_shard_stats()
        if TXc is None or TXi is None:
            _, _, TXc, TXi = self._random_shard_stats()
        s0 = self._build_state(cst, ist, TXc, TXi)
        info = {"epoch": 0, "prefix_to_shard": self._prefix_to_shard.copy()}
        return s0, info

    def _random_shard_stats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        随机生成一组 shard 级别的统计量。

        这里只是为了让环境「能跑起来」，不代表真实主网分布；
        真正实验时会换成由 BlockEmulator / 真实数据驱动的统计。
        """
        N, P = self.num_shards, self.num_prefixes
        # Random positive counts per shard
        cst = self.rng.integers(10, 500, size=N)
        ist = self.rng.integers(100, 2000, size=N)
        TXc = self.rng.integers(0, 50, size=(P, N)).astype(np.float32)
        TXi = self.rng.integers(0, 100, size=(P, N)).astype(np.float32)
        return cst, ist, TXc, TXi

    def _build_state(
        self,
        CST_per_shard: np.ndarray,
        IST_per_shard: np.ndarray,
        TXc_t: np.ndarray,
        TXi_t: np.ndarray,
        TXc_prev: np.ndarray = None,
        TXi_prev: np.ndarray = None,
    ) -> AEROState:
        N, P = self.num_shards, self.num_prefixes
        if TXc_prev is None:
            TXc_prev = np.zeros_like(TXc_t)
        if TXi_prev is None:
            TXi_prev = np.zeros_like(TXi_t)
        total = np.maximum(CST_per_shard + IST_per_shard, 1)
        T_t = (IST_per_shard.astype(float) / total).tolist()
        c_ratio = CST_per_shard.astype(float) / total
        C_t = c_ratio.tolist()
        V_t = (np.ones(N) * np.var(CST_per_shard)).tolist()
        return AEROState(
            num_shards=N,
            num_prefixes=P,
            T_t=T_t,
            C_t=C_t,
            V_t=V_t,
            CST_per_shard=CST_per_shard.tolist(),
            IST_per_shard=IST_per_shard.tolist(),
            TXc_t=TXc_t,
            TXi_t=TXi_t,
            TXc_prev=TXc_prev,
            TXi_prev=TXi_prev,
        )

    def step(
        self,
        state: AEROState,
        action: np.ndarray,
    ) -> Tuple[AEROState, float, bool, bool, Dict[str, Any]]:
        """
        环境一步：执行「本轮 epoch 的迁移计划」。

        参数：
        - state: 当前 epoch 结束时的状态 s_t；
        - action: shape 为 (max_migrations, 3)，每一行是 (src_shard, tgt_shard, prefix_id)，
          对应一条迁移指令（论文里的 {sender, receiver, p}）。

        注意：
        - 这里我们容忍一些无效行（比如 src==tgt），这些会被简单忽略；
        - 在接入 BlockEmulator 时，你可以在调用前自己做更严格的合法性检查。
        """
        migrations = self._parse_action(action)
        # Apply migrations to prefix_to_shard
        for m in migrations:
            if 0 <= m.prefix < self.num_prefixes and 0 <= m.sender_shard < self.num_shards and 0 <= m.receiver_shard < self.num_shards:
                if self._prefix_to_shard[m.prefix] == m.sender_shard:
                    self._prefix_to_shard[m.prefix] = m.receiver_shard
        self._action_history = [m.to_tuple() for m in migrations]
        self._epoch += 1
        # Simulate next state (replace with BlockEmulator data when integrated)
        next_cst, next_ist, next_TXc, next_TXi = self._step_transition(state, migrations)
        next_state = self._build_state(
            next_cst, next_ist, next_TXc, next_TXi,
            TXc_prev=state.TXc_t,
            TXi_prev=state.TXi_t,
        )
        reward = reward_from_state(next_state, w1=self.config.w1, w2=self.config.w2)
        # One epoch per step; run for fixed num_epochs or until done
        done = False
        truncated = False
        info = {"epoch": self._epoch, "migrations": len(migrations)}
        return next_state, reward, done, truncated, info

    def _parse_action(self, action: np.ndarray) -> List[MigrationTransaction]:
        """Convert raw action tensor to list of MigrationTransaction, discarding no-ops."""
        if action.ndim == 1:
            action = action.reshape(1, -1)
        out = []
        N, P = self.num_shards, self.num_prefixes
        for i in range(action.shape[0]):
            src = int(np.clip(np.round(action[i, 0]), 0, N - 1))
            tgt = int(np.clip(np.round(action[i, 1]), 0, N - 1))
            pref = int(np.clip(np.round(action[i, 2]), 0, P - 1))
            if src != tgt:
                out.append(MigrationTransaction(sender_shard=src, receiver_shard=tgt, prefix=pref))
        return out

    def _step_transition(
        self,
        state: AEROState,
        migrations: List[MigrationTransaction],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate next epoch stats from current state and migrations.
        Replace this with BlockEmulator feedback when integrating.
        """
        N, P = self.num_shards, self.num_prefixes
        # Heuristic: moving prefix from i to j reduces cross-shard flow for that prefix.
        # Next stats: damped toward more balanced, plus noise
        cst = np.array(state.CST_per_shard, dtype=np.float64)
        ist = np.array(state.IST_per_shard, dtype=np.float64)
        # Slight rebalance and noise
        for _ in migrations:
            # Move some load from sender to receiver
            for m in migrations:
                i, j = m.sender_shard, m.receiver_shard
                if cst[i] > 5:
                    cst[i] -= self.rng.integers(0, 5)
                    cst[j] += self.rng.integers(0, 5)
        cst = np.maximum(1, cst + self.rng.normal(0, 5, size=N))
        ist = np.maximum(10, ist + self.rng.normal(0, 20, size=N))
        TXc = state.TXc_t + self.rng.normal(0, 2, size=(P, N)).astype(np.float32)
        TXi = state.TXi_t + self.rng.normal(0, 3, size=(P, N)).astype(np.float32)
        TXc = np.maximum(0, TXc)
        TXi = np.maximum(0, TXi)
        return cst.astype(np.int32), ist.astype(np.int32), TXc, TXi

    def get_action_history_for_policy(self) -> np.ndarray:
        """Return last L actions as (L, 3) for policy input. Padding if shorter."""
        L = self.config.action_history_len
        out = np.zeros((L, 3), dtype=np.float32)
        h = self._action_history[-L:]
        for i, (s, t, p) in enumerate(h):
            out[i, 0], out[i, 1], out[i, 2] = s, t, p
        return out
