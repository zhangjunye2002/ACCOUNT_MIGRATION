"""
用 CSV 交易数据驱动的 AERO 环境：每个 step = 一个 epoch 的 CSV 数据。

- reset(): 从 CSV 流取第一个 epoch，用当前 prefix_to_shard 聚合成 AEROState 返回。
- step(state, action): 先按 action 更新 prefix_to_shard（执行迁移），再取下一个 epoch，
  用更新后的 prefix_to_shard 聚合成 next_state，reward = reward_from_state(next_state)。
CSV 读完后 truncated=True，训练脚本会 reset 并从 CSV 头重新开始（多轮跑满 total_steps）。
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from .config import AEROConfig
from .state import AEROState
from .action import MigrationTransaction
from .data_adapter import raw_stats_to_aero_state
from .reward import reward_from_state


def _parse_action(action: np.ndarray, num_shards: int, num_prefixes: int) -> List[MigrationTransaction]:
    """与 AEROEnv 一致的 action 解析。"""
    if action.ndim == 1:
        action = action.reshape(1, -1)
    out = []
    for i in range(action.shape[0]):
        src = int(np.clip(np.round(action[i, 0]), 0, num_shards - 1))
        tgt = int(np.clip(np.round(action[i, 1]), 0, num_shards - 1))
        pref = int(np.clip(np.round(action[i, 2]), 0, num_prefixes - 1))
        if src != tgt:
            out.append(MigrationTransaction(sender_shard=src, receiver_shard=tgt, prefix=pref))
    return out


class AEROEnvCSV:
    """
    由 CSV 流驱动的环境：状态与奖励均来自真实交易数据的聚合。
    与 AEROEnv 接口一致（reset / step），便于 train.py 直接替换。
    """

    def __init__(
        self,
        config: AEROConfig,
        csv_path: Union[str, Path],
        *,
        blocks_per_epoch: int = 100,
        seed: int = 0,
    ):
        self.config = config
        self.num_shards = config.num_shards
        self.num_prefixes = config.num_prefixes
        self.max_migrations = config.max_migrations_per_epoch
        self.csv_path = Path(csv_path)
        self.blocks_per_epoch = blocks_per_epoch
        self.rng = np.random.default_rng(seed)
        self._prefix_to_shard: np.ndarray = np.zeros(config.num_prefixes, dtype=np.int32)
        self._epoch = 0
        self._action_history: List[Tuple[int, int, int]] = []
        self._iterator = None
        self._prev_stats = None

    def _next_batch(self):
        """从 CSV 流取下一个 epoch batch；若耗尽则返回 None。"""
        if self._iterator is None:
            return None
        try:
            return next(self._iterator)
        except StopIteration:
            return None

    def reset(
        self,
        *,
        prefix_to_shard: np.ndarray = None,
    ) -> Tuple[AEROState, Dict[str, Any]]:
        """从 CSV 头开始，取第一个 epoch 聚合成状态并返回。"""
        from data import stream_epochs, aggregate_epoch

        self._iterator = iter(stream_epochs(self.csv_path, blocks_per_epoch=self.blocks_per_epoch))
        self._epoch = 0
        self._action_history = []

        if prefix_to_shard is not None:
            self._prefix_to_shard = np.asarray(prefix_to_shard, dtype=np.int32)
        else:
            self._prefix_to_shard = self.rng.integers(
                0, self.num_shards, size=self.num_prefixes, dtype=np.int32
            )

        batch = self._next_batch()
        if batch is None:
            raise RuntimeError(f"CSV 无数据: {self.csv_path}")

        stats = aggregate_epoch(
            batch,
            prefix_to_shard=self._prefix_to_shard.tolist(),
            num_shards=self.num_shards,
            num_prefixes=self.num_prefixes,
            prefix_bits=self.config.prefix_bits,
        )
        self._prev_stats = stats
        state = raw_stats_to_aero_state(stats, num_shards=self.num_shards, num_prefixes=self.num_prefixes, prev=None)
        info = {"epoch": self._epoch, "prefix_to_shard": self._prefix_to_shard.copy(), "block_end": batch.block_end}
        return state, info

    def step(
        self,
        state: AEROState,
        action: np.ndarray,
    ) -> Tuple[AEROState, float, bool, bool, Dict[str, Any]]:
        """
        执行迁移（更新 prefix_to_shard），再取下一个 epoch 聚合成 next_state，奖励用 reward_from_state(next_state)。
        """
        from data import aggregate_epoch

        migrations = _parse_action(action, self.num_shards, self.num_prefixes)
        for m in migrations:
            if 0 <= m.prefix < self.num_prefixes and 0 <= m.sender_shard < self.num_shards and 0 <= m.receiver_shard < self.num_shards:
                if self._prefix_to_shard[m.prefix] == m.sender_shard:
                    self._prefix_to_shard[m.prefix] = m.receiver_shard
        self._action_history = [m.to_tuple() for m in migrations]
        self._epoch += 1

        batch = self._next_batch()
        if batch is None:
            return state, 0.0, False, True, {"epoch": self._epoch, "migrations": len(migrations), "truncated": True}

        next_stats = aggregate_epoch(
            batch,
            prefix_to_shard=self._prefix_to_shard.tolist(),
            num_shards=self.num_shards,
            num_prefixes=self.num_prefixes,
            prefix_bits=self.config.prefix_bits,
        )
        next_state = raw_stats_to_aero_state(
            next_stats,
            num_shards=self.num_shards,
            num_prefixes=self.num_prefixes,
            prev=self._prev_stats,
        )
        self._prev_stats = next_stats
        reward = reward_from_state(next_state, w1=self.config.w1, w2=self.config.w2)
        info = {"epoch": self._epoch, "migrations": len(migrations), "block_end": batch.block_end}
        return next_state, reward, False, False, info

    def get_action_history_for_policy(self) -> np.ndarray:
        L = self.config.action_history_len
        out = np.zeros((L, 3), dtype=np.float32)
        h = self._action_history[-L:]
        for i, (s, t, p) in enumerate(h):
            out[i, 0], out[i, 1], out[i, 2] = s, t, p
        return out

    def set_state(
        self,
        prefix_to_shard: np.ndarray,
        epoch: int,
        action_history_tuples: List[Tuple[int, int, int]],
    ) -> None:
        """仅做兼容；CSV 流位置无法完全恢复，一般用于 --resume 时只恢复网络。"""
        self._prefix_to_shard = np.asarray(prefix_to_shard, dtype=np.int32)
        self._epoch = int(epoch)
        self._action_history = list(action_history_tuples)
