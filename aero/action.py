"""
AERO 中「动作」的表示（对应论文 Section 4.2, 式 (6)–(7)）。

在一轮 epoch 里，RL 代理不会只迁移一个账号，而是生成一串「前缀级」迁移：

    a_t = [a^(1)_t, ..., a^(n_t)_t]
    a^(i)_t = (A^(i1)_t, A^(i2)_t, p)

其中：
- A^(i1)_t: 源 shard 的编号（sender shard index）；
- A^(i2)_t: 目标 shard 的编号（receiver shard index）；
- p: 要迁移的账户前缀（prefix id）。

这里用 MigrationTransaction 把这三者打包成一个简单的数据结构。
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class MigrationTransaction:
    """
    Single migration: move accounts with prefix p from sender_shard to receiver_shard.
    Eq. (1): MigrationTransaction = {sender, receiver, p}
    """

    sender_shard: int  # source physical shard index
    receiver_shard: int  # target physical shard index
    prefix: int  # prefix id in [0, num_prefixes-1]; or use string "0x1a" externally

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.sender_shard, self.receiver_shard, self.prefix)


def action_to_migrations(action_tensor: np.ndarray) -> List[MigrationTransaction]:
    """
    Convert raw action matrix (n x 3) to list of MigrationTransaction.
    action_tensor[i] = [src_shard, tgt_shard, prefix].
    """
    out = []
    for i in range(action_tensor.shape[0]):
        s, t, p = int(action_tensor[i, 0]), int(action_tensor[i, 1]), int(action_tensor[i, 2])
        out.append(MigrationTransaction(sender_shard=s, receiver_shard=t, prefix=p))
    return out


def migrations_to_action_list(migrations: List[MigrationTransaction]) -> List[Tuple[int, int, int]]:
    """List of (sender, receiver, prefix) for encoding as action history."""
    return [m.to_tuple() for m in migrations]


def dedup_migrations(
    plan: List[MigrationTransaction],
    num_shards: int,
    num_prefixes: int,
) -> List[MigrationTransaction]:
    """
    同一 epoch 内按 prefix 去重：保留每个 prefix 的最后一条有效迁移。

    过滤规则：
    - sender_shard == receiver_shard（无效：原地不动）
    - prefix / shard 越界
    - 同一 prefix 出现多次时只保留最后一条（避免互相冲突 / 回滚）
    """
    seen: set = set()
    out: List[MigrationTransaction] = []
    for m in reversed(plan):
        if not (0 <= m.prefix < num_prefixes):
            continue
        if not (0 <= m.sender_shard < num_shards and 0 <= m.receiver_shard < num_shards):
            continue
        if m.sender_shard == m.receiver_shard:
            continue
        if m.prefix in seen:
            continue
        seen.add(m.prefix)
        out.append(m)
    out.reverse()
    return out
