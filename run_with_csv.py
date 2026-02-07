"""
示例：用 22000000to22249999_BlockTransaction 下的大 CSV 流式读入，作为 AERO 的输入。

与 AERO 的耦合仅在「RawEpochStats -> AEROState」一处（aero.data_adapter），
后续改为 BlockEmulator 时，只需替换数据来源为 BlockEmulator 产出的 RawEpochStats（或等价结构）。
"""

import sys
from pathlib import Path

# 保证项目根在 path 中，便于 import data / aero
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Optional

from data import stream_epochs, aggregate_epoch, RawEpochStats
from aero.data_adapter import raw_stats_to_aero_state
from aero.config import AEROConfig
from aero.reward import reward_from_state


def main():
    csv_path = ROOT / "22000000to22249999_BlockTransaction" / "22000000to22249999_BlockTransaction.csv"
    if not csv_path.exists():
        print(f"未找到 CSV: {csv_path}")
        return

    config = AEROConfig()
    N, P = config.num_shards, config.num_prefixes
    blocks_per_epoch = 100

    # prefix -> shard 初始可随机或从别处加载；此处仅为示例
    import numpy as np
    rng = np.random.default_rng(42)
    prefix_to_shard = (rng.integers(0, N, size=P)).tolist()

    prev: Optional[RawEpochStats] = None
    for batch in stream_epochs(csv_path, blocks_per_epoch=blocks_per_epoch):
        stats = aggregate_epoch(
            batch,
            prefix_to_shard=prefix_to_shard,
            num_shards=N,
            num_prefixes=P,
            prefix_bits=config.prefix_bits,
        )
        state = raw_stats_to_aero_state(stats, num_shards=N, num_prefixes=P, prev=prev)
        r = reward_from_state(state, w1=config.w1, w2=config.w2)
        print(
            f"epoch blocks [{batch.block_start},{batch.block_end}] "
            f"txs={len(batch.transactions)} reward={r:.4f} state_dim={state.to_vector().shape[0]}"
        )
        prev = stats

        # 仅跑前几 epoch 做连通性检查；正式训练可接 train 逻辑
        if batch.block_start >= 22000100:
            print("(提前结束示例)")
            break


if __name__ == "__main__":
    main()
