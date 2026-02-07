"""
示例：用 22000000to22249999_BlockTransaction 下的大 CSV 流式读入，作为 AERO 的输入。

与 AERO 的耦合仅在「RawEpochStats -> AEROState」一处（aero.data_adapter），
后续改为 BlockEmulator 时，只需替换数据来源为 BlockEmulator 产出的 RawEpochStats（或等价结构）。

支持检查点：--checkpoint 加载 AERO 模型；--save-state / --resume-state 保存/恢复 CSV 处理进度。
"""

import sys
from pathlib import Path

# 保证项目根在 path 中，便于 import data / aero
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Optional
import argparse
import numpy as np

from data import stream_epochs, aggregate_epoch, RawEpochStats
from aero.data_adapter import raw_stats_to_aero_state
from aero.config import AEROConfig
from aero.reward import reward_from_state


def save_csv_state(
    path: Path,
    prefix_to_shard: list,
    prev: Optional[RawEpochStats],
    block_end: int,
    epoch_count: int,
) -> None:
    """Save progress so run can be resumed."""
    np.savez(
        path,
        prefix_to_shard=np.array(prefix_to_shard),
        block_end=np.int64(block_end),
        epoch_count=np.int64(epoch_count),
        **(
            {
                "cst_per_shard": np.array(prev.cst_per_shard),
                "ist_per_shard": np.array(prev.ist_per_shard),
                "txc": np.array(prev.txc),
                "txi": np.array(prev.txi),
                "prev_block_start": np.int64(prev.block_start),
                "prev_block_end": np.int64(prev.block_end),
            }
            if prev is not None
            else {}
        ),
    )


def load_csv_state(path: Path):
    """Load progress. Returns (prefix_to_shard, prev RawEpochStats or None, block_end, epoch_count)."""
    d = np.load(path, allow_pickle=False)
    prefix_to_shard = d["prefix_to_shard"].tolist()
    block_end = int(d["block_end"])
    epoch_count = int(d["epoch_count"])
    if "cst_per_shard" in d:
        prev = RawEpochStats(
            block_start=int(d["prev_block_start"]),
            block_end=int(d["prev_block_end"]),
            cst_per_shard=d["cst_per_shard"].tolist(),
            ist_per_shard=d["ist_per_shard"].tolist(),
            txc=d["txc"].tolist(),
            txi=d["txi"].tolist(),
        )
    else:
        prev = None
    return prefix_to_shard, prev, block_end, epoch_count


def main():
    parser = argparse.ArgumentParser(description="Stream CSV into AERO state and compute reward (with optional checkpoint).")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to CSV (default: 22000000to22249999_BlockTransaction/...)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load AERO model checkpoint for inference (e.g. checkpoints/aero_final.pt)")
    parser.add_argument("--save-state", type=str, default=None, help="Save progress to this path (e.g. csv_state.npz)")
    parser.add_argument("--save-state-every", type=int, default=10, help="Save state every N epochs (default 10)")
    parser.add_argument("--resume-state", type=str, default=None, help="Resume from saved state file")
    parser.add_argument("--max-epochs", type=int, default=None, help="Stop after this many epochs (default: run until 22000100 for demo)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else (ROOT / "22000000to22249999_BlockTransaction" / "22000000to22249999_BlockTransaction.csv")
    if not csv_path.exists():
        print(f"未找到 CSV: {csv_path}")
        return

    config = AEROConfig()
    N, P = config.num_shards, config.num_prefixes
    blocks_per_epoch = 100

    net, aero_config = None, config
    if args.checkpoint:
        import torch
        from aero.infer import load_aero
        net, aero_config = load_aero(args.checkpoint)
        print(f"Loaded AERO checkpoint: {args.checkpoint}")

    # prefix -> shard: from resume or random
    if args.resume_state:
        resume_path = Path(args.resume_state)
        if resume_path.exists():
            prefix_to_shard, prev, resume_after_block, epoch_start = load_csv_state(resume_path)
            print(f"Resumed from {resume_path} (after block {resume_after_block}, epoch count {epoch_start})")
        else:
            print(f"Resume state not found: {resume_path}, starting from scratch")
            rng = np.random.default_rng(42)
            prefix_to_shard = (rng.integers(0, N, size=P)).tolist()
            prev = None
            resume_after_block = -1
            epoch_start = 0
    else:
        rng = np.random.default_rng(42)
        prefix_to_shard = (rng.integers(0, N, size=P)).tolist()
        prev = None
        resume_after_block = -1
        epoch_start = 0

    epoch_count = epoch_start
    last_batch_end = resume_after_block
    for batch in stream_epochs(csv_path, blocks_per_epoch=blocks_per_epoch):
        if batch.block_end <= resume_after_block:
            continue

        last_batch_end = batch.block_end
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
        epoch_count += 1

        if args.save_state and (epoch_count % args.save_state_every == 0):
            save_csv_state(
                Path(args.save_state),
                prefix_to_shard,
                prev,
                batch.block_end,
                epoch_count,
            )
            print(f"  Saved state to {args.save_state} (block_end={batch.block_end})")

        if args.max_epochs is not None and epoch_count >= args.max_epochs:
            print(f"(Reached max_epochs={args.max_epochs})")
            break
        if args.max_epochs is None and batch.block_start >= 22000100:
            print("(提前结束示例)")
            break

    if args.save_state and prev is not None and last_batch_end >= 0:
        save_csv_state(Path(args.save_state), prefix_to_shard, prev, last_batch_end, epoch_count)
        print(f"Final state saved to {args.save_state}")


if __name__ == "__main__":
    main()
