"""
Training script for AERO DRL (CSV-only).

Usage:
  python train.py --csv-path <path_to_BlockTransaction.csv> [--total-steps ...]

本脚本仅保留基于真实 CSV 的训练流程，不再支持模拟环境训练分支。
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from aero.config import AEROConfig
from aero.env_csv import AEROEnvCSV
from aero.network import AEROPolicyValueNet
from aero.ppo import autoregressive_sample, ppo_update


def save_csv_state(path: Path, prefix_to_shard: list, prev, block_end: int, epoch_count: int) -> None:
    """Save dryrun progress so run can be resumed."""
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


def load_csv_state(path: Path, RawEpochStats):
    """Load dryrun progress."""
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


def run_dryrun(args):
    """
    Dryrun mode: stream CSV -> aggregate -> build state -> print reward.
    No PPO updates, no model parameter changes.
    """
    from data import stream_epochs, aggregate_epoch, RawEpochStats
    from aero.data_adapter import raw_stats_to_aero_state
    from aero.reward import reward_from_state

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    config = AEROConfig()
    N, P = config.num_shards, config.num_prefixes
    blocks_per_epoch = args.blocks_per_epoch

    if args.resume_state:
        resume_path = Path(args.resume_state)
        if resume_path.exists():
            prefix_to_shard, prev, resume_after_block, epoch_start = load_csv_state(resume_path, RawEpochStats)
            print(f"Resumed from {resume_path} (after block {resume_after_block}, epoch count {epoch_start})")
        else:
            print(f"Resume state not found: {resume_path}, starting from scratch")
            rng = np.random.default_rng(args.seed)
            prefix_to_shard = (rng.integers(0, N, size=P)).tolist()
            prev = None
            resume_after_block = -1
            epoch_start = 0
    else:
        rng = np.random.default_rng(args.seed)
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
        r = reward_from_state(state, w1=config.w1, w2=config.w2, v_scale=config.reward_v_scale)
        print(
            f"epoch blocks [{batch.block_start},{batch.block_end}] "
            f"txs={len(batch.transactions)} reward={r:.6f} state_dim={state.to_vector().shape[0]}"
        )
        prev = stats
        epoch_count += 1

        if args.save_state and (epoch_count % args.save_state_every == 0):
            save_csv_state(Path(args.save_state), prefix_to_shard, prev, batch.block_end, epoch_count)
            print(f"  Saved state to {args.save_state} (block_end={batch.block_end})")

        if args.max_epochs is not None and epoch_count >= args.max_epochs:
            print(f"(Reached max_epochs={args.max_epochs})")
            break

    if args.save_state and prev is not None and last_batch_end >= 0:
        save_csv_state(Path(args.save_state), prefix_to_shard, prev, last_batch_end, epoch_count)
        print(f"Final state saved to {args.save_state}")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "dryrun"], default="train", help="train=CSV PPO training, dryrun=stream CSV and compute reward only")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", type=str, default=None, help="Override device: 'cuda', 'cpu', or leave empty for auto (GPU if available)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path (e.g. checkpoints/aero_5000.pt)")
    parser.add_argument("--save-every", type=int, default=2500, help="Save checkpoint every N steps (default 2500)")
    parser.add_argument("--csv-path", type=str, required=True, help="CSV data for training (e.g. 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv)")
    parser.add_argument("--blocks-per-epoch", type=int, default=100, help="Blocks per epoch when using CSV (default 100)")
    parser.add_argument("--save-state", type=str, default=None, help="(dryrun) Save stream progress to npz")
    parser.add_argument("--save-state-every", type=int, default=10, help="(dryrun) Save state every N epochs")
    parser.add_argument("--resume-state", type=str, default=None, help="(dryrun) Resume from saved state npz")
    parser.add_argument("--max-epochs", type=int, default=None, help="(dryrun) Stop after this many epochs")
    args = parser.parse_args()

    if args.mode == "dryrun":
        run_dryrun(args)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = AEROConfig()
    config.total_timesteps = args.total_steps
    config.batch_size = args.batch_size
    config.log_interval = args.log_interval

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    env = AEROEnvCSV(config, csv_path, blocks_per_epoch=args.blocks_per_epoch, seed=args.seed)
    print(f"Training with CSV data: {csv_path} (blocks_per_epoch={args.blocks_per_epoch})")
    s0, _ = env.reset()
    state_dim = s0.to_vector().shape[0]
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    net = AEROPolicyValueNet(config, state_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    L = config.action_history_len
    K = config.max_migrations_per_epoch
    N, P = config.num_shards, config.num_prefixes

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    next_state = s0
    action_history = np.asarray(env.get_action_history_for_policy(), dtype=np.float32).reshape(1, L, 3)

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            net.load_state_dict(ckpt["net"])
            if "config" in ckpt:
                config = ckpt["config"]
                config.total_timesteps = args.total_steps
                config.batch_size = args.batch_size
                # 关键修复：resume 后 config 可能变化，需同步重算依赖变量
                L = config.action_history_len
                K = config.max_migrations_per_epoch
                N, P = config.num_shards, config.num_prefixes
            global_step = ckpt.get("global_step", 0)
            # CSV 流位置无法严格恢复；resume 时仅恢复网络参数/步数，环境从 CSV 头重置继续
            next_state, _ = env.reset()
            action_history = np.asarray(env.get_action_history_for_policy(), dtype=np.float32).reshape(1, L, 3)
            print(f"Resumed from {ckpt_path} at step {global_step}")
        else:
            print(f"Resume path not found: {ckpt_path}, starting from scratch")

    print(f"Training started (total_steps={config.total_timesteps}, batch_size={config.batch_size}).")
    print("First log line appears after collecting 1 full batch + PPO update (may take 1–2 min on CPU)...")
    print("Press Ctrl+C to stop and save checkpoint.")

    try:
        while global_step < config.total_timesteps:
            states_buf = []
            action_hist_buf = []
            action_seqs_buf = []
            old_log_probs_buf = []
            rewards_buf = []
            dones_buf = []

            for _ in range(config.batch_size):
                # 记录“执行动作前”的 history，供 PPO 重算 log_prob 时使用
                action_history_before = action_history.copy()
                state_vec = next_state.to_vector()
                st = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
                ah = torch.tensor(action_history_before, dtype=torch.float32, device=device)
                with torch.no_grad():
                    actions, log_prob = autoregressive_sample(
                        net, st, ah, K, N, P
                    )
                act_np = actions[0].cpu().numpy()

                next_state, reward, done, truncated, info = env.step(next_state, act_np)
                action_history = np.zeros((1, L, 3), dtype=np.float32)
                if info.get("migrations"):
                    nh = min(L, info["migrations"])
                    for i, m in enumerate(env._action_history[-nh:]):
                        action_history[0, L - nh + i, :] = m

                states_buf.append(state_vec)
                action_hist_buf.append(action_history_before[0])
                action_seqs_buf.append(act_np)
                old_log_probs_buf.append(log_prob[0].item())
                rewards_buf.append(reward)
                dones_buf.append(done or truncated)

                if done or truncated:
                    next_state, _ = env.reset()
                    action_history = np.asarray(env.get_action_history_for_policy(), dtype=np.float32).reshape(1, L, 3)

                global_step += 1
                # 每 16 步打印一次，避免长时间无输出
                if global_step % 16 == 0:
                    print(f"  ... step {global_step}/{config.total_timesteps} (collecting batch)")
                if global_step >= config.total_timesteps:
                    break

            states = np.stack(states_buf)
            action_hist = np.stack(action_hist_buf)
            action_seqs = np.stack(action_seqs_buf)
            old_lp = np.array(old_log_probs_buf)
            rewards = np.array(rewards_buf)
            dones = np.array(dones_buf)

            metrics = ppo_update(
                net, optimizer, config,
                states, action_hist, action_seqs, old_lp,
                rewards, dones,
            )
            if (global_step // config.batch_size) % config.log_interval == 0:
                print(
                    f"step {global_step} | reward_mean={rewards.mean():.4f} | "
                    f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f}"
                )
            if global_step > 0 and global_step % args.save_every == 0:
                p = Path(args.save_dir) / f"aero_{global_step}.pt"
                save_dict = {
                    "net": net.state_dict(),
                    "config": config,
                    "global_step": global_step,
                    "state_vec": next_state.to_vector(),
                    "CST_per_shard": np.array(next_state.CST_per_shard),
                    "IST_per_shard": np.array(next_state.IST_per_shard),
                    "action_history": action_history,
                    "prefix_to_shard": env._prefix_to_shard.copy(),
                    "epoch": env._epoch,
                    "env_action_history": env._action_history,
                }
                torch.save(save_dict, p)
                print(f"Saved {p}")

        print("Training done.")
        save_dict = {
            "net": net.state_dict(),
            "config": config,
            "global_step": global_step,
            "state_vec": next_state.to_vector(),
            "CST_per_shard": np.array(next_state.CST_per_shard),
            "IST_per_shard": np.array(next_state.IST_per_shard),
            "action_history": action_history,
            "prefix_to_shard": env._prefix_to_shard.copy(),
            "epoch": env._epoch,
            "env_action_history": env._action_history,
        }
        torch.save(save_dict, Path(args.save_dir) / "aero_final.pt")
    except KeyboardInterrupt:
        save_dir = Path(args.save_dir)
        p = save_dir / f"aero_interrupt_{global_step}.pt"
        save_dict = {
            "net": net.state_dict(),
            "config": config,
            "global_step": global_step,
            "state_vec": next_state.to_vector(),
            "CST_per_shard": np.array(next_state.CST_per_shard),
            "IST_per_shard": np.array(next_state.IST_per_shard),
            "action_history": action_history,
            "prefix_to_shard": env._prefix_to_shard.copy(),
            "epoch": env._epoch,
            "env_action_history": env._action_history,
        }
        torch.save(save_dict, p)
        print(f"\nInterrupted at step {global_step}. Checkpoint saved to {p}")


if __name__ == "__main__":
    run()
