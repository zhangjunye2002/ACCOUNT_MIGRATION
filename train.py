"""
Training script for AERO DRL (standalone).

Usage:
  python train.py [--total-steps 100000] [--batch-size 128] [--seed 0]

When integrating with BlockEmulator, replace AEROEnv with an adapter that
feeds state/action to the emulator and returns next_state, reward from the
emulator.
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from aero.config import AEROConfig
from aero.env import AEROEnv
from aero.env_csv import AEROEnvCSV
from aero.network import AEROPolicyValueNet
from aero.ppo import autoregressive_sample, ppo_update
from aero.state import AEROState


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", type=str, default=None, help="Override device: 'cuda', 'cpu', or leave empty for auto (GPU if available)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path (e.g. checkpoints/aero_5000.pt)")
    parser.add_argument("--save-every", type=int, default=2500, help="Save checkpoint every N steps (default 2500)")
    parser.add_argument("--csv-path", type=str, default=None, help="Use CSV data for training (e.g. 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv)")
    parser.add_argument("--blocks-per-epoch", type=int, default=100, help="Blocks per epoch when using CSV (default 100)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = AEROConfig()
    config.total_timesteps = args.total_steps
    config.batch_size = args.batch_size
    config.log_interval = args.log_interval

    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        env = AEROEnvCSV(config, csv_path, blocks_per_epoch=args.blocks_per_epoch, seed=args.seed)
        print(f"Training with CSV data: {csv_path} (blocks_per_epoch={args.blocks_per_epoch})")
    else:
        env = AEROEnv(config, seed=args.seed)
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
            global_step = ckpt.get("global_step", 0)
            if not args.csv_path and "state_vec" in ckpt and "action_history" in ckpt:
                next_state = AEROState.from_vector(
                    ckpt["state_vec"],
                    ckpt["CST_per_shard"],
                    ckpt["IST_per_shard"],
                    N, P,
                )
                ah = ckpt["action_history"]
                if ah.ndim == 2:
                    action_history = np.asarray(ah, dtype=np.float32).reshape(1, L, 3)
                else:
                    action_history = np.asarray(ah, dtype=np.float32)
                if "prefix_to_shard" in ckpt and "epoch" in ckpt and "env_action_history" in ckpt:
                    env.set_state(
                        ckpt["prefix_to_shard"],
                        int(ckpt["epoch"]),
                        [tuple(x) for x in ckpt["env_action_history"]],
                    )
            elif args.csv_path:
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
                state_vec = next_state.to_vector()
                st = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
                ah = torch.tensor(action_history, dtype=torch.float32, device=device)
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
                action_hist_buf.append(action_history[0])
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
