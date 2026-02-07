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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = AEROConfig()
    config.total_timesteps = args.total_steps
    config.batch_size = args.batch_size
    config.log_interval = args.log_interval

    env = AEROEnv(config, seed=args.seed)
    s0, _ = env.reset()
    state_dim = s0.to_vector().shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = AEROPolicyValueNet(config, state_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    L = config.action_history_len
    K = config.max_migrations_per_epoch
    N, P = config.num_shards, config.num_prefixes

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    next_state, _ = env.reset()
    action_history = np.zeros((1, L, 3), dtype=np.float32)

    print(f"Training started (total_steps={config.total_timesteps}, batch_size={config.batch_size}).")
    print("First log line appears after collecting 1 full batch + PPO update (may take 1–2 min on CPU)...")

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
                action_history = np.zeros((1, L, 3), dtype=np.float32)

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
        if global_step % (config.save_freq * config.batch_size) == 0 and global_step > 0:
            p = Path(args.save_dir) / f"aero_{global_step}.pt"
            torch.save({"net": net.state_dict(), "config": config}, p)
            print(f"Saved {p}")

    print("Training done.")
    torch.save(
        {"net": net.state_dict(), "config": config},
        Path(args.save_dir) / "aero_final.pt",
    )


if __name__ == "__main__":
    run()
