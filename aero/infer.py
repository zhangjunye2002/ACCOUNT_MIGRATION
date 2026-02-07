"""
AERO 的「推理接口」：给定当前状态（以及可选的历史动作），
输出一份迁移计划（若干条 MigrationTransaction）。

当你把 AERO 接到 BlockEmulator 上时，可以按如下方式使用：
- BlockEmulator 每个 epoch 提供一个 AEROState（或等价统计）；
- 把这个状态丢给 get_migration_plan()，得到整轮需要执行的迁移列表；
- 再由区块链系统在 PBFT 重配置阶段按这份计划打 migration transaction。
"""

import numpy as np
import torch
from typing import List, Optional, Union

from .config import AEROConfig
from .state import AEROState
from .action import MigrationTransaction
from .network import AEROPolicyValueNet


def load_aero(checkpoint_path: str, device: str = "cpu") -> tuple:
    """Load trained AERO net and config from checkpoint. Returns (net, config)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("net", ckpt)
    config = ckpt.get("config")
    if config is None:
        config = AEROConfig()
    s0 = AEROState.dummy(config.num_shards, config.num_prefixes)
    state_dim = s0.to_vector().shape[0]
    net = AEROPolicyValueNet(config, state_dim).to(device)
    if "net" in ckpt:
        net.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)
    return net, config


def get_migration_plan(
    net: AEROPolicyValueNet,
    config: AEROConfig,
    state: AEROState,
    action_history: Optional[np.ndarray] = None,
    max_migrations: Optional[int] = None,
    deterministic: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> List[MigrationTransaction]:
    """
    Produce migration plan (list of MigrationTransaction) for the given state.

    - state: current AEROState from the sharding system (or BlockEmulator).
    - action_history: (L, 3) last L migrations for sliding window; if None, use zeros.
    - max_migrations: max length of plan; default config.max_migrations_per_epoch.
    - deterministic: if True, use mean action; else sample.
    """
    if isinstance(device, str):
        device = torch.device(device)
    K = max_migrations or config.max_migrations_per_epoch
    L = config.action_history_len
    N, P = config.num_shards, config.num_prefixes

    state_vec = state.to_vector()
    if action_history is None:
        action_history = np.zeros((L, 3), dtype=np.float32)
    else:
        action_history = np.asarray(action_history, dtype=np.float32)
        if action_history.ndim == 2 and action_history.shape[0] < L:
            pad = np.zeros((L - action_history.shape[0], 3), dtype=np.float32)
            action_history = np.concatenate([pad, action_history], axis=0)
        elif action_history.ndim == 2 and action_history.shape[0] > L:
            action_history = action_history[-L:]

    st = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
    ah = torch.tensor(action_history, dtype=torch.float32, device=device).unsqueeze(0)
    migrations = []
    h = ah
    with torch.no_grad():
        for _ in range(K):
            action_mean, log_std, _ = net(st, h)
            if deterministic:
                a = action_mean[0].cpu().numpy()
            else:
                std = log_std.exp()
                a = (action_mean + torch.randn_like(action_mean, device=device) * std)[0].cpu().numpy()
            src = int(np.clip(np.round(a[0]), 0, N - 1))
            tgt = int(np.clip(np.round(a[1]), 0, N - 1))
            pref = int(np.clip(np.round(a[2]), 0, P - 1))
            if src != tgt:
                migrations.append(MigrationTransaction(sender_shard=src, receiver_shard=tgt, prefix=pref))
            # Update history
            na = np.array([[src, tgt, pref]], dtype=np.float32)
            h_np = h[0].cpu().numpy()
            h_np = np.concatenate([h_np[1:], na], axis=0)
            h = torch.tensor(h_np, dtype=torch.float32, device=device).unsqueeze(0)
    return migrations
