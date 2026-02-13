"""
PPO 训练逻辑（对应论文 Section 4 中「用 PPO 优化策略」的部分）。

--- 关键改动 ---
动作分布从连续 Gaussian 改为离散 Categorical：
  - src_shard, tgt_shard: Categorical(num_shards)
  - prefix: Categorical(num_prefixes)
自回归函数（sample / log_prob / entropy）相应更新。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Tuple

from .config import AEROConfig
from .network import AEROPolicyValueNet
from .state import AEROState


def autoregressive_log_prob(
    net: AEROPolicyValueNet,
    state_vec: torch.Tensor,
    init_action_history: torch.Tensor,
    action_seq: torch.Tensor,
    num_shards: int,
    num_prefixes: int,
) -> torch.Tensor:
    """
    计算整段动作序列在当前策略下的 log_prob（Categorical 版本）。

    action_seq: (B, K, 3)，值为离散索引（整数值的 float）。
    """
    B, K, _ = action_seq.shape
    action_history = init_action_history.clone()
    log_probs = []
    for k in range(K):
        src_logits, tgt_logits, prefix_logits, _ = net(state_vec, action_history)
        src_dist = torch.distributions.Categorical(logits=src_logits)
        tgt_dist = torch.distributions.Categorical(logits=tgt_logits)
        prefix_dist = torch.distributions.Categorical(logits=prefix_logits)
        a_k = action_seq[:, k, :]  # (B, 3) float, integer-valued
        lp = (src_dist.log_prob(a_k[:, 0].long())
              + tgt_dist.log_prob(a_k[:, 1].long())
              + prefix_dist.log_prob(a_k[:, 2].long()))
        log_probs.append(lp)
        # 用当前动作更新 history（滑窗）
        a_clip = a_k.clone()
        a_clip[:, 0] = a_k[:, 0].clamp(0, num_shards - 1)
        a_clip[:, 1] = a_k[:, 1].clamp(0, num_shards - 1)
        a_clip[:, 2] = a_k[:, 2].clamp(0, num_prefixes - 1)
        action_history = torch.cat([action_history[:, 1:], a_clip.unsqueeze(1)], dim=1)
    return torch.stack(log_probs, dim=1).sum(dim=1)


def autoregressive_entropy(
    net: AEROPolicyValueNet,
    state_vec: torch.Tensor,
    init_action_history: torch.Tensor,
    action_seq: torch.Tensor,
    num_shards: int,
    num_prefixes: int,
) -> torch.Tensor:
    """按序列累计策略熵（Categorical 版本）。"""
    B, K, _ = action_seq.shape
    action_history = init_action_history.clone()
    entropies = []
    for k in range(K):
        src_logits, tgt_logits, prefix_logits, _ = net(state_vec, action_history)
        src_dist = torch.distributions.Categorical(logits=src_logits)
        tgt_dist = torch.distributions.Categorical(logits=tgt_logits)
        prefix_dist = torch.distributions.Categorical(logits=prefix_logits)
        ent = src_dist.entropy() + tgt_dist.entropy() + prefix_dist.entropy()
        entropies.append(ent)
        a_k = action_seq[:, k, :]
        a_clip = a_k.clone()
        a_clip[:, 0] = a_k[:, 0].clamp(0, num_shards - 1)
        a_clip[:, 1] = a_k[:, 1].clamp(0, num_shards - 1)
        a_clip[:, 2] = a_k[:, 2].clamp(0, num_prefixes - 1)
        action_history = torch.cat([action_history[:, 1:], a_clip.unsqueeze(1)], dim=1)
    return torch.stack(entropies, dim=1).sum(dim=1)


def autoregressive_sample(
    net: AEROPolicyValueNet,
    state_vec: torch.Tensor,
    action_history: torch.Tensor,
    max_migrations: int,
    num_shards: int,
    num_prefixes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    自回归地采样一整段离散迁移序列。
    返回 actions (B, K, 3) float tensor（值为离散索引）, log_prob (B,).
    """
    B = state_vec.shape[0]
    device = state_vec.device
    actions_list = []
    log_probs_list = []
    h = action_history
    for _ in range(max_migrations):
        src_logits, tgt_logits, prefix_logits, _ = net(state_vec, h)
        src_dist = torch.distributions.Categorical(logits=src_logits)
        tgt_dist = torch.distributions.Categorical(logits=tgt_logits)
        prefix_dist = torch.distributions.Categorical(logits=prefix_logits)
        src = src_dist.sample()      # (B,) long
        tgt = tgt_dist.sample()      # (B,) long
        pref = prefix_dist.sample()  # (B,) long
        a = torch.stack([src, tgt, pref], dim=-1).float()  # (B, 3)
        lp = (src_dist.log_prob(src)
              + tgt_dist.log_prob(tgt)
              + prefix_dist.log_prob(pref))
        log_probs_list.append(lp)
        actions_list.append(a)
        # Update history: append and keep last L
        h = torch.cat([h[:, 1:], a.unsqueeze(1)], dim=1)
    actions = torch.stack(actions_list, dim=1)  # (B, K, 3)
    log_prob = torch.stack(log_probs_list, dim=1).sum(dim=1)  # (B,)
    return actions, log_prob


def ppo_update(
    net: AEROPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    config: AEROConfig,
    states: np.ndarray,
    action_histories: np.ndarray,
    action_seqs: np.ndarray,
    old_log_probs: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
) -> dict:
    """
    用一批采样到的轨迹数据做一次 PPO 更新。
    action_seqs 里的值是离散索引（整数值的 float）。

    关键步骤：
    1. 反向累积奖励得到 return（折扣和）；
    2. return - value 得到 advantage，并做标准化；
    3. 重新用当前策略算 log_prob，得到 ratio = exp(new - old)；
    4. 按 PPO 的 clip 公式计算策略损失，加上价值损失和熵正则，共同反向传播。
    """
    device = next(net.parameters()).device
    T = states.shape[0]
    returns = np.zeros(T)
    g = 0
    for t in reversed(range(T)):
        if dones[t]:
            g = 0
        g = rewards[t] + config.gamma * g
        returns[t] = g
    st = torch.tensor(states, dtype=torch.float32, device=device)
    ah = torch.tensor(action_histories, dtype=torch.float32, device=device)
    _, _, _, values = net(st, ah)
    values_np = values.detach().cpu().numpy()
    adv = returns - values_np
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
    action_seqs_t = torch.tensor(action_seqs, dtype=torch.float32, device=device)
    old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)

    dataset = TensorDataset(st, ah, action_seqs_t, old_lp_t, returns_t, adv_t)
    loader = DataLoader(dataset, batch_size=config.mini_batch_size, shuffle=True)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_batches = 0
    for _ in range(config.epochs_per_update):
        for sb_st, sb_ah, sb_act, sb_old_lp, sb_ret, sb_adv in loader:
            new_log_prob = autoregressive_log_prob(
                net, sb_st, sb_ah, sb_act,
                config.num_shards, config.num_prefixes,
            )
            _, _, _, new_value = net(sb_st, sb_ah)
            seq_entropy = autoregressive_entropy(
                net, sb_st, sb_ah, sb_act,
                config.num_shards, config.num_prefixes,
            )
            ratio = (new_log_prob - sb_old_lp).exp()
            surr1 = ratio * sb_adv
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * sb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_value, sb_ret)
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * seq_entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += seq_entropy.mean().item()
            n_batches += 1
    return {
        "policy_loss": total_policy_loss / max(n_batches, 1),
        "value_loss": total_value_loss / max(n_batches, 1),
        "entropy": total_entropy / max(n_batches, 1),
    }
