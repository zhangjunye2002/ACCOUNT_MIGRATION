"""
基于注意力机制的策略 + 价值网络（对应论文 Appendix C, 式 (14)–(17)）。

--- 关键设计 ---
动作空间改为「离散 Categorical」：
  - src_shard / tgt_shard: 各 Categorical(num_shards)
  - prefix: Categorical(num_prefixes)

之前用连续 Gaussian + round 的方式：
  - 16 个分片的 Gaussian 还勉强可用（std=1 覆盖 ~5 个值）；
  - 但 256 个前缀时 Gaussian 只能覆盖中心附近极少数值 → 严重的"前缀塌缩"。
改用 Categorical 后，每个 prefix 都有独立 logit，不受 Gaussian 覆盖范围限制。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import AEROConfig


class ActionHistoryEncoder(nn.Module):
    """
    把最近 L 条迁移动作编码成向量序列 H_enc (L, d_model)。

    每条动作是 (src, tgt, prefix) 三个整数：
    - 分别查 embedding（就像 NLP 里查词向量）；
    - 再拼接后过一层线性层投影到统一的 d_model 维空间。
    """

    def __init__(self, d_model: int, num_shards: int, num_prefixes: int):
        super().__init__()
        self.d_model = d_model
        # 每个分量用 d_model//3 维，拼接后为 3*(d_model//3)，可能 < d_model（如 256→255）
        self._emb_dim = d_model // 3
        self.embed_src = nn.Embedding(num_shards + 1, self._emb_dim)  # +1 for padding
        self.embed_tgt = nn.Embedding(num_shards + 1, self._emb_dim)
        self.embed_prefix = nn.Embedding(num_prefixes + 1, self._emb_dim)
        self.proj = nn.Linear(3 * self._emb_dim, d_model)

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """
        action_history: (batch, L, 3) in [0, N-1] x [0, N-1] x [0, P-1]; use N/P for padding.
        """
        B, L, _ = action_history.shape
        src = action_history[..., 0].long().clamp(0, self.embed_src.num_embeddings - 1)
        tgt = action_history[..., 1].long().clamp(0, self.embed_tgt.num_embeddings - 1)
        pref = action_history[..., 2].long().clamp(0, self.embed_prefix.num_embeddings - 1)
        e = torch.cat([
            self.embed_src(src),
            self.embed_tgt(tgt),
            self.embed_prefix(pref),
        ], dim=-1)
        return self.proj(e)  # (B, L, d_model)


class AEROAttentionPolicy(nn.Module):
    """
    策略网络：(状态 s + 动作历史) -> 上下文 h -> 离散动作分布。

    Q (query) 来自状态 s，K/V 来自历史动作编码 H_enc，
    Multi-Head Attention 融合后得到上下文 h，
    再通过三个独立的线性头输出 src / tgt / prefix 的 Categorical logits。
    """

    def __init__(self, config: AEROConfig, state_dim: int):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        d_model = config.d_model
        self.d_h = d_h = max(1, config.d_model // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.d_model = d_model
        N = config.num_shards
        P = config.num_prefixes

        # ---- 状态编码 ----
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config.num_neurons),
            nn.ReLU(),
            nn.Linear(config.num_neurons, d_model),
        )
        # ---- 动作历史编码 ----
        self.action_encoder = ActionHistoryEncoder(d_model, N, P)

        # ---- Multi-Head Attention ----
        # Q from state, K/V from action history
        self.W_q = nn.Linear(d_model, self.d_h * self.num_heads)
        self.W_k = nn.Linear(d_model, self.d_h * self.num_heads)
        self.W_v = nn.Linear(d_model, self.d_h * self.num_heads)
        self.W_O = nn.Linear(self.num_heads * self.d_h, d_model)

        # ---- 离散 Categorical 动作头 ----
        # 共享隐藏层 → 两个独立 logits 分支（去掉 src_head，因为环境不使用策略输出的 src）
        self.action_hidden = nn.Sequential(
            nn.Linear(d_model, config.num_neurons),
            nn.ReLU(),
        )
        self.tgt_head = nn.Linear(config.num_neurons, N)    # → Categorical(N)
        self.prefix_head = nn.Linear(config.num_neurons, P) # → Categorical(P)

    def _context(self, state_vec: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        """
        state_vec: (B, state_dim), action_history: (B, L, 3)
        Returns context h: (B, d_model).
        """
        B = state_vec.shape[0]
        s_enc = self.state_encoder(state_vec)   # (B, d_model)
        Henc = self.action_encoder(action_history)  # (B, L, d_model)

        Q = self.W_q(s_enc).view(B, self.num_heads, self.d_h)   # (B, H, d_h)
        K = self.W_k(Henc).view(B, Henc.size(1), self.num_heads, self.d_h).transpose(1, 2)  # (B, H, L, d_h)
        V = self.W_v(Henc).view(B, Henc.size(1), self.num_heads, self.d_h).transpose(1, 2)   # (B, H, L, d_h)

        # Q (B,H,d_h), K (B,H,L,d_h) -> scores (B,H,L)
        scores = torch.matmul(Q.unsqueeze(2), K.transpose(-2, -1)).squeeze(2) / (self.d_h ** 0.5)
        attn = F.softmax(scores, dim=-1)  # (B, H, L)
        head_out = torch.einsum("bhl,bhld->bhd", attn, V)
        h_cat = head_out.reshape(B, -1)  # (B, H*d_h)
        return self.W_O(h_cat)  # (B, d_model)

    def _logits_from_context(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从上下文 h 计算两个动作维度的 logits（tgt_shard, prefix）。"""
        a_h = self.action_hidden(h)
        return self.tgt_head(a_h), self.prefix_head(a_h)

    def forward(
        self,
        state_vec: torch.Tensor,
        action_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (tgt_logits, prefix_logits).
        - tgt_logits:    (B, num_shards)
        - prefix_logits: (B, num_prefixes)
        """
        h = self._context(state_vec, action_history)
        return self._logits_from_context(h)


class AEROPolicyValueNet(nn.Module):
    """Policy + value in one module for PPO (shared representation)."""

    def __init__(self, config: AEROConfig, state_dim: int):
        super().__init__()
        self.policy_net = AEROAttentionPolicy(config, state_dim)
        self.value_head = nn.Linear(config.d_model, 1)

    def get_context(self, state_vec: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        return self.policy_net._context(state_vec, action_history)

    def forward(
        self,
        state_vec: torch.Tensor,
        action_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (tgt_logits, prefix_logits, value).
        只计算一次 _context，同时用于策略和价值。
        """
        h = self.policy_net._context(state_vec, action_history)
        tgt_l, pref_l = self.policy_net._logits_from_context(h)
        value = self.value_head(h).squeeze(-1)
        return tgt_l, pref_l, value

    def get_action_and_value(
        self,
        state_vec: torch.Tensor,
        action_history: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样/评估一条离散动作 (tgt, prefix)。
        action: (B, 2) float tensor（值为离散索引）; None 则采样。
        Returns (action, log_prob, entropy, value).
        """
        tgt_l, pref_l, value = self.forward(state_vec, action_history)
        tgt_dist = torch.distributions.Categorical(logits=tgt_l)
        prefix_dist = torch.distributions.Categorical(logits=pref_l)
        if action is None:
            tgt = tgt_dist.sample()
            pref = prefix_dist.sample()
            action = torch.stack([tgt, pref], dim=-1).float()
        else:
            tgt = action[..., 0].long()
            pref = action[..., 1].long()
        log_prob = (tgt_dist.log_prob(tgt)
                    + prefix_dist.log_prob(pref))
        entropy = (tgt_dist.entropy()
                   + prefix_dist.entropy())
        return action, log_prob, entropy, value
