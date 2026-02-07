"""
基于注意力机制的策略 + 价值网络（对应论文 Appendix C, 式 (14)–(17)）。

可以把它理解成一件事：
- 输入：当前状态 s_t + 最近若干轮的迁移历史 a_{t-1}, a_{t-2}, ...；
- 通过 Multi-Head Attention，把「当前形势」和「历史行为模式」融合成一个上下文向量 h；
- 再由一个小 MLP 输出「本轮下一个迁移」的动作分布（src shard, tgt shard, prefix）。

相比简单 MLP，这种结构更容易捕捉：
- 某些 prefix 之前经常被迁移到哪些 shard；
- 某种迁移模式之后，系统的 CST/负载会如何变化。
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
        # Each action (src, tgt, prefix): normalize to embeddings
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
    策略网络： (状态 s + 动作历史) -> 上下文 h -> 动作分布。

    对应论文 Appendix C 的描述：
    - Q（query）来自当前状态 s；
    - K/V（key/value）来自历史动作编码 H_enc；
    - Multi-Head Attention 计算出若干个 head 的输出，再拼接成 h；
    - 用 W_out h + b_out 得到一个 3 维的动作向量，表示 (src, tgt, prefix) 的「均值」。

    在 PPO 里我们把它当作高斯策略：
    - action_mean: h 经过 MLP 的输出；
    - log_std: 额外学习到的对数标准差（全局 3 维）；
    - 采样时：Normal(action_mean, std)。
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

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, config.num_neurons),
            nn.ReLU(),
            nn.Linear(config.num_neurons, d_model),
        )
        self.action_encoder = ActionHistoryEncoder(d_model, N, P)

        # Q from state: per head W_qi (d_h, d_model) — we use state encoded to d_model
        self.W_q = nn.Linear(d_model, self.d_h * self.num_heads)  # state -> all Qs
        # K, V from Henc: (L, d_model) -> (L, d_h) per head
        self.W_k = nn.Linear(d_model, self.d_h * self.num_heads)
        self.W_v = nn.Linear(d_model, self.d_h * self.num_heads)
        self.W_O = nn.Linear(self.num_heads * self.d_h, d_model)

        # Decoder: h -> action (src, tgt, prefix) as continuous R^3; PPO will use Gaussian policy
        self.action_head = nn.Sequential(
            nn.Linear(d_model, config.num_neurons),
            nn.ReLU(),
            nn.Linear(config.num_neurons, 3),
        )
        self.log_std_head = nn.Parameter(torch.zeros(3))

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

    def forward(
        self,
        state_vec: torch.Tensor,
        action_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns action_mean (B, 3), log_std (3) for Gaussian policy.
        action in R^3 will be interpreted as (src_shard, tgt_shard, prefix) after scaling.
        """
        h = self._context(state_vec, action_history)
        action_mean = self.action_head(h)  # (B, 3)
        log_std = self.log_std_head.clamp(-20, 2)
        return action_mean, log_std



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
        action_mean, log_std = self.policy_net(state_vec, action_history)
        h = self.policy_net._context(state_vec, action_history)
        value = self.value_head(h).squeeze(-1)
        return action_mean, log_std, value

    def get_action_and_value(
        self,
        state_vec: torch.Tensor,
        action_history: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, log_std, value = self.forward(state_vec, action_history)
        std = log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value
