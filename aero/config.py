"""
AERO 超参数与整体配置（对应论文 Section 5.1, Appendix B）。

你可以把这个文件理解成：
- 把论文里散落的数字（γ、学习率、head 数量等）集中在一个地方；
- 同时也约定了「链有多少个 shard、地址前多少位作为 prefix」等环境设置。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class AEROConfig:
    """
    AERO DRL 框架的配置。

    使用习惯：
    - 强相关 DRL 的参数（gamma、learning_rate、clip_epsilon 等）都放这里；
    - 如果你以后想调参，只改这个类的默认值即可，不用在训练代码里到处找数字。
    """

    # --- 分片相关 ---
    num_shards: int = 16
    # 前多少 bit 作为地址前缀（比如 8 表示取 0x 后前 2 个十六进制字符），
    # 这样就有 2^prefix_bits 个 prefix 分组，对应论文里的前缀分组思想。
    prefix_bits: int = 8  # 2^8 = 256 prefix groups (e.g. "0x00".."0xff")
    # 每个 epoch 最多迁移多少个前缀（动作序列长度上限 n_t）
    max_migrations_per_epoch: int = 32

    # --- DRL / PPO 相关 (Appendix B, Section 4) ---
    # 折扣因子 γ：越接近 1 越看重「长期」收益
    gamma: float = 0.99
    learning_rate: float = 1e-5
    # 每次更新 PPO 时，用多少条「时间步」样本
    batch_size: int = 128
    # 一个 batch 再拆成几个小 batch 做多次迭代更新
    mini_batch_size: int = 4
    # PPO 的 clip 范围 ε：控制新旧策略的差异不要太大
    clip_epsilon: float = 0.2
    # 价值函数损失的权重
    value_coef: float = 0.5
    # 熵正则系数：越大越鼓励「多探索」
    entropy_coef: float = 0.01
    # 梯度裁剪阈值，防止训练不稳定
    max_grad_norm: float = 0.5
    # 每次收集完一批数据后，对同一批数据重复多少轮 PPO 更新
    epochs_per_update: int = 4

    # --- 奖励函数权重 (Section 4.3) ---
    # w1: 控制「降低跨片交易比例」的重要程度
    w1: float = 1.0
    # w2: 控制「降低 shard 负载方差（更均衡）」的重要程度
    w2: float = 1.0
    # v_t（负方差）缩放系数，避免其量级远大于 u_t 导致训练塌缩到无迁移动作
    # 可以按数据集调整：1e-6 ~ 1e-4
    reward_v_scale: float = 1e-6

    # --- 神经网络结构 (Appendix B, C) ---
    num_attention_heads: int = 6
    # d_model：Attention 中的「特征维度」，可以简单理解成隐藏层大小
    d_model: int = 256
    # d_h：单个 head 的维度，这里主要作为参考，不必和 d_model 完全整除
    d_h: int = 256
    state_embed_dim: int = 256
    # 动作历史长度 L：策略网络会看最近多少条迁移动作作为「上下文」
    action_history_len: int = 20
    # 全连接层的神经元数量
    num_neurons: int = 256

    # --- 训练过程控制 ---
    total_timesteps: int = 1_000_000
    save_freq: int = 100
    log_interval: int = 10
    seed: int = 0

    # prefix 的总数量：2^prefix_bits
    @property
    def num_prefixes(self) -> int:
        return 1 << self.prefix_bits  # 2^prefix_bits
