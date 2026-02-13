# AERO DRL 实现说明

本目录实现论文 **AERO: Enhancing Sharding Blockchain via Deep Reinforcement Learning for Account Migration**（WWW 2025）中的 **DRL 部分**。与 BlockEmulator 的对接留作后续迁移。

## 论文对应关系

| 论文位置 | 实现 |
|---------|------|
| Section 4 MDP | `aero/state.py`（状态）、`aero/action.py`（动作 + 去重）、`aero/reward.py`（奖励） |
| Section 4.1 状态 | `AEROState`：T_t, C_t, V_t, TX^c_t, TX^i_t 及上一 epoch 的 TX^c/TX^i |
| Section 4.2 动作 | `MigrationTransaction(sender_shard, receiver_shard, prefix)`，按 prefix 成组迁移；`dedup_migrations()` 在每 epoch 内按 prefix 去重 |
| Section 4.3 奖励 | R_t = w1·(1 - u_t) + w2·v_t，u_t 为 CSTX 占比，v_t 为负方差。**注意**：最大化 reward 等价于最小化跨片交易比例 |
| Appendix C 网络 | `aero/network.py`：Query 来自 state，Key/Value 来自 action history 的 Multi-Head Attention；动作输出改用 **离散 Categorical 分布**（3 个独立 head：src_shard / tgt_shard / prefix） |
| Appendix B 超参 | `aero/config.py`：γ=0.99, 6 heads, batch 128, lr 1e-5 等 |
| PPO | `aero/ppo.py`：clip + value/entropy loss，自回归序列的 log_prob（Categorical） |

---

## 近期重大变更

### 1. Gaussian → Categorical 动作空间

旧版使用连续 Gaussian 分布对 `(src_shard, tgt_shard, prefix)` 三维空间采样，再 round + clip 取离散值。
这导致 **prefix 坍缩问题**：Gaussian 对 256 个离散 prefix 只能有效覆盖一小段，大量 prefix 永远无法被选中。

新版将策略网络输出改为 **3 个独立的 Categorical 分布**：
- `src_head`：输出 logits `(B, num_shards)` —— 源分片
- `tgt_head`：输出 logits `(B, num_shards)` —— 目标分片
- `prefix_head`：输出 logits `(B, num_prefixes)` —— prefix 选择

每个维度的 log_prob 和 entropy 独立计算后求和。

**影响文件**：`network.py`、`ppo.py`、`infer.py`

> **兼容性**：旧版 Gaussian checkpoint 与新版 Categorical 网络架构不兼容。`train.py --resume` 和 `infer.py` 中有 `try/except` 处理，遇到旧 checkpoint 会打印警告并回退到随机初始化。

### 2. 奖励公式修正

旧版：`R = w1 * u_t + w2 * v_t`，其中 u_t 为跨片交易占比。这会**奖励更高的跨片比例**（方向错误）。

新版：`R = w1 * (1 - u_t) + w2 * v_t`，最大化 reward 正确对应**最小化跨片交易比例**。

**影响文件**：`reward.py`

### 3. 按交易条数切分 epoch（txs-per-epoch）

新增 `--txs-per-epoch`（默认 5000），与旧的 `--blocks-per-epoch` 互斥。按交易条数切分保证每个 epoch 的样本规模更稳定（不受不同区块交易数差异影响）。

**影响文件**：`csv_stream.py`、`env_csv.py`、`train.py`、`eval_with_csv.py`

### 4. 迁移去重与改进的执行逻辑

- **`dedup_migrations()`**（`action.py`）：同一 epoch 内按 prefix 去重，只保留最后一次有效迁移；过滤无效 index 和 no-op（sender == receiver）。
- **迁移执行**（`env_csv.py`、`eval_with_csv.py`）：使用**当前真实的 prefix→shard 映射**作为源分片，而非信任策略网络预测的 sender_shard。避免因 sender 预测不准导致大面积无效迁移。
- **eval 动作历史持久化**：`eval_with_csv.py` 现在跨 epoch 维护动作历史并传递给策略网络，与训练环境语义一致。

---

## 环境与依赖

```bash
pip install -r requirements.txt   # numpy, torch（建议安装带 CUDA 的 torch 以使用 GPU）
```

---

## 训练（train.py）

### 数据来源

当前仅支持 **CSV 数据**驱动训练：用交易 CSV 驱动，每步 = 一个 epoch 的聚合状态与真实 reward。`--csv-path` 为必填参数。

> **历史**：旧版支持 `AEROEnv`（仿真环境，随机/启发式生成状态与奖励）训练分支，现已移除。

### 基本用法

```bash
# 推荐：按交易条数切分 epoch（默认 5000 条/epoch）
python train.py --csv-path 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv --total-steps 50000 --batch-size 128 --seed 0

# 兼容：按区块切分 epoch
python train.py --csv-path 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv --blocks-per-epoch 100 --total-steps 50000
```

### 训练参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--csv-path` | **（必填）** | CSV 数据路径 |
| `--txs-per-epoch` | 5000 | 每 epoch 交易条数（推荐，与 `--blocks-per-epoch` 互斥） |
| `--blocks-per-epoch` | - | 每 epoch 区块数（兼容模式，与 `--txs-per-epoch` 互斥） |
| `--total-steps` | 50000 | 总训练步数（每步 = 1 个 epoch） |
| `--batch-size` | 128 | 每批样本数，收集满一批后做一次 PPO 更新 |
| `--seed` | 0 | 随机种子 |
| `--save-dir` | checkpoints | 检查点保存目录 |
| `--save-every` | 500 | 每多少步保存一次检查点（如 aero_500.pt, aero_1000.pt） |
| `--log-interval` | 10 | 每多少个 batch 打印一次 loss/reward |
| `--device` | 自动 | 强制设备：`cuda` 或 `cpu`；不传则自动选 GPU（若有） |
| `--resume` | - | 从检查点恢复：`--resume checkpoints/aero_5000.pt` |

> **注意**：`--txs-per-epoch` 与 `--blocks-per-epoch` 不能同时指定。两者都不指定时默认使用 `--txs-per-epoch 5000`。

### 检查点与恢复

- **定期保存**：使用阈值触发机制（而非整除判断），确保不因 batch 粒度与 save-every 不对齐而遗漏保存。保存到 `checkpoints/aero_{step}.pt`。
- **训练结束**：保存 `checkpoints/aero_final.pt`。
- **中途中断**：按 **Ctrl+C** 会保存 `checkpoints/aero_interrupt_{step}.pt`，可据此恢复。
- **从检查点继续**：  
  ```bash
  python train.py --resume checkpoints/aero_500.pt --csv-path 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv --total-steps 50000
  ```
- **旧版 Gaussian checkpoint**：如果 resume 的是旧版 checkpoint（网络架构不兼容），会打印警告并自动回退到随机初始化从 step 0 开始训练。

### GPU 加速

- 若已安装带 CUDA 的 PyTorch，脚本会自动使用 GPU，并打印 `Using device: cuda` 与显卡名称。
- 强制使用 CPU：`--device cpu`。
- 仅 CPU 版 PyTorch 时只会用 CPU，需从 [pytorch.org](https://pytorch.org/get-started/locally/) 安装 CUDA 版方可使用 GPU。

---

## 用 CSV 做数据流/推理（run_with_csv.py）

不训练、仅按 epoch 流式读 CSV，聚合成 `AEROState` 并打印 reward（可选加载已训练模型）。

### 基本用法

```bash
# 默认读 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv
python run_with_csv.py
```

### 参数

| 参数 | 说明 |
|------|------|
| `--csv-path` | 指定 CSV 路径（否则用默认路径） |
| `--checkpoint` | 加载 AERO 模型（如 `checkpoints/aero_final.pt`） |
| `--save-state` | 将进度保存到文件（如 `csv_state.npz`） |
| `--save-state-every` | 每 N 个 epoch 保存一次进度（默认 10） |
| `--resume-state` | 从保存的进度文件恢复，跳过已处理 block |
| `--max-epochs` | 最多跑多少个 epoch 后停止 |

示例：边跑边保存、之后恢复：

```bash
python run_with_csv.py --save-state csv_state.npz --save-state-every 10
python run_with_csv.py --resume-state csv_state.npz
```

---

## 评估（eval_with_csv.py）

### 作用

在任意一段 `BlockTransaction.csv` 上评估 AERO 策略效果，支持两种模式：

- **`--mode nomig`**：完全不迁移，prefix→shard 映射固定，作为基线；
- **`--mode drl`**：用 AERO 模型生成迁移计划，逐 epoch 更新映射。

### 基本用法

```bash
# 不迁移基线
python eval_with_csv.py --mode nomig

# AERO 迁移
python eval_with_csv.py --mode drl

# AERO 迁移 + 确定性推理 + 自定义参数
python eval_with_csv.py --mode drl --deterministic --txs-per-epoch 10000 --plan-topk 10
```

### 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--csv-path` | `22250000to22499999_BlockTransaction/...csv` | 要评估的 CSV 路径 |
| `--checkpoint` | `checkpoints/aero_final.pt` | checkpoint 路径（不存在时自动回退到最新 `.pt`） |
| `--txs-per-epoch` | 5000 | 每 epoch 交易条数（推荐） |
| `--blocks-per-epoch` | - | 每 epoch 区块数（兼容模式） |
| `--max-epochs` | 50 | 最多评估 epoch 数 |
| `--mode` | `drl` | `drl` / `nomig` |
| `--deterministic` | 关 | 确定性推理（argmax 而非采样） |
| `--plan-topk` | 0 | 打印/落盘前 K 条迁移计划；0 = 全部 |
| `--out-csv` | 自动命名 | 结果 CSV 路径；传 `none`/`off` 关闭 |

### 输出说明

控制台输出示例：

```text
Loaded checkpoint from checkpoints/aero_final.pt
Mode=drl, num_shards=16, num_prefixes=256, txs_per_epoch=5000
[mode=drl] [epoch 0] blocks [22250000,22250099], txs=5000, R=0.123456, u_t=0.654321, v_t=-0.001234
         migrations proposed_raw=32, proposed_effective=28, applied=25
         unique_prefixes_epoch=28, cumulative_unique_prefixes=28/256
         plan_all=3->7:42;0->12:128;...
```

输出 CSV 包含列：`mode`, `epoch`, `block_start`, `block_end`, `txs`, `reward`, `u_t`, `v_t`, `migrations_proposed_raw`, `migrations_proposed_effective`, `migrations_applied`, `checkpoint`, `plan_preview`, `applied_plan_preview`, `unique_prefixes_epoch`, `cumulative_unique_prefixes`。

其中：
- `migrations_proposed_raw`：策略原始输出条数
- `migrations_proposed_effective`：去重后有效迁移条数
- `migrations_applied`：实际执行的迁移条数（当前映射与目标不同才执行）
- 输出 CSV 默认自动命名：`eval_results_drl_<checkpoint_name>.csv` 或 `eval_results_nomig.csv`

---

## 与 BlockEmulator 的对接思路

1. **状态**：由 BlockEmulator 在每 epoch 末统计各 shard 的 CST/IST、TX^c/TX^i 等，构造 `AEROState` 或等价向量，并调用 `state.to_vector()` 得到 `state_vec`。
2. **动作**：用训练好的 AERO 生成迁移计划：
   ```python
   from aero.infer import load_aero, get_migration_plan
   net, config = load_aero("checkpoints/aero_final.pt")
   plan = get_migration_plan(net, config, state, action_history=last_migrations)
   # plan: List[MigrationTransaction]，即论文式 (1) 的 migration list
   ```
3. **奖励**：在 emulator 中按同一公式计算 R_t（或由 DRL 侧用 `reward_from_state(next_state, w1, w2)` 计算），用于在线微调或仅做监控。
4. **环境替换**：在 `AEROEnvCSV.step()` 中，将 CSV 驱动替换为：把当前 `state` 和 `action` 发给 BlockEmulator，用其返回的下一状态和统计量构造 `next_state` 与 `reward`。

---

## 数据加载模块（与 AERO 低耦合）

`data/` 负责**按块/按交易条数流式读大 CSV**，产出通用结构，**不依赖 aero**；和 AERO 的耦合只在一处：`aero/data_adapter.py` 把 `RawEpochStats` 转成 `AEROState`。

- **data/schema.py**：`EpochBatch(block_start, block_end, transactions)`、`RawEpochStats(cst_per_shard, ist_per_shard, txc, txi)`，无 aero 依赖。
- **data/csv_stream.py**：`stream_epochs(csv_path, blocks_per_epoch, txs_per_epoch)`，按行流式读 CSV。支持两种切分模式：
  - `blocks_per_epoch`：按区块数切分（旧模式）；
  - `txs_per_epoch`：按交易条数切分（推荐，样本量更稳定）。
  - 两者互斥，需且只需指定一个。
  列名默认 `blockNumber` / `from` / `to`（XBlock-ETH 格式），可传参覆盖。
- **data/aggregator.py**：`aggregate_epoch(batch, prefix_to_shard, num_shards, num_prefixes, ...)`，从 `(from, to)` 列表得到 `RawEpochStats`。地址→prefix 默认取 0x 后前 8 bit。
- **aero/data_adapter.py**：`raw_stats_to_aero_state(current, num_shards, num_prefixes, prev)`，唯一从「通用统计」到「AERO 状态」的转换点。

**前提**：CSV 按 `blockNumber` 递增（或至少同一 block 内连续），否则需先按块排序再喂给 `stream_epochs`。

---

## 目录结构

```
drl_account_migration/
├── data/                    # 通用数据层，不依赖 aero
│   ├── schema.py            # EpochBatch, RawEpochStats
│   ├── csv_stream.py        # stream_epochs() 按块/按交易条数流式读 CSV
│   ├── aggregator.py        # aggregate_epoch() -> RawEpochStats
│   └── __init__.py
├── aero/
│   ├── data_adapter.py      # RawEpochStats -> AEROState（唯一与 data 的耦合点）
│   ├── config.py            # AEROConfig 超参数
│   ├── state.py             # AEROState
│   ├── action.py            # MigrationTransaction + dedup_migrations()
│   ├── reward.py            # reward_from_state() — R = w1·(1-u_t) + w2·v_t
│   ├── env.py               # 仿真环境 AEROEnv（已弃用，仅保留兼容）
│   ├── env_csv.py           # CSV 驱动环境 AEROEnvCSV（训练主力）
│   ├── network.py           # AEROAttentionPolicy + AEROPolicyValueNet（Categorical 输出）
│   ├── ppo.py               # PPO 更新（Categorical log_prob / entropy）
│   ├── infer.py             # load_aero() + get_migration_plan()（Categorical 推理）
│   └── __init__.py
├── train.py                 # 训练入口（CSV-only，支持 --resume / --save-every / --txs-per-epoch）
├── run_with_csv.py          # CSV 流式 -> 状态/reward（兼容入口）
├── eval_with_csv.py         # 评估脚本（nomig 基线 / drl 对照，输出 CSV 报告）
├── requirements.txt
├── README.md                # 快速上手指南
└── README_AERO_DRL.md       # 本文件：详细设计说明
```

---

## 注意事项

- **仿真模式**（`AEROEnv`）：已弃用。旧版支持不传 `--csv-path` 使用仿真环境训练，现在 `train.py` 要求必须提供 `--csv-path`。`aero/env.py` 仅保留兼容，不再作为训练入口。
- **CSV 模式**（`AEROEnvCSV`）：状态与奖励来自真实交易聚合，CSV 读完后会自动从头再流式读，直到跑满 `--total-steps`。
- **旧版 Gaussian checkpoint**：与新版 Categorical 网络不兼容。resume 或 eval 时会自动检测并给出警告。需使用新版代码重新训练。
- 真实效果需在接入 BlockEmulator 后，用真实交易数据与论文中的实验设置（如 16 shards、100 blocks/epoch、Ethereum 数据）复现。
