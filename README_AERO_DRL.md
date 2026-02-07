# AERO DRL 实现说明

本目录实现论文 **AERO: Enhancing Sharding Blockchain via Deep Reinforcement Learning for Account Migration**（WWW 2025）中的 **DRL 部分**。与 BlockEmulator 的对接留作后续迁移。

## 论文对应关系

| 论文位置 | 实现 |
|---------|------|
| Section 4 MDP | `aero/state.py`（状态）、`aero/action.py`（动作）、`aero/reward.py`（奖励） |
| Section 4.1 状态 | `AEROState`：T_t, C_t, V_t, TX^c_t, TX^i_t 及上一 epoch 的 TX^c/TX^i |
| Section 4.2 动作 | `MigrationTransaction(sender_shard, receiver_shard, prefix)`，按 prefix 成组迁移 |
| Section 4.3 奖励 | R_t = w1·u_t + w2·v_t，u_t 为 CSTX 占比，v_t 为负方差 |
| Appendix C 网络 | `aero/network.py`：Query 来自 state，Key/Value 来自 action history 的 Multi-Head Attention |
| Appendix B 超参 | `aero/config.py`：γ=0.99, 6 heads, batch 128, lr 1e-5 等 |
| PPO | `aero/ppo.py`：clip + value/entropy loss，自回归序列的 log_prob |

---

## 环境与依赖

```bash
pip install -r requirements.txt   # numpy, torch（建议安装带 CUDA 的 torch 以使用 GPU）
```

---

## 训练（train.py）

### 两种数据来源

| 模式 | 说明 | 用法 |
|------|------|------|
| **仿真环境** | 不读文件，用 `AEROEnv` 随机/启发式生成状态与奖励 | 不传 `--csv-path` |
| **CSV 数据** | 用交易 CSV 驱动，每步 = 一个 epoch 的聚合状态与真实 reward | 传 `--csv-path <CSV 路径>` |

### 基本用法

**仅用仿真环境训练（默认）：**

```bash
python train.py --total-steps 50000 --batch-size 128 --seed 0
```

**用 CSV 数据训练（推荐，与论文数据一致）：**

```bash
python train.py --csv-path 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv --total-steps 50000 --batch-size 128 --seed 0
```

### 训练参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--total-steps` | 50000 | 总训练步数（每步 = 1 个 epoch） |
| `--batch-size` | 128 | 每批样本数，收集满一批后做一次 PPO 更新 |
| `--seed` | 0 | 随机种子 |
| `--save-dir` | checkpoints | 检查点保存目录 |
| `--save-every` | 2500 | 每多少步保存一次检查点（如 aero_2500.pt, aero_5000.pt） |
| `--log-interval` | 10 | 每多少个 batch 打印一次 loss/reward |
| `--device` | 自动 | 强制设备：`cuda` 或 `cpu`；不传则自动选 GPU（若有） |
| `--resume` | - | 从检查点恢复：`--resume checkpoints/aero_5000.pt` |
| `--csv-path` | - | 使用 CSV 训练时传入 CSV 路径 |
| `--blocks-per-epoch` | 100 | 使用 CSV 时，每个 epoch 包含的区块数 |

### 检查点与恢复

- **定期保存**：每 `--save-every` 步（默认 2500）保存到 `checkpoints/aero_{step}.pt`。
- **训练结束**：保存 `checkpoints/aero_final.pt`。
- **中途中断**：按 **Ctrl+C** 会保存 `checkpoints/aero_interrupt_{step}.pt`，可据此恢复。
- **从检查点继续**：  
  ```bash
  python train.py --resume checkpoints/aero_2500.pt --total-steps 50000 ...
  ```  
  用 CSV 时需同时带上 `--csv-path`，例如：  
  ```bash
  python train.py --resume checkpoints/aero_interrupt_2688.pt --csv-path 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv --total-steps 50000 --batch-size 128 --seed 0
  ```

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
4. **环境替换**：在 `AEROEnv.step()` 中，将 `_step_transition(...)` 改为：把当前 `state` 和 `action` 发给 BlockEmulator，用其返回的下一状态和统计量构造 `next_state` 与 `reward`。

---

## 数据加载模块（与 AERO 低耦合）

`data/` 负责**按块流式读大 CSV**，产出通用结构，**不依赖 aero**；和 AERO 的耦合只在一处：`aero/data_adapter.py` 把 `RawEpochStats` 转成 `AEROState`。

- **data/schema.py**：`EpochBatch(block_start, block_end, transactions)`、`RawEpochStats(cst_per_shard, ist_per_shard, txc, txi)`，无 aero 依赖。
- **data/csv_stream.py**：`stream_epochs(csv_path, blocks_per_epoch)`，按行流式读 CSV，按 block 汇聚为 `EpochBatch`。列名默认 `blockNumber` / `from` / `to`（XBlock-ETH 格式），可传参覆盖。
- **data/aggregator.py**：`aggregate_epoch(batch, prefix_to_shard, num_shards, num_prefixes, ...)`，从 `(from, to)` 列表得到 `RawEpochStats`。地址→prefix 默认取 0x 后前 8 bit。
- **aero/data_adapter.py**：`raw_stats_to_aero_state(current, num_shards, num_prefixes, prev)`，唯一从「通用统计」到「AERO 状态」的转换点。

**前提**：CSV 按 `blockNumber` 递增（或至少同一 block 内连续），否则需先按块排序再喂给 `stream_epochs`。

---

## 目录结构

```
drl_account_migration/
├── data/                    # 通用数据层，不依赖 aero
│   ├── schema.py            # EpochBatch, RawEpochStats
│   ├── csv_stream.py        # stream_epochs() 按块流式读 CSV
│   ├── aggregator.py        # aggregate_epoch() -> RawEpochStats
│   └── __init__.py
├── aero/
│   ├── data_adapter.py      # RawEpochStats -> AEROState（唯一与 data 的耦合点）
│   ├── config.py, state.py, action.py, reward.py
│   ├── env.py               # 仿真环境 AEROEnv
│   ├── env_csv.py           # CSV 驱动环境 AEROEnvCSV（用于 --csv-path 训练）
│   ├── network.py, ppo.py, infer.py
│   └── __init__.py
├── train.py                 # 训练入口（仿真或 CSV，支持 --resume / --save-every）
├── run_with_csv.py          # CSV 流式 -> 状态/reward（可选 checkpoint、保存/恢复进度）
├── requirements.txt
└── README_AERO_DRL.md
```

---

## 注意事项

- **仿真模式**（无 `--csv-path`）：`AEROEnv._step_transition` 为简单仿真（在 CST/IST 上加噪声和启发式），仅用于验证 DRL 流程。
- **CSV 模式**（`--csv-path`）：状态与奖励来自真实交易聚合，CSV 读完后会自动从头再流式读，直到跑满 `--total-steps`。
- 真实效果需在接入 BlockEmulator 后，用真实交易数据与论文中的实验设置（如 16 shards、100 blocks/epoch、Ethereum 数据）复现。
