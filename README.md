## 工程简介

这个仓库实现了论文 **"AERO: Enhancing Sharding Blockchain via Deep Reinforcement Learning for Account Migration"** 中的 DRL 部分，
以及一套基于以太坊 XBlock-ETH 交易数据的评估脚本。链本身可以后续接入 BlockEmulator 或你自己的分片实现。

详细的算法设计与代码对应关系见 `README_AERO_DRL.md`，这里只讲「有哪些可执行脚本、怎么跑」。

> **最近重大更新**：
> - 策略网络从连续 Gaussian 分布切换为 **离散 Categorical 分布**（解决 prefix 坍缩问题）；
> - 奖励公式修正：`w1 * u_t` → `w1 * (1 - u_t)`，正确激励降低跨片交易比例；
> - 新增 **按交易条数切分 epoch**（`--txs-per-epoch`），替代旧的按块切分，每 epoch 样本量更稳定；
> - 迁移计划增加 **去重逻辑**（同 epoch 内每个 prefix 只保留最后一次有效迁移）；
> - `train.py` 中 `--csv-path` 变为**必填**参数，不再支持仿真环境训练分支。

---

## 环境准备

在项目根目录下（`d:\project\drl_account_migration`）：

```bash
pip install -r requirements.txt
```

> 说明：  
> - 目前依赖只有 `numpy` 和 `torch`；  
> - 若有 NVIDIA 显卡并想用 GPU，可按 `README_AERO_DRL.md` 里的「GPU 加速」说明装带 CUDA 的 PyTorch。

---

## 可执行的 Python 脚本一览

### 1. `train.py` —— 统一训练入口（CSV-only）

- **作用**：基于真实 CSV 数据训练 AERO 策略网络（PPO），不再支持仿真环境。
  - `--mode train`：用 PPO 训练 AERO；
  - `--mode dryrun`：只做 CSV 流式读取 + 聚合 + 奖励计算（不训练）。
- **适用场景**：训练与数据流检查都走同一个入口，减少脚本分叉。

**用法示例：**

```bash
cd d:\project\drl_account_migration

# 使用默认按交易数切分（每 epoch 5000 条交易）
python train.py --mode train --csv-path 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv

# 使用按块切分（兼容模式）
python train.py --mode train --csv-path 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv --blocks-per-epoch 100
```

常用参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mode` | `train` | `train` / `dryrun` |
| `--csv-path` | **（必填）** | 训练/检查使用的 BlockTransaction CSV 路径 |
| `--txs-per-epoch` | 5000 | 每个 epoch 包含的交易条数（推荐，与 `--blocks-per-epoch` 互斥） |
| `--blocks-per-epoch` | - | 每个 epoch 包含的区块数（兼容模式，与 `--txs-per-epoch` 互斥） |
| `--total-steps` | 50000 | 总训练步数（仅 `mode=train`） |
| `--batch-size` | 128 | 每次 PPO 更新前收集多少步数据 |
| `--save-every` | 500 | 每多少步保存一次 checkpoint |
| `--seed` | 0 | 随机种子 |
| `--device` | 自动 | 强制设备：`cuda` / `cpu`；不传则自动选 GPU |
| `--resume` | - | 从检查点恢复，如 `--resume checkpoints/aero_5000.pt` |

> **注意**：`--txs-per-epoch` 与 `--blocks-per-epoch` 不能同时指定。两者都不指定时默认使用 `--txs-per-epoch 5000`。

例如先小规模试跑：

```bash
python train.py --mode train --csv-path 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv --total-steps 5000 --batch-size 64 --seed 0
```

训练过程中会打印：

- `Using device: cpu/cuda` —— 当前是否用 GPU  
- `step xxx | reward_mean=...` —— 每个 batch 的平均奖励  
- 定期保存 checkpoint 到 `checkpoints/` 下（如 `aero_500.pt`、`aero_1000.pt`、`aero_final.pt`）

---

### 2. `run_with_csv.py` —— 用真实 CSV 验证数据流是否通畅

- **作用**：兼容入口（内部转发到 `train.py --mode dryrun`）。  
- **适用场景**：保留旧命令习惯；新用法建议直接用 `train.py --mode dryrun`。

**用法：**

```bash
cd d:\project\drl_account_migration
python run_with_csv.py
```

等价的新写法：

```bash
python train.py --mode dryrun --csv-path 22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv --max-epochs 50
```

运行时会输出每个 epoch 的：

- 区块范围 `blocks [start, end]`
- 交易数 `txs=...`
- 奖励 `R=...`

---

### 3. `eval_with_csv.py` —— 在指定 CSV 上评估 AERO（含「不迁移」基线）

- **作用**：在任意一段 `BlockTransaction.csv` 上，用两种模式评估 AERO：
  1. **`--mode nomig`**：完全不迁移，prefix→shard 映射固定，只看原始分布下的指标（**基线**）；  
  2. **`--mode drl`**：用训练好的 AERO 模型生成迁移计划，逐 epoch 更新 prefix→shard，再看指标变化（**对照组**）。
- **适用场景**：离线评估 AERO 在真实数据区间上的表现，比如在 22250000~22499999 这段区间上看 AERO 是否降低了跨片交易比例。

**典型用法（对 22250000~22499999 区间评估）：**

1. 先确保已有训练好的模型，例如 `checkpoints/aero_final.pt`（由 `train.py` 训练得到）。
2. 在项目根目录执行：

```bash
cd d:\project\drl_account_migration

# 1) 不迁移基线（prefix→shard 固定）
python eval_with_csv.py --mode nomig

# 2) AERO 迁移（用 DRL 生成迁移计划）
python eval_with_csv.py --mode drl

# 3) AERO 迁移 + 确定性推理 + 自定义 epoch 大小
python eval_with_csv.py --mode drl --deterministic --txs-per-epoch 10000

# 4) 使用按块切分（兼容模式）
python eval_with_csv.py --mode drl --blocks-per-epoch 100
```

参数说明（常用）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--csv-path` | `22250000to22499999_BlockTransaction/...csv` | 要评估的 CSV 路径 |
| `--checkpoint` | `checkpoints/aero_final.pt` | AERO 模型 checkpoint（不存在时自动回退到 `checkpoints/` 下最新 `.pt`） |
| `--txs-per-epoch` | 5000 | 每个 epoch 包含的交易条数（推荐，与 `--blocks-per-epoch` 互斥） |
| `--blocks-per-epoch` | - | 每个 epoch 包含的区块数（兼容模式） |
| `--max-epochs` | 50 | 最多评估多少个 epoch |
| `--mode` | `drl` | `drl` 或 `nomig` |
| `--deterministic` | 关 | 推理时是否用确定性策略（argmax 而非采样） |
| `--plan-topk` | 0 | 每 epoch 打印/落盘前 K 条迁移计划；0 = 输出完整计划 |
| `--out-csv` | 自动命名 | 结果输出 CSV 路径；传 `none`/`off` 可关闭落盘 |

> **注意**：`--txs-per-epoch` 与 `--blocks-per-epoch` 不能同时指定。两者都不指定时默认使用 `--txs-per-epoch 5000`。

输出示例（简化）：

```text
Loaded checkpoint from checkpoints/aero_final.pt
Mode=drl, num_shards=16, num_prefixes=256, txs_per_epoch=5000
[mode=drl] [epoch 0] blocks [22250000,22250099], txs=5000, R=0.123456, u_t=0.654321, v_t=-0.001234
         migrations proposed_raw=32, proposed_effective=28, applied=25
         unique_prefixes_epoch=28, cumulative_unique_prefixes=28/256
         plan_all=3->7:42;0->12:128;...
         applied_plan_all=3->7:42;0->12:128;...
...
```

输出 CSV 包含以下列：

| 列名 | 说明 |
|------|------|
| `mode` | 评估模式 (`drl` / `nomig`) |
| `epoch` | Epoch 序号 |
| `block_start` / `block_end` | 本 epoch 覆盖的区块范围 |
| `txs` | 本 epoch 交易条数 |
| `reward` | 当前 epoch 的奖励 R |
| `u_t` | 跨片交易占比（越小越好） |
| `v_t` | CST 方差的负值（越接近 0 代表负载越均衡） |
| `migrations_proposed_raw` | 策略网络提出的原始迁移条数 |
| `migrations_proposed_effective` | 去重后有效迁移条数 |
| `migrations_applied` | 实际执行的迁移条数 |
| `checkpoint` | 使用的 checkpoint 路径 |
| `plan_preview` / `applied_plan_preview` | 迁移计划预览 |
| `unique_prefixes_epoch` | 本 epoch 涉及的不同 prefix 数 |
| `cumulative_unique_prefixes` | 截至当前 epoch 累计涉及的不同 prefix 数 |

你可以先跑 `--mode nomig` 记录一条基线曲线，再跑 `--mode drl` 看 AERO 是否在同一段数据上把 `u_t` 压低、`R` 提高。

---

## 进一步阅读

- 详细的 DRL 设计与代码对应关系：`README_AERO_DRL.md`
- 数据加载与聚合逻辑：`data/` 目录（`csv_stream.py`, `aggregator.py`, `schema.py`）
- AERO DRL 主要代码：`aero/` 目录（`state.py`, `action.py`, `reward.py`, `network.py`, `ppo.py`, `env_csv.py`, `infer.py`）
