## 工程简介

这个仓库实现了论文 **“AERO: Enhancing Sharding Blockchain via Deep Reinforcement Learning for Account Migration”** 中的 DRL 部分，
以及一套基于以太坊 XBlock-ETH 交易数据的评估脚本。链本身可以后续接入 BlockEmulator 或你自己的分片实现。

详细的算法设计与代码对应关系见 `README_AERO_DRL.md`，这里只讲「有哪些可执行脚本、怎么跑」。

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

### 1. `train.py` —— 训练 AERO DRL 模型

- **作用**：在一个「模拟的 sharding 环境」上，用 PPO 训练 AERO 策略网络（不依赖真实链）。  
- **适用场景**：本地调试 / 先把 DRL 流程跑通。

**用法示例：**

```bash
cd d:\project\drl_account_migration
python train.py
```

常用参数：

- `--total-steps`：总训练步数（默认 50000）
- `--batch-size`：每次 PPO 更新前收集多少步数据（默认 128）
- `--seed`：随机种子

例如先小规模试跑：

```bash
python train.py --total-steps 5000 --batch-size 64 --seed 0
```

训练过程中会打印：

- `Using device: cpu/cuda` —— 当前是否用 GPU  
- `step xxx | reward_mean=...` —— 每个 batch 的平均奖励  
- 定期保存 checkpoint 到 `checkpoints/` 下（如 `aero_6400.pt`、`aero_final.pt`）

---

### 2. `run_with_csv.py` —— 用真实 CSV 验证数据流是否通畅

- **作用**：从 `22000000to22249999_BlockTransaction.csv` 流式读取交易数据，统计每个 epoch 的 shard 级别指标，
  看 `reward / u_t / v_t` 的计算是否正常（不做策略迁移，只是「读数据 + 聚合 + 计算状态/奖励」）。
- **适用场景**：验证 XBlock-ETH 的 CSV 格式与 `data/` 模块能否正确配合，方便调试数据加载逻辑。

**用法：**

```bash
cd d:\project\drl_account_migration
python run_with_csv.py
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
```

参数说明（常用）：

- `--csv-path`：要评估的 CSV 路径，默认是  
  `22250000to22499999_BlockTransaction/22250000to22499999_BlockTransaction.csv`
- `--checkpoint`：AERO 模型 checkpoint，默认 `checkpoints/aero_final.pt`
- `--blocks-per-epoch`：每个 epoch 包含多少个区块（默认 100，与论文类似）
- `--max-epochs`：最多评估多少个 epoch（避免一次性把 12GB CSV 全扫完，默认 50）
- `--mode`：`drl` 或 `nomig`
- `--deterministic`：推理时是否用均值而不是采样

输出示例（简化）：

```text
Loaded checkpoint from checkpoints/aero_final.pt
Mode=nomig, num_shards=16, num_prefixes=256, blocks_per_epoch=100
[epoch 0] blocks [22250000,22250099], txs=..., R=..., u_t=..., v_t=...
         [nomig baseline] no migrations applied (mapping fixed)
...

Loaded checkpoint from checkpoints/aero_final.pt
Mode=drl, num_shards=16, num_prefixes=256, blocks_per_epoch=100
[epoch 0] blocks [22250000,22250099], txs=..., R=..., u_t=..., v_t=...
         migrations proposed=32, applied=xx
...
```

这里：

- `R`：当前 epoch 的奖励；
- `u_t`：跨片交易占比（越小越好）；
- `v_t`：CST 方差的负值（越接近 0 代表负载越均衡）；
- `migrations proposed / applied`：AERO 策略提出/真正生效的迁移条数。

你可以先跑 `--mode nomig` 记录一条基线曲线，再跑 `--mode drl` 看 AERO 是否在同一段数据上把 `u_t` 压低、`R` 提高。

---

## 进一步阅读

- 详细的 DRL 设计与代码对应关系：`README_AERO_DRL.md`
- 数据加载与聚合逻辑：`data/` 目录（`csv_stream.py`, `aggregator.py`, `schema.py`）
- AERO DRL 主要代码：`aero/` 目录（`state.py`, `action.py`, `reward.py`, `network.py`, `ppo.py`, `env.py`, `infer.py`）
