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

## 环境与运行

```bash
pip install -r requirements.txt   # numpy, torch
python train.py --total-steps 50000 --batch-size 128 --seed 0
```

训练脚本会使用**仿真环境**（`aero/env.py` 中的 `_step_transition`）生成下一状态与奖励。将来接入 BlockEmulator 时，只需替换为从 emulator 取 `(s', r)` 的逻辑。

### GPU 加速（可选）

- **当前检测**：本机若已安装的是 `torch` 的 **CPU 版**（如 `2.x.x+cpu`），则不会使用 GPU；训练开始时会打印 `Using device: cpu`。
- **若你有 NVIDIA 显卡**：
  1. 在终端执行 `nvidia-smi`，能正常输出显卡与驱动版本则说明驱动可用。
  2. 先卸载 CPU 版再安装带 CUDA 的 PyTorch（在 [pytorch.org](https://pytorch.org/get-started/locally/) 按系统与 CUDA 版本选命令），例如 CUDA 12.1：
     ```bash
     pip uninstall torch -y
     pip install torch --index-url https://download.pytorch.org/whl/cu121
     ```
  3. 无需改代码：`train.py` 里已用 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`，装好 CUDA 版 PyTorch 后重新运行 `python train.py`，会打印 `Using device: cuda` 并自动用 GPU。
- **若没有 NVIDIA 显卡或 `nvidia-smi` 不可用**：用 CPU 训练即可，只是每批会慢一些。

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

## 数据加载模块（与 AERO 低耦合）

`data/` 负责**按块流式读大 CSV**，产出通用结构，**不依赖 aero**；和 AERO 的耦合只在一处：`aero/data_adapter.py` 把 `RawEpochStats` 转成 `AEROState`。后续改成 BlockEmulator 时，只需让 BlockEmulator 产出同结构的 `RawEpochStats`（或在此 adapter 里改为「BlockEmulator 输出 → AEROState」）。

- **data/schema.py**：`EpochBatch(block_start, block_end, transactions)`、`RawEpochStats(cst_per_shard, ist_per_shard, txc, txi)`，无 aero 依赖。
- **data/csv_stream.py**：`stream_epochs(csv_path, blocks_per_epoch)`，按行流式读 CSV，按 block 汇聚为 `EpochBatch`。列名默认 `blockNumber` / `from` / `to`（XBlock-ETH 格式），可传参覆盖。
- **data/aggregator.py**：`aggregate_epoch(batch, prefix_to_shard, num_shards, num_prefixes, ...)`，从 `(from, to)` 列表得到 `RawEpochStats`。地址→prefix 默认取 0x 后前 8 bit。
- **aero/data_adapter.py**：`raw_stats_to_aero_state(current, num_shards, num_prefixes, prev)`，唯一从「通用统计」到「AERO 状态」的转换点。

示例：用本地交易 CSV 驱动 AERO 输入（不做训练，仅做数据流连通性检查）：

```bash
python run_with_csv.py
```

`run_with_csv.py` 会从 `22000000to22249999_BlockTransaction/22000000to22249999_BlockTransaction.csv` 流式读 epoch，做聚合后经 adapter 得到 `AEROState` 并打印 reward。**前提**：CSV 按 `blockNumber` 递增（或至少同一 block 内连续），否则需先按块排序再喂给 `stream_epochs`。

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
│   ├── config.py, state.py, action.py, reward.py, env.py, network.py, ppo.py, infer.py
│   └── __init__.py
├── train.py                 # 训练入口（当前用仿真 env）
├── run_with_csv.py          # 示例：CSV 流式 -> AERO 输入
├── requirements.txt
└── README_AERO_DRL.md
```

## 注意事项

- 当前 `AEROEnv._step_transition` 为简单仿真（在 CST/IST 上加噪声和启发式调整），仅用于验证 DRL 训练流程。
- 真实效果需在接入 BlockEmulator 后，用真实交易数据与论文中的实验设置（如 16 shards、100 blocks/epoch、Ethereum 数据）复现。
