"""
用训练好的 AERO 模型，在任意一段 BlockTransaction CSV 上做「推理 + 统计」的评估脚本。

流程（不再用仿真环境，而是真实 CSV）：
1. 按交易数（默认 10000 条/epoch）或按块流式读 CSV（data.stream_epochs）；
2. 用当前的 prefix->shard 映射做一次聚合（data.aggregate_epoch），得到 RawEpochStats；
3. 转成 AEROState（aero.data_adapter.raw_stats_to_aero_state），算出当前 reward / CST 情况；
4. （AERO 模式）把 AEROState 丢给策略网络（aero.infer.get_migration_plan），得到本 epoch 的迁移计划；
5. （AERO 模式）更新 prefix->shard 映射（模拟 AERO 的迁移效果），再进入下一个 epoch。

注意：
- 这里的「迁移」只体现在统计上的 prefix->shard 映射变化，不会真正改动链或 BlockEmulator；
- 你可以用 22000000~22249999 训练好的 checkpoint，在 22250000~22499999 这段 CSV 上跑一遍，看
  reward / CST 走势如何；
- 也支持 **nomig 基线**：完全不迁移，只看在固定映射下的指标走势，便于和 AERO 对比。
"""

import argparse
from pathlib import Path
from typing import List, Optional
import sys
import csv

import numpy as np

# 确保项目根目录在 sys.path 中，才能以包形式导入 data / aero
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compute_u_v(state) -> tuple[float, float]:
    """
    方便打印的 u_t / v_t 计算（和 reward.py 里的定义一致）。
    - u_t: 跨片交易占比（越低越好）
    - v_t: CST 方差的负值（越接近 0 越好）
    """
    CST = np.asarray(state.CST_per_shard, dtype=np.float64)
    IST = np.asarray(state.IST_per_shard, dtype=np.float64)
    N = len(CST)
    if N == 0:
        return 0.0, 0.0
    c_t = CST.mean()
    b_t = IST.mean()
    denom = b_t + c_t
    if denom <= 0:
        u_t = 0.0
    else:
        u_t = c_t / denom
    var_cst = np.mean((CST - c_t) ** 2)
    v_t = -var_cst
    return float(u_t), float(v_t)


def _format_plan(plan, topk: int) -> str:
    if not plan:
        return ""
    if topk > 0:
        plan = plan[:topk]
    out = []
    for m in plan:
        if isinstance(m, tuple):
            s, t, p = m
            out.append(f"{s}->{t}:{p}")
        else:
            out.append(f"{m.sender_shard}->{m.receiver_shard}:{m.prefix}")
    return ";".join(out)


def _dedup_plan_by_prefix(plan, num_shards: int, num_prefixes: int):
    """封装共享去重逻辑（见 aero/action.py::dedup_migrations）。"""
    from aero.action import dedup_migrations
    return dedup_migrations(plan, num_shards, num_prefixes)


def main():
    # 延迟导入依赖 data/aero 的模块，确保上面的 sys.path 修正已生效
    from data import stream_epochs, aggregate_epoch, RawEpochStats
    from aero.infer import load_aero
    from aero.data_adapter import raw_stats_to_aero_state
    from aero.reward import reward_from_state
    from aero.config import AEROConfig

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(
            Path("22250000to22499999_BlockTransaction")
            / "22250000to22499999_BlockTransaction.csv"
        ),
        help="要评估的 BlockTransaction CSV 路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/aero_final.pt",
        help="训练好的 AERO checkpoint 路径（mode=drl 时需要；若不存在会自动回退到 checkpoints 下最新 .pt）",
    )
    epoch_group = parser.add_mutually_exclusive_group()
    epoch_group.add_argument(
        "--blocks-per-epoch",
        type=int,
        default=None,
        help="(兼容模式) 每个 epoch 包含的块数量",
    )
    epoch_group.add_argument(
        "--txs-per-epoch",
        type=int,
        default=None,
        help="每个 epoch 包含的交易数量（推荐）",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="最多评估多少个 epoch（避免一次性把整个 12GB CSV 全跑完；可自行调大）",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="推理时是否用确定性策略（直接用均值，而不是采样）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["drl", "nomig"],
        default="drl",
        help="评估模式：drl=使用 AERO 生成迁移；nomig=完全不迁移，仅作基线对比",
    )
    parser.add_argument(
        "--plan-topk",
        type=int,
        default=0,
        help="mode=drl 时每个 epoch 打印并落盘前 K 条迁移计划；<=0 表示输出完整计划",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="评估结果输出 CSV 路径；默认自动命名。传 none/off 可关闭落盘",
    )
    args = parser.parse_args()
    if args.blocks_per_epoch is None and args.txs_per_epoch is None:
        # 论文设定 100 blocks/epoch，Ethereum 每 block 约 100~200 tx，对应 ~10K~20K tx
        args.txs_per_epoch = 10000

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"[ERR] CSV 不存在: {csv_path}")
        return

    # mode=drl 时需要加载模型；mode=nomig 只用默认配置，不依赖 checkpoint
    net = None
    checkpoint_used = ""
    if args.mode == "drl":
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            ckpt_dir = ROOT / "checkpoints"
            fallback = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if fallback:
                ckpt_path = fallback[0]
                print(
                    f"[WARN] checkpoint not found: {args.checkpoint}. "
                    f"Fallback to latest checkpoint: {ckpt_path}"
                )
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found: {args.checkpoint}, and no *.pt under {ckpt_dir}"
                )
        checkpoint_used = str(ckpt_path.resolve())
        net, config = load_aero(str(ckpt_path), device="cpu")
    else:
        config = AEROConfig()
        checkpoint_used = "N/A"
    N, P = config.num_shards, config.num_prefixes

    if args.mode == "drl":
        print(f"Loaded checkpoint from {checkpoint_used}")
    else:
        print("Mode=nomig: checkpoint not required, using default AEROConfig.")
    print(
        f"Mode={args.mode}, num_shards={N}, num_prefixes={P}, "
        + (
            f"txs_per_epoch={args.txs_per_epoch}"
            if args.txs_per_epoch is not None
            else f"blocks_per_epoch={args.blocks_per_epoch}"
        )
    )

    # 初始化 prefix -> shard 映射：这里简单用均匀随机，后续每个 epoch 会被策略更新
    rng = np.random.default_rng(123)
    prefix_to_shard = rng.integers(0, N, size=P, dtype=np.int32)

    prev_stats: Optional[RawEpochStats] = None
    epoch_idx = 0
    seen_prefixes = set()
    action_history_tuples = []  # 跨 epoch 维护的动作历史，供策略网络使用
    L = config.action_history_len
    from aero.infer import get_migration_plan

    writer = None
    f_out = None
    out_csv_arg = (args.out_csv or "").strip()
    disable_out = out_csv_arg.lower() in {"none", "off", "disable", "-"}
    if not disable_out:
        if out_csv_arg:
            out_path = Path(out_csv_arg)
        else:
            if args.mode == "drl":
                ckpt_tag = Path(checkpoint_used).stem if checkpoint_used not in ("", "N/A") else "unknown_ckpt"
                out_path = Path(f"eval_results_drl_{ckpt_tag}.csv")
            else:
                out_path = Path("eval_results_nomig.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f_out = open(out_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "mode",
                "epoch",
                "block_start",
                "block_end",
                "txs",
                "reward",
                "u_t",
                "v_t",
                "migrations_proposed_raw",
                "migrations_proposed_effective",
                "migrations_applied",
                "checkpoint",
                "plan_preview",
                "applied_plan_preview",
                "unique_prefixes_epoch",
                "cumulative_unique_prefixes",
            ],
        )
        writer.writeheader()
        print(f"Result log file: {out_path.resolve()}")

    for batch in stream_epochs(
        csv_path,
        blocks_per_epoch=args.blocks_per_epoch,
        txs_per_epoch=args.txs_per_epoch,
    ):
        if epoch_idx >= args.max_epochs:
            print("(达到 max_epochs，提前结束)")
            break

        # 1) 按当前 prefix->shard 映射聚合统计
        stats = aggregate_epoch(
            batch,
            prefix_to_shard=prefix_to_shard.tolist(),
            num_shards=N,
            num_prefixes=P,
            prefix_bits=config.prefix_bits,
        )

        # 2) 转成 AEROState，计算 reward / u_t / v_t
        state = raw_stats_to_aero_state(stats, num_shards=N, num_prefixes=P, prev=prev_stats)
        R = reward_from_state(
            state,
            w1=config.w1,
            w2=config.w2,
            v_scale=config.reward_v_scale,
        )
        u_t, v_t = compute_u_v(state)

        print(
            f"[mode={args.mode}] [epoch {epoch_idx}] blocks [{batch.block_start},{batch.block_end}], "
            f"txs={len(batch.transactions)}, R={R:.6f}, u_t={u_t:.6f}, v_t={v_t:.6f}"
        )

        proposed_raw = 0
        proposed_effective = 0
        applied = 0
        plan_preview = ""
        applied_plan_preview = ""
        unique_prefixes_epoch = 0
        cumulative_unique_prefixes = len(seen_prefixes)
        if args.mode == "drl":
            # 3) 用策略网络生成本 epoch 的迁移计划，并更新 prefix->shard
            # 构建动作历史数组（从跨 epoch 维护的 tuples 列表中取最近 L 条）
            _recent = action_history_tuples[-L:]
            _ah_arr = np.zeros((L, 3), dtype=np.float32)
            _off = L - len(_recent)
            for _i, (_s, _t, _p) in enumerate(_recent):
                _ah_arr[_off + _i] = [_s, _t, _p]
            plan = get_migration_plan(
                net,
                config,
                state,
                action_history=_ah_arr,
                max_migrations=config.max_migrations_per_epoch,
                deterministic=args.deterministic,
                device="cpu",
            )
            proposed_raw = len(plan)
            plan = _dedup_plan_by_prefix(plan, N, P)
            proposed_effective = len(plan)
            epoch_prefixes = {m.prefix for m in plan}
            unique_prefixes_epoch = len(epoch_prefixes)
            seen_prefixes.update(epoch_prefixes)
            cumulative_unique_prefixes = len(seen_prefixes)
            if proposed_effective > 0:
                plan_preview = _format_plan(plan, args.plan_topk)

            applied_plan = []
            for m in plan:
                # 评估执行时采用“当前真实映射作为源分片”，避免 sender 预测误差导致大面积无效迁移
                current_src = int(prefix_to_shard[m.prefix])
                if current_src != m.receiver_shard:
                    prefix_to_shard[m.prefix] = m.receiver_shard
                    applied += 1
                    applied_plan.append((current_src, m.receiver_shard, m.prefix))

            applied_plan_preview = _format_plan(applied_plan, args.plan_topk)

            # 更新动作历史（记录实际应用的迁移，与训练环境一致）
            for t in applied_plan:
                action_history_tuples.append(t)
            action_history_tuples = action_history_tuples[-L:]

            print(
                f"         migrations proposed_raw={proposed_raw}, "
                f"proposed_effective={proposed_effective}, applied={applied}"
            )
            print(
                f"         unique_prefixes_epoch={unique_prefixes_epoch}, "
                f"cumulative_unique_prefixes={cumulative_unique_prefixes}/"
                f"{P}"
            )
            if plan_preview:
                if args.plan_topk > 0:
                    print(f"         plan_top{args.plan_topk}={plan_preview}")
                else:
                    print(f"         plan_all={plan_preview}")
            if applied_plan_preview:
                if args.plan_topk > 0:
                    print(f"         applied_plan_top{args.plan_topk}={applied_plan_preview}")
                else:
                    print(f"         applied_plan_all={applied_plan_preview}")
        else:
            # nomig：完全不迁移，prefix->shard 映射固定不变，用作基线
            print("         [nomig baseline] no migrations applied (mapping fixed)")

        if writer is not None:
            writer.writerow(
                {
                    "mode": args.mode,
                    "epoch": epoch_idx,
                    "block_start": batch.block_start,
                    "block_end": batch.block_end,
                    "txs": len(batch.transactions),
                    "reward": f"{R:.6f}",
                    "u_t": f"{u_t:.6f}",
                    "v_t": f"{v_t:.6f}",
                    "migrations_proposed_raw": proposed_raw,
                    "migrations_proposed_effective": proposed_effective,
                    "migrations_applied": applied,
                    "checkpoint": checkpoint_used,
                    "plan_preview": plan_preview,
                    "applied_plan_preview": applied_plan_preview,
                    "unique_prefixes_epoch": unique_prefixes_epoch,
                    "cumulative_unique_prefixes": cumulative_unique_prefixes,
                }
            )

        prev_stats = stats
        epoch_idx += 1

    if f_out is not None:
        f_out.close()

    print("Eval done.")


if __name__ == "__main__":
    main()

