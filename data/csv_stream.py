"""
按块流式读取大 CSV，产出 EpochBatch。
仅依赖 data.schema，不与 AERO 耦合；后续可替换为从 BlockEmulator 拉同一接口。
"""

import csv
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from .schema import EpochBatch


# XBlock-ETH 交易 CSV 列名（与 22000000to22249999_BlockTransaction.csv 一致）
COL_BLOCK = "blockNumber"
COL_FROM = "from"
COL_TO = "to"


def _row_to_pair(
    row: dict,
    from_key: str = COL_FROM,
    to_key: str = COL_TO,
) -> Optional[tuple]:
    """从一行解析 (from_addr, to_addr)，无效行返回 None。"""
    try:
        f = (row.get(from_key) or "").strip()
        t = (row.get(to_key) or "").strip()
        if f and t and f != "None" and t != "None":
            return (f, t)
    except Exception:
        pass
    return None


def stream_epochs(
    csv_path: Union[str, Path],
    blocks_per_epoch: int = 100,
    *,
    block_col: str = COL_BLOCK,
    from_col: str = COL_FROM,
    to_col: str = COL_TO,
    encoding: str = "utf-8",
) -> Iterator[EpochBatch]:
    """
    流式按 epoch 产出 (block_start, block_end, [(from, to), ...])。
    不把整个 CSV 载入内存，按行读并按 block 汇聚。

    - csv_path: 交易 CSV 路径（如 22000000to22249999_BlockTransaction.csv）
    - blocks_per_epoch: 每 epoch 含多少连续块
    - block_col / from_col / to_col: 列名，便于不同 schema 复用
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with open(path, "r", encoding=encoding, newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        if block_col not in (reader.fieldnames or []):
            raise ValueError(f"CSV 缺少列 '{block_col}'，当前列: {reader.fieldnames}")

        buf: List[Tuple[str, str]] = []
        epoch_start: Optional[int] = None
        epoch_end: Optional[int] = None

        for row in reader:
            try:
                bn = int(row.get(block_col, 0))
            except (ValueError, TypeError):
                continue
            pair = _row_to_pair(row, from_key=from_col, to_key=to_col)
            if pair is None:
                continue

            if epoch_start is None:
                # 对齐到 blocks_per_epoch 边界，便于多文件/多 run 一致
                epoch_start = (bn // blocks_per_epoch) * blocks_per_epoch
                epoch_end = epoch_start + blocks_per_epoch - 1

            if bn > epoch_end:
                # 当前 epoch 已结束，先产出
                if epoch_start is not None and buf:
                    yield EpochBatch(
                        block_start=epoch_start,
                        block_end=epoch_end,
                        transactions=buf,
                    )
                buf = []
                # 进入新 epoch
                epoch_start = (bn // blocks_per_epoch) * blocks_per_epoch
                epoch_end = epoch_start + blocks_per_epoch - 1

            buf.append(pair)

        if epoch_start is not None and buf:
            yield EpochBatch(
                block_start=epoch_start,
                block_end=epoch_end,
                transactions=buf,
            )
