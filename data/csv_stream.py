import csv
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from .schema import EpochBatch


COL_BLOCK = "blockNumber"
COL_FROM = "from"
COL_TO = "to"


def _row_to_pair(row: dict, from_key: str = COL_FROM, to_key: str = COL_TO) -> Optional[Tuple[str, str]]:
    """从一行中提取 (from_addr, to_addr)。"""
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
    blocks_per_epoch: Optional[int] = 100,
    txs_per_epoch: Optional[int] = None,
    *,
    block_col: str = COL_BLOCK,
    from_col: str = COL_FROM,
    to_col: str = COL_TO,
    encoding: str = "utf-8",
) -> Iterator[EpochBatch]:
    """
    按行流式读取 CSV，并切分为 epoch 批次。

    支持两种切分方式（二选一）：
    1) blocks_per_epoch: 按区块窗口切分（原有行为）
    2) txs_per_epoch: 按固定交易条数切分（推荐，epoch 样本量更稳定）
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if txs_per_epoch is not None and blocks_per_epoch is not None:
        raise ValueError("blocks_per_epoch 和 txs_per_epoch 只能二选一")
    if txs_per_epoch is None and blocks_per_epoch is None:
        raise ValueError("blocks_per_epoch 和 txs_per_epoch 不能同时为空")
    if txs_per_epoch is not None and txs_per_epoch <= 0:
        raise ValueError("txs_per_epoch 必须 > 0")
    if blocks_per_epoch is not None and blocks_per_epoch <= 0:
        raise ValueError("blocks_per_epoch 必须 > 0")

    with open(path, "r", encoding=encoding, newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        if block_col not in (reader.fieldnames or []):
            raise ValueError(f"CSV 缺少列 '{block_col}'，当前列: {reader.fieldnames}")

        buf: List[Tuple[str, str]] = []
        epoch_start: Optional[int] = None  # tx 模式下表示本 epoch 首条交易所在区块
        epoch_end: Optional[int] = None    # tx 模式下表示本 epoch 最后一条交易所在区块

        for row in reader:
            try:
                bn = int(row.get(block_col, 0))
            except (ValueError, TypeError):
                continue

            pair = _row_to_pair(row, from_key=from_col, to_key=to_col)
            if pair is None:
                continue

            if txs_per_epoch is not None:
                if epoch_start is None:
                    epoch_start = bn
                epoch_end = bn
                buf.append(pair)
                if len(buf) >= txs_per_epoch:
                    yield EpochBatch(epoch_start, epoch_end, buf)
                    buf = []
                    epoch_start = None
                    epoch_end = None
            else:
                if epoch_start is None:
                    epoch_start = (bn // blocks_per_epoch) * blocks_per_epoch
                    epoch_end = epoch_start + blocks_per_epoch - 1

                if bn > epoch_end:
                    if epoch_start is not None and buf:
                        yield EpochBatch(epoch_start, epoch_end, buf)
                    buf = []
                    epoch_start = (bn // blocks_per_epoch) * blocks_per_epoch
                    epoch_end = epoch_start + blocks_per_epoch - 1

                buf.append(pair)

        if epoch_start is not None and buf and epoch_end is not None:
            yield EpochBatch(epoch_start, epoch_end, buf)

