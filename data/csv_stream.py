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
    blocks_per_epoch: int = 100,
    *,
    block_col: str = COL_BLOCK,
    from_col: str = COL_FROM,
    to_col: str = COL_TO,
    encoding: str = "utf-8",
) -> Iterator[EpochBatch]:
    """
    按行流式读取 CSV，并按 block 区间聚合为 epoch 批次。
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
                epoch_start = (bn // blocks_per_epoch) * blocks_per_epoch
                epoch_end = epoch_start + blocks_per_epoch - 1

            if bn > epoch_end:
                if epoch_start is not None and buf:
                    yield EpochBatch(epoch_start, epoch_end, buf)
                buf = []
                epoch_start = (bn // blocks_per_epoch) * blocks_per_epoch
                epoch_end = epoch_start + blocks_per_epoch - 1

            buf.append(pair)

        if epoch_start is not None and buf:
            yield EpochBatch(epoch_start, epoch_end, buf)

