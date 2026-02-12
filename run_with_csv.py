"""
兼容入口：run_with_csv 功能已并入 train.py 的 dryrun 模式。

请直接用：
  python train.py --mode dryrun --csv-path <path_to_csv> [--max-epochs ...]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import run


if __name__ == "__main__":
    # 自动注入 --mode dryrun，保持原 run_with_csv 命令体验
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "dryrun"])
    if "--csv-path" not in sys.argv:
        sys.argv.extend(
            [
                "--csv-path",
                str(
                    ROOT
                    / "22000000to22249999_BlockTransaction"
                    / "22000000to22249999_BlockTransaction.csv"
                ),
            ]
        )
    run()
