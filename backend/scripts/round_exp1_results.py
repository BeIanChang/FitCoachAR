"""
Round experimental CSV numbers to 2 decimal places for reporting.

作用对象：fit3d_results/exp1_model/*.csv
  - fit3d_lifter_model_vs_gt.csv
  - table_hrnet2d.csv
  - table_mediapipe_2d.csv
  - table_mediapipe_3d.csv
  - table_movenet_3d.csv
  - table_romp3d.csv

策略：
  - 逐列检查，如果某个单元格可以转换为 float，就保留两位小数；
  - 否则保持原始字符串。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


def round_csv(path: Path, places: int = 2) -> None:
    if not path.exists():
        return
    with path.open() as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = list(reader)
        fieldnames = reader.fieldnames or []

    def fmt(v: str) -> str:
        if v is None or v == "":
            return v
        try:
            x = float(v)
        except Exception:
            return v
        return f"{x:.{places}f}"

    for r in rows:
        for k, v in list(r.items()):
            r[k] = fmt(v)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] rounded numeric fields in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Round exp1_model CSVs to 2 decimal places.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("fit3d_results/exp1_model"),
        help="Directory containing exp1 CSV files.",
    )
    parser.add_argument(
        "--places",
        type=int,
        default=2,
        help="Decimal places to keep for floats.",
    )
    args = parser.parse_args()

    csvs = [
        "fit3d_lifter_model_vs_gt.csv",
        "table_hrnet2d.csv",
        "table_mediapipe_2d.csv",
        "table_mediapipe_3d.csv",
        "table_movenet_3d.csv",
        "table_romp3d.csv",
    ]
    for name in csvs:
        round_csv(args.root / name, args.places)


if __name__ == "__main__":
    main()

