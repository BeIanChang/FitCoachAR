"""
Merge base offline_eval_fit3d_gt.csv with HRNet2D metrics.

- base CSV:   offline_eval_fit3d_gt.csv  (mediapipe / movenet backends)
- hrnet CSV:  fit3d_results/offline_eval_fit3d_hrnet2d.csv

We simply take the union of all fieldnames across the two files and
concatenate all rows, filling missing fields with empty strings.

Usage (from repo root):

  python3 -m backend.scripts.merge_offline_eval_with_hrnet \\
    --base-csv offline_eval_fit3d_gt.csv \\
    --hrnet-csv fit3d_results/offline_eval_fit3d_hrnet2d.csv \\
    --out-csv  fit3d_results/offline_eval_fit3d_with_hrnet.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge base offline_eval CSV with HRNet2D CSV.")
    parser.add_argument(
        "--base-csv",
        type=Path,
        required=True,
        help="Path to offline_eval_fit3d_gt.csv.",
    )
    parser.add_argument(
        "--hrnet-csv",
        type=Path,
        required=True,
        help="Path to offline_eval_fit3d_hrnet2d.csv.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output merged CSV path.",
    )
    args = parser.parse_args()

    base_rows = read_rows(args.base_csv)
    hrnet_rows = read_rows(args.hrnet_csv)

    if not base_rows and not hrnet_rows:
        print("[WARN] No rows in either CSV; nothing to merge.")
        return

    # Union of all keys across all rows
    fieldnames: List[str] = []
    seen = set()

    def _accumulate_keys(rows: Iterable[Dict[str, str]]) -> None:
        nonlocal fieldnames, seen
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

    _accumulate_keys(base_rows)
    _accumulate_keys(hrnet_rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in base_rows + hrnet_rows:
            writer.writerow(r)

    print(
        f"[INFO] Merged {len(base_rows)} base rows + {len(hrnet_rows)} HRNet rows "
        f"into {args.out_csv}"
    )


if __name__ == "__main__":
    main()

