# preprocess_data/split_als_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from common.als_io import load_als_jsonl, save_als_jsonl  # ✅ 공통 유틸

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


def split_by_user_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - 각 user에 대해:
      * timestamp 오름차순 정렬 후
      * 마지막 1개 → test
      * 나머지     → train
    """
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df_sorted = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    train_parts = []
    test_parts = []

    for user_id, g in df_sorted.groupby("user_id", sort=False):
        g = g.sort_values("timestamp")

        train_parts.append(g.iloc[:-1])
        test_parts.append(g.iloc[[-1]])

    if train_parts:
        train_df = pd.concat(train_parts, ignore_index=True)
    else:
        train_df = df.iloc[0:0].copy()

    if test_parts:
        test_df = pd.concat(test_parts, ignore_index=True)
    else:
        test_df = df.iloc[0:0].copy()

    print(f"[SPLIT] train: {len(train_df):,}, test: {len(test_df):,}")
    return train_df, test_df


def main():
    src_path = Path("data/Amazon_Fashion_v2_for_als.jsonl")
    out_dir = Path("data/als_splits_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_als_jsonl(src_path)

    print(f"[ALS_IO] 로드 행 수: {len(df):,}")
    print(f"[ALS_IO] 유저 수: {df['user_id'].nunique():,}")
    print(f"[ALS_IO] 아이템 수: {df['PARENT_ASIN'].nunique():,}")

    train_df, test_df = split_by_user_time(df)

    save_als_jsonl(train_df, out_dir / "train.jsonl")
    save_als_jsonl(test_df, out_dir / "test.jsonl")


if __name__ == "__main__":
    main()
