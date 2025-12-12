# preprocess_data/split_als_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import argparse

import pandas as pd

from common.als_io import load_als_jsonl, save_als_jsonl

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

"""
python -m preprocess_data.split_als_dataset \
  -s data/Amazon_Fashion_v2_for_als.jsonl \
  -o data/als_splits_v3 \
  --tail-k 2

"""


def split_by_user_time(
    df: pd.DataFrame,
    tail_k: int = 1,
    use_test: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - 각 user에 대해:
      * timestamp 오름차순 정렬 후
      * (use_test=True일 때만) 마지막 tail_k개 → test
      * 나머지                              → train
      * (use_test=False면) 전부 train, test는 빈 DataFrame
      ...
    """
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df_sorted = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # 테스트 분리 비활성화 옵션
    if not use_test:
        print("[SPLIT] 테스트 분리 비활성화: 전체 데이터를 train으로 사용합니다.")
        train_df = df_sorted
        test_df = df.iloc[0:0].copy()  # 빈 DataFrame
        print(f"[SPLIT] train: {len(train_df):,}, test: {len(test_df):,}")
        return train_df, test_df

    if tail_k <= 0:
        raise ValueError(f"tail_k는 1 이상이어야 합니다. (지금 값: {tail_k})")

    train_parts = []
    test_parts = []

    for user_id, g in df_sorted.groupby("user_id", sort=False):
        g = g.sort_values("timestamp")

        if len(g) <= tail_k:
            # 로그가 너무 적은 유저는 전부 train으로만 사용 (컨텍스트 확보용)
            train_parts.append(g)
            continue

        # 마지막 tail_k개를 test, 나머지를 train
        train_parts.append(g.iloc[:-tail_k])
        test_parts.append(g.iloc[-tail_k:])

    if train_parts:
        train_df = pd.concat(train_parts, ignore_index=True)
    else:
        train_df = df.iloc[0:0].copy()

    if test_parts:
        test_df = pd.concat(test_parts, ignore_index=True)
    else:
        test_df = df.iloc[0:0].copy()

    print(
        f"[SPLIT] tail_k={tail_k}, " f"train: {len(train_df):,}, test: {len(test_df):,}"
    )
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Split ALS dataset into train/test by user & time."
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="테스트 데이터를 만들지 않고, 전체를 train으로만 사용합니다.",
    )
    parser.add_argument(
        "--src-path",
        "-s",
        type=str,
        default="data/Amazon_Fashion_v2_for_als.jsonl",
        help="입력 JSONL 경로 (기본: data/Amazon_Fashion_v2_for_als.jsonl)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        required=True,
        help="train/test JSONL을 저장할 디렉터리",
    )
    parser.add_argument(
        "--tail-k",
        type=int,
        default=1,
        help="각 user별 마지막 K개를 test로 사용 (기본: 1)",
    )
    args = parser.parse_args()

    use_test = not args.no_test
    src_path = Path(args.src_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_als_jsonl(src_path)

    print(f"[ALS_IO] 로드 행 수: {len(df):,}")
    print(f"[ALS_IO] 유저 수: {df['user_id'].nunique():,}")
    print(f"[ALS_IO] 아이템 수: {df['PARENT_ASIN'].nunique():,}")

    train_df, test_df = split_by_user_time(
        df,
        tail_k=args.tail_k,
        use_test=use_test,
    )

    save_als_jsonl(train_df, out_dir / "train.jsonl")

    if use_test:
        save_als_jsonl(test_df, out_dir / "test.jsonl")
    else:
        print("[ALS_IO] --no-test 옵션 사용: test.jsonl 파일은 생성하지 않습니다.")


if __name__ == "__main__":
    main()
