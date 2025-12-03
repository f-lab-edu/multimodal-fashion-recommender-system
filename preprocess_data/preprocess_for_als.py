# preprocess_data/preprocess_for_als.py
# 중복제거, 리뷰가 최소 2개이상인 유저, 아이템 필터링
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import logging

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

logger = logging.getLogger(__name__)


def load_reviews_for_als(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            user = obj.get("user_id")
            parent_asin = obj.get("parent_asin") or obj.get("PARENT_ASIN")
            rating = obj.get("rating")
            ts = obj.get("timestamp")
            verified = obj.get("verified_purchase", False)
            helpful_vote = obj.get("helpful_vote", 0)

            if user is None or parent_asin is None or rating is None:
                continue

            rows.append(
                {
                    "user_id": user,
                    "item_id": parent_asin,
                    "rating": float(rating),
                    "timestamp": ts,
                    "verified_purchase": bool(verified),
                    "helpful_vote": int(helpful_vote),
                }
            )
    df = pd.DataFrame(rows)
    logger.info("[load] 행 수: %s", f"{len(df):,}")
    return df


def preprocess_for_als(
    df: pd.DataFrame,
    min_user_interactions: int = 2,
    min_item_interactions: int = 2,
) -> pd.DataFrame:
    logger.info("1. (user, item) 중복 병합")
    df_agg = df.groupby(["user_id", "item_id"], as_index=False).agg(
        rating=("rating", "mean"),
        helpful_vote=("helpful_vote", "sum"),
        verified_purchase=("verified_purchase", "max"),
        timestamp=("timestamp", "max"),
    )
    logger.info(" 병합 전 행 수: %s", f"{len(df):,}")
    logger.info(" 병합 후 행 수: %s", f"{len(df_agg):,}")

    logger.info("2. 유저/아이템별 인터랙션 수")
    user_counts = df_agg["user_id"].value_counts()
    item_counts = df_agg["item_id"].value_counts()
    logger.info("유저당 평가 수 분포:\n%s", user_counts.describe())
    logger.info("아이템당 평가 수 분포:\n%s", item_counts.describe())

    logger.info("3. 필터 기준")
    logger.info("- min_user_interactions : %d", min_user_interactions)
    logger.info("- min_item_interactions : %d", min_item_interactions)

    valid_users = user_counts[user_counts >= min_user_interactions].index
    valid_items = item_counts[item_counts >= min_item_interactions].index

    df_f = df_agg[
        df_agg["user_id"].isin(valid_users) & df_agg["item_id"].isin(valid_items)
    ].reset_index(drop=True)

    logger.info("4. 필터링 결과")
    logger.info("- 필터 후 행 수: %s", f"{len(df_f):,}")
    logger.info("- 필터 후 유저 수: %s", f"{df_f['user_id'].nunique():,}")
    logger.info("- 필터 후 아이템 수: %s", f"{df_f['item_id'].nunique():,}")

    num_rows = len(df_f)
    num_users = df_f["user_id"].nunique()
    num_items = df_f["item_id"].nunique()
    logger.info("5. 필터 후 밀도 ===")
    logger.info("- 유저당 평균 평가 수: %.2f", num_rows / num_users)
    logger.info("- 아이템당 평균 평가 수: %.2f", num_rows / num_items)

    return df_f


def save_as_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {
                "user_id": row["user_id"],
                "PARENT_ASIN": row["item_id"],
                "rating": float(row["rating"]),
                "timestamp": row["timestamp"],
                "verified_purchase": bool(row["verified_purchase"]),
                "helpful_vote": int(row["helpful_vote"]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info("[저장 완료] %s (%s rows)", out_path, f"{len(df):,}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    src_path = Path("data/Amazon_Fashion_v1.jsonl")
    out_path = Path("data/Amazon_Fashion_v2_for_als.jsonl")

    df = load_reviews_for_als(src_path)
    df_als = preprocess_for_als(
        df,
        min_user_interactions=5,
        min_item_interactions=2,
    )
    save_as_jsonl(df_als, out_path)


if __name__ == "__main__":
    main()
