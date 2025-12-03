# common/als_io.py

from __future__ import annotations

import json
from pathlib import Path
import logging

import pandas as pd

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

logger = logging.getLogger(__name__)


def load_als_jsonl(path: Path) -> pd.DataFrame:
    """
    Amazon_Fashion_v1_for_als.jsonl 형식을 DataFrame으로 로드.
    공통 스키마:
      - user_id
      - PARENT_ASIN
      - rating
      - timestamp
      - verified_purchase
      - helpful_vote
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            user = obj.get("user_id")
            parent_asin = obj.get("PARENT_ASIN") or obj.get("parent_asin")
            rating = obj.get("rating")
            ts = obj.get("timestamp")
            verified = obj.get("verified_purchase", False)
            helpful = obj.get("helpful_vote", 0)

            if user is None or parent_asin is None or rating is None:
                continue

            rows.append(
                {
                    "user_id": user,
                    "PARENT_ASIN": parent_asin,
                    "rating": float(rating),
                    "timestamp": ts,
                    "verified_purchase": bool(verified),
                    "helpful_vote": int(helpful),
                }
            )

    df = pd.DataFrame(rows)
    return df


def save_als_jsonl(df: pd.DataFrame, path: Path) -> None:
    """
    공통 스키마로 jsonl 저장:
      - user_id
      - PARENT_ASIN
      - rating
      - timestamp
      - verified_purchase
      - helpful_vote
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {
                "user_id": row["user_id"],
                "PARENT_ASIN": row["PARENT_ASIN"],
                "rating": float(row["rating"]),
                "timestamp": row["timestamp"],
                "verified_purchase": bool(row["verified_purchase"]),
                "helpful_vote": int(row["helpful_vote"]),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info("[ALS_IO] 저장 완료: %s (%s rows)", path, f"{len(df):,}")
