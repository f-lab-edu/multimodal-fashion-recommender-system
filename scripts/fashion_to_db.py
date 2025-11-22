# scripts/fashion_to_db.py
# 패션 데이터를 db에 저장
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm
import argparse
from scripts.text_normalization import tokenize


def pick_main_image_url(item: Dict[str, Any]) -> Optional[str]:
    """
    images 리스트에서 대표 이미지를 하나 고른다.
    - variant == "MAIN" 우선
    - 없으면 첫 번째 이미지
    - hi_res > large > thumb 순으로 선택
    """
    images = item.get("images") or []
    if not images:
        return None

    main = None
    for img in images:
        if img.get("variant") == "MAIN":
            main = img
            break
    if main is None:
        main = images[0]

    return main.get("hi_res") or main.get("large") or main.get("thumb")


def insert_one_item(conn: sqlite3.Connection, item: Dict[str, Any]) -> Optional[int]:
    """
    JSON 한 개를 받아서:
      - items 1 row
      - item_features N rows
      - item_images M rows
      - item_details K rows
      - item_docs 1 row (전처리된 문서)
    를 INSERT하고, items.id를 반환
    """
    parent_asin = item.get("parent_asin")
    main_category = item.get("main_category") or ""
    title = item.get("title") or ""
    average_rating = item.get("average_rating")
    rating_number = item.get("rating_number")
    price = item.get("price")
    store = item.get("store") or ""
    details = item.get("details") or {}
    date_first_available = details.get("Date First Available")
    image_main_url = pick_main_image_url(item)

    cur = conn.cursor()

    # 1) items 테이블에 INSERT
    cur.execute(
        """
        INSERT INTO items
            (parent_asin, main_category, title,
             average_rating, rating_number, price,
             store, date_first_available, image_main_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parent_asin,
            main_category,
            title,
            average_rating,
            rating_number,
            price,
            store,
            date_first_available,
            image_main_url,
        ),
    )
    item_id = cur.lastrowid
    if not item_id:
        return None

    # 2) item_features (+ features_text도 동시에 만들기)
    features = item.get("features") or []
    feature_strs = []
    for pos, feat in enumerate(features):
        s = str(feat)
        feature_strs.append(s)
        cur.execute(
            """
            INSERT INTO item_features (item_id, position, feature)
            VALUES (?, ?, ?)
            """,
            (item_id, pos, s),
        )
    features_text = " ".join(feature_strs)

    # 3) item_images
    images = item.get("images") or []
    for pos, img in enumerate(images):
        cur.execute(
            """
            INSERT INTO item_images
                (item_id, position, variant, thumb_url, large_url, hi_res_url)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                pos,
                img.get("variant"),
                img.get("thumb"),
                img.get("large"),
                img.get("hi_res"),
            ),
        )

    # 4) item_details
    for k, v in details.items():
        cur.execute(
            """
            INSERT INTO item_details (item_id, key, value)
            VALUES (?, ?, ?)
            """,
            (item_id, str(k), str(v)),
        )

    # 5) item_docs: 전처리된 문서 만들기
    #    main_category + title + store + features_text 를 하나의 문서로 본다.
    raw_doc_parts = [main_category, title, store, features_text]
    raw_doc = " . ".join(p for p in raw_doc_parts if p)

    tokens = tokenize(raw_doc)  # 셔츠/티셔츠 등 규칙 + stopword 제거 포함
    doc_text = " ".join(tokens)  # 나중에 BM25에서 split()만 하면 되게

    cur.execute(
        """
        INSERT INTO item_docs (item_id, doc_text)
        VALUES (?, ?)
        """,
        (item_id, doc_text),
    )

    return item_id


def parse_args():
    p = argparse.ArgumentParser(
        description="Amazon Fashion JSONL → SQLite DB(items, item_features, item_images, item_details, item_docs) 적재"
    )
    p.add_argument(
        "--jsonl-path",
        type=Path,
        required=True,
        help="meta_Amazon_Fashion_v1.jsonl 경로",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="SQLite DB 경로 (예: data/fashion_items.db)",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="앞에서 N개만 로딩 (테스트용, 기본: 전체)",
    )
    p.add_argument(
        "--commit-interval",
        type=int,
        default=1000,
        help="N개마다 commit (기본: 1000)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    jsonl_path: Path = args.jsonl_path
    db_path: Path = args.db_path

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    inserted = 0
    total_lines = 0
    nonempty_lines = 0

    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(
                tqdm(f, desc=f"[import] reading {jsonl_path.name}")
            ):
                total_lines += 1

                line = line.strip()
                if not line:
                    # 빈 줄은 JSON 아이템으로 치지 않음
                    continue

                nonempty_lines += 1
                item = json.loads(line)

                item_id = insert_one_item(conn, item)
                if item_id is None:
                    # INSERT 실패한 경우는 카운트에서 제외
                    continue

                inserted += 1

                if args.max_items is not None and inserted >= args.max_items:
                    break

                if inserted % args.commit_interval == 0:
                    conn.commit()
                    print(f"[INFO] committed {inserted} items")

            conn.commit()
    finally:
        conn.close()

    print(f"[INFO] Done. Inserted {inserted} items into DB: {db_path}")
    print(f"[INFO] Total lines (raw):        {total_lines}")
    print(f"[INFO] Non-empty JSON lines:     {nonempty_lines}")
    print(f"[INFO] items in DB (by script):  {inserted}")


if __name__ == "__main__":
    main()
