# scripts/search_text_bm25_from_item_docs.py
from __future__ import annotations

import argparse
import pickle
import sqlite3
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from text_normalization import tokenize


def parse_args():
    p = argparse.ArgumentParser(description="item_docs + BM25 인덱스로 텍스트 검색")
    p.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="SQLite DB 경로 (예: data/fashion_items.db)",
    )
    p.add_argument(
        "--bm25-path",
        type=Path,
        required=True,
        help="BM25 pickle 경로 (예: data/bm25_from_item_docs.pkl)",
    )
    p.add_argument(
        "--query",
        type=str,
        required=True,
        help='검색 쿼리 (예: "black shirts")',
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="상위 몇 개를 볼지 (기본: 10)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) BM25 로드
    with args.bm25_path.open("rb") as f:
        data = pickle.load(f)
    bm25: BM25Okapi = data["bm25"]
    item_ids = data["item_ids"]
    print(f"[INFO] Loaded BM25 docs = {len(item_ids)}")

    # 2) 쿼리 토큰화 (DB에 넣을 때와 같은 규칙)
    query_tokens = tokenize(args.query)
    scores = bm25.get_scores(query_tokens)  # numpy array

    if len(scores) == 0:
        print("[INFO] No docs in BM25 index.")
        return

    top_k = min(args.top_k, len(scores))

    # 상위 top_k 인덱스 뽑기 (점수 높은 순)
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    top_item_ids = [item_ids[i] for i in top_idx]

    # 3) DB에서 메타 정보 조회
    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row

    placeholders = ",".join("?" for _ in top_item_ids)
    rows = conn.execute(
        f"""
        SELECT id, parent_asin, title, store, image_main_url
        FROM items
        WHERE id IN ({placeholders})
        """,
        top_item_ids,
    ).fetchall()
    conn.close()

    row_map = {row["id"]: row for row in rows}

    print(f"[INFO] query = {args.query!r}")
    print(f"[INFO] top-{top_k} results (BM25, 높은 점수가 더 관련도 높음):\n")

    for rank, i in enumerate(top_idx[:top_k], start=1):
        item_id = item_ids[i]
        score = scores[i]
        row = row_map.get(item_id)

        if not row:
            continue

        title = (row["title"] or "")[:80]
        store = row["store"] or ""
        img = row["image_main_url"] or ""
        asin = row["parent_asin"] or f"ITEM-{item_id}"

        print(f"#{rank}")
        print(f"  item_id : {item_id} (asin={asin})")
        print(f"  score   : {score:.4f}")
        print(f"  title   : {title}")
        print(f"  store   : {store}")
        print(f"  image   : {img}")
        print("-" * 60)


if __name__ == "__main__":
    main()
