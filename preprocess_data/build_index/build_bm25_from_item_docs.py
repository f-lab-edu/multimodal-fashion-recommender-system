# scripts/build_bm25_from_item_docs.py
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List
import logging

import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="item_docs(doc_text) 기반 BM25 인덱스 생성")
    p.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="SQLite DB 경로 (예: data/fashion_items.db)",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="BM25 pickle 저장 경로 (예: data/bm25_from_item_docs.pkl)",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="앞에서 N개만 사용 (테스트용, 기본: 전체)",
    )
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    args = parse_args()
    db_path: Path = args.db_path
    out_path: Path = args.output_path

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    cur.execute(
        """
        SELECT item_id, doc_text
        FROM item_docs
        ORDER BY item_id
        """
    )
    rows = cur.fetchall()
    total = len(rows)
    logger.info("[INFO] item_docs rows = %s", total)

    if args.max_items is not None:
        rows = rows[: args.max_items]
        logger.info("[INFO] limiting to first %s docs", len(rows))

    docs_tokens: List[List[str]] = []
    item_ids: List[int] = []

    for row in tqdm(rows, desc="Loading docs"):
        item_id = int(row["item_id"])
        doc_text = row["doc_text"] or ""
        tokens = doc_text.split()  # 이미 전처리된 토큰들이라 split만 하면 됨

        docs_tokens.append(tokens)
        item_ids.append(item_id)

    conn.close()

    logger.info("[INFO] Building BM25 index ...")
    bm25 = BM25Okapi(docs_tokens)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "item_ids": item_ids,
            },
            f,
        )

    logger.info("[INFO] Saved BM25 index to %s", out_path)


if __name__ == "__main__":
    main()
