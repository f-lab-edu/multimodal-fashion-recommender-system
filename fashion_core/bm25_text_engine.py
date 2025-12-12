# fashion_core/bm25_text_engine.py
# BM25 텍스트 검색 엔진

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import logging
import pickle
import sqlite3

import numpy as np
from rank_bm25 import BM25Okapi

from .search_results import SearchHit, TokenizeFn
from .search_utils import load_item_meta_for_ids, deduplicate_hits_by_asin

logger = logging.getLogger(__name__)


# BM25 텍스트 엔진
class BM25TextSearchEngine:
    """
    - precomputed BM25Okapi + item_docs 기반 텍스트 검색 엔진
    - BM25 pickle 안에는 {"bm25": BM25Okapi, "item_ids": List[int]} 구조가 들어있다고 가정
      * item_ids[i] == items.id (DB의 PK)
    """

    def __init__(
        self,
        db_path: Path,
        bm25_path: Path,
        tokenizer: Optional[TokenizeFn] = None,
    ):
        self.db_path = Path(db_path)
        self.bm25_path = Path(bm25_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"DB not found: {self.db_path}")
        if not self.bm25_path.exists():
            raise FileNotFoundError(f"BM25 file not found: {self.bm25_path}")

        # BM25 로드
        with self.bm25_path.open("rb") as f:
            data = pickle.load(f)
        self.bm25: BM25Okapi = data["bm25"]
        self.item_ids: List[int] = data["item_ids"]

        logger.info(
            "[BM25TextSearchEngine] Loaded BM25: docs=%d",
            len(self.item_ids),
        )

        # tokenizer
        if tokenizer is None:
            self.tokenizer = lambda s: s.lower().split()
        else:
            self.tokenizer = tokenizer

    def search(
        self,
        query: str,
        conn: sqlite3.Connection,
        top_k: int = 10,
    ) -> List[SearchHit]:
        tokens = self.tokenizer(query)
        logger.info("[BM25TextSearchEngine] query tokens = %s", tokens)

        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0:
            return []

        requested_top_k = top_k  # 최종적으로 돌려줄 개수
        raw_top_k = min(requested_top_k * 3, len(scores))  # 넉넉하게 뽑기

        top_idx = np.argpartition(scores, -raw_top_k)[-raw_top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        # item_ids 매핑
        top_item_ids = [int(self.item_ids[i]) for i in top_idx]
        meta_map = load_item_meta_for_ids(conn, top_item_ids)

        hits: List[SearchHit] = []
        for rank, idx in enumerate(top_idx, start=1):
            item_id = int(self.item_ids[idx])
            score = float(scores[idx])
            meta = meta_map.get(item_id, {})
            hit = SearchHit(
                item_id=item_id,
                asin=meta.get("asin"),
                score=score,
                rank=rank,
                title=meta.get("title"),
                store=meta.get("store"),
                image_url=meta.get("image_url"),
            )
            hits.append(hit)

        return deduplicate_hits_by_asin(hits, requested_top_k)

    def close(self) -> None:
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None
