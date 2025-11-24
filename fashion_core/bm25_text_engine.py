# fashion_core/bm25_text_engine.py
# BM25 텍스트 검색 엔진

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pickle
import sqlite3

import numpy as np
from rank_bm25 import BM25Okapi

from .search_models import SearchHit, TokenizeFn


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

        # DB 연결
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # BM25 로드
        with self.bm25_path.open("rb") as f:
            data = pickle.load(f)
        self.bm25: BM25Okapi = data["bm25"]
        self.item_ids: List[int] = data["item_ids"]
        print(f"[BM25TextSearchEngine] Loaded BM25: docs={len(self.item_ids)}")

        # tokenizer
        if tokenizer is None:
            self.tokenizer = lambda s: s.lower().split()
        else:
            self.tokenizer = tokenizer

    # 내부: DB에서 items 메타 로드
    def _load_item_meta_for_ids(self, item_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        if not item_ids:
            return {}

        placeholders = ",".join("?" for _ in item_ids)
        rows = self.conn.execute(
            f"""
            SELECT
                id,
                parent_asin,
                title,
                store,
                image_main_url
            FROM items
            WHERE id IN ({placeholders})
        """,
            item_ids,
        ).fetchall()

        meta_map: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            iid = int(r["id"])
            meta_map[iid] = {
                "item_id": iid,
                "asin": r["parent_asin"],
                "title": r["title"],
                "store": r["store"],
                "image_url": r["image_main_url"],
            }
        return meta_map

    def search(self, query: str, top_k: int = 10) -> List[SearchHit]:
        tokens = self.tokenizer(query)
        print(f"[BM25TextSearchEngine] query tokens = {tokens}")

        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0:
            return []

        requested_top_k = top_k  # 최종적으로 돌려줄 개수
        raw_top_k = min(requested_top_k * 3, len(scores))  # 넉넉하게 뽑기

        top_idx = np.argpartition(scores, -raw_top_k)[-raw_top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        # item_ids 매핑
        top_item_ids = [int(self.item_ids[i]) for i in top_idx]
        meta_map = self._load_item_meta_for_ids(top_item_ids)

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

        deduped: List[SearchHit] = []
        seen_asin: set[str] = set()

        for h in hits:
            key = h.asin or f"ITEM-{h.item_id}"
            if key in seen_asin:
                continue
            seen_asin.add(key)
            deduped.append(h)
            if len(deduped) >= requested_top_k:
                break

        # deduped 안에서 rank 재부여 (1부터)
        for i, h in enumerate(deduped, start=1):
            h.rank = i

        return deduped

    def close(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None
