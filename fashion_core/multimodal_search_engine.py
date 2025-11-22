# fashion_core/multimodal_search_engine.py
# BM25 텍스트 검색 + CLIP/FAISS 이미지 검색 + RRF Fusion 엔진

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pickle
import sqlite3

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import CLIPModel, CLIPProcessor

TokenizeFn = Callable[[str], Sequence[str]]


@dataclass
class SearchHit:
    item_id: int
    asin: Optional[str]
    score: float
    rank: int
    title: Optional[str] = None
    store: Optional[str] = None
    image_url: Optional[str] = None  # DB에 저장된 대표 이미지(or 로컬 경로)


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

        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
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

        return hits

    def close(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None


# CLIP + FAISS + DB 이미지 엔진
class DbImageSearchEngine:
    """
    - FAISS 인덱스 + DB(items) 를 이용한 이미지 검색 엔진
    """

    def __init__(
        self,
        db_path: Path,
        index_path: Path,
        model_name: str = "patrickjohncyh/fashion-clip",
        device: Optional[str] = None,
        tokenizer: Optional[TokenizeFn] = None,
    ):
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        self.model_name = model_name

        if not self.db_path.exists():
            raise FileNotFoundError(f"DB not found: {self.db_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        # DB
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # FAISS
        self.index = faiss.read_index(str(self.index_path))
        print(f"[DbImageSearchEngine] Loaded FAISS index: ntotal={self.index.ntotal}")

        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # CLIP
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        print(
            f"[DbImageSearchEngine] Using device={self.device}, model={self.model_name}"
        )

        # tokenizer
        if tokenizer is None:
            self.tokenizer = lambda s: s.lower().split()
        else:
            self.tokenizer = tokenizer

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
        return x / norms

    def _encode_text_query(self, query: str) -> np.ndarray:
        tokens = self.tokenizer(query)
        norm_query = " ".join(tokens)
        print(f"[DbImageSearchEngine] norm_query = {norm_query!r}")

        inputs = self.processor(
            text=[norm_query],
            images=None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.get_text_features(**inputs)

        emb = out.cpu().numpy().astype("float32")
        emb = self._normalize(emb)
        return emb  # shape: (1, d)

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
        q_emb = self._encode_text_query(query)  # (1, d)

        scores, ids = self.index.search(q_emb, top_k)
        scores = scores[0]
        ids = ids[0]

        # -1 필터
        valid = [(int(i), float(s)) for i, s in zip(ids, scores) if i >= 0]
        if not valid:
            return []

        item_ids = [i for i, _ in valid]
        meta_map = self._load_item_meta_for_ids(item_ids)

        hits: List[SearchHit] = []
        for rank, (item_id, score) in enumerate(valid, start=1):
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

        return hits

    def close(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None


# BM25 + CLIP Fusion 엔진 (RRF)
@dataclass
class FusionHit:
    asin: str
    db_item_id: Optional[int]
    title: Optional[str]
    store: Optional[str]
    image_url: Optional[str]
    bm25_rank: Optional[int]
    bm25_score_raw: Optional[float]
    image_rank: Optional[int]
    image_score_raw: Optional[float]
    rrf_score: float
    rank: Optional[int] = None  # 최종 순위


class BM25ClipFusionEngine:
    """
    BM25 텍스트 엔진 + DbImageSearchEngine 를 조합해서
    asin 기준으로 RRF(rank fusion) 후 최종 top-k 결과를 반환
    """

    def __init__(
        self,
        text_engine: BM25TextSearchEngine,
        image_engine: DbImageSearchEngine,
        rrf_k: int = 60,
        w_text: float = 1.0,
        w_image: float = 1.0,
    ):
        self.text_engine = text_engine
        self.image_engine = image_engine
        self.rrf_k = rrf_k
        self.w_text = w_text
        self.w_image = w_image

    @staticmethod
    def _rrf(rank: int, k: int) -> float:
        return 1.0 / (k + rank)

    def search(
        self,
        query: str,
        top_k: int = 10,
        stage1_factor: int = 3,
    ) -> List[FusionHit]:
        """
        1) BM25에서 top_k * stage1_factor 만큼 검색
        2) 이미지에서 top_k * stage1_factor 만큼 검색
        3) asin 기준으로 RRF 점수 합산 후 글로벌 top_k 반환
        """
        stage1_k = max(top_k, top_k * stage1_factor)

        print(f"[Fusion] query={query!r}, stage1_k={stage1_k}, final_top_k={top_k}")

        # 1) 텍스트(BM25) 검색
        bm25_hits = self.text_engine.search(query=query, top_k=stage1_k)
        print(f"[Fusion] BM25 hits(stage1) = {len(bm25_hits)}")

        # 2) 이미지 검색
        image_hits = self.image_engine.search(query=query, top_k=stage1_k)
        print(f"[Fusion] Image hits(stage1) = {len(image_hits)}")

        fused: Dict[str, FusionHit] = {}

        # ---------- BM25 반영 ----------
        for h in bm25_hits:
            asin = h.asin or f"ITEM-{h.item_id}"
            if asin not in fused:
                fused[asin] = FusionHit(
                    asin=asin,
                    db_item_id=h.item_id,
                    title=h.title,
                    store=h.store,
                    image_url=h.image_url,
                    bm25_rank=None,
                    bm25_score_raw=None,
                    image_rank=None,
                    image_score_raw=None,
                    rrf_score=0.0,
                )

            fused[asin].bm25_rank = h.rank
            fused[asin].bm25_score_raw = h.score
            fused[asin].rrf_score += self.w_text * self._rrf(h.rank, self.rrf_k)

        # ---------- 이미지 반영 ----------
        for h in image_hits:
            asin = h.asin or f"ITEM-{h.item_id}"
            if asin not in fused:
                fused[asin] = FusionHit(
                    asin=asin,
                    db_item_id=h.item_id,
                    title=h.title,
                    store=h.store,
                    image_url=h.image_url,
                    bm25_rank=None,
                    bm25_score_raw=None,
                    image_rank=None,
                    image_score_raw=None,
                    rrf_score=0.0,
                )
            else:
                # 이미 BM25에서 들어온 asin인데, 이미지 쪽 메타가 더 풍부하면 덮어쓸 수도 있음
                if h.title:
                    fused[asin].title = h.title
                if h.store:
                    fused[asin].store = h.store
                if h.image_url:
                    fused[asin].image_url = h.image_url
                if fused[asin].db_item_id is None:
                    fused[asin].db_item_id = h.item_id

            fused[asin].image_rank = h.rank
            fused[asin].image_score_raw = h.score
            fused[asin].rrf_score += self.w_image * self._rrf(h.rank, self.rrf_k)

        # ---------- 정렬 & 최종 top-k ----------
        results = list(fused.values())
        results.sort(key=lambda x: x.rrf_score, reverse=True)

        final = results[:top_k]
        for i, r in enumerate(final, start=1):
            r.rank = i

        return final
