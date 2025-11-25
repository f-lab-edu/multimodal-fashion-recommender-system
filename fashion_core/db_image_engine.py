# fashion_core/db_image_search_engine.py
# CLIP + FAISS + DB 기반 이미지 검색 엔진

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import sqlite3

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from .search_results import SearchHit, TokenizeFn
from .search_utils import load_item_meta_for_ids, deduplicate_hits_by_asin


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

    def search(self, query: str, top_k: int = 10) -> List[SearchHit]:
        q_emb = self._encode_text_query(query)  # (1, d)

        raw_top_k = min(top_k * 3, self.index.ntotal)
        scores, ids = self.index.search(q_emb, raw_top_k)
        scores = scores[0]
        ids = ids[0]

        # -1 필터
        valid = [(int(i), float(s)) for i, s in zip(ids, scores) if i >= 0]
        if not valid:
            return []

        item_ids = [i for i, _ in valid]
        meta_map = load_item_meta_for_ids(self.conn, item_ids)

        hits: List[SearchHit] = []
        for rank_raw, (item_id, score) in enumerate(valid, start=1):
            meta = meta_map.get(item_id, {})
            hits.append(
                SearchHit(
                    item_id=item_id,
                    asin=meta.get("asin"),
                    score=score,
                    rank=rank_raw,  # raw rank
                    title=meta.get("title"),
                    store=meta.get("store"),
                    image_url=meta.get("image_url"),
                )
            )

        return deduplicate_hits_by_asin(hits, top_k)

    def close(self):
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None
