# app/search/service.py
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any, List, Dict

import numpy as np
import pandas as pd

from app.search.clip_text import ClipTextEncoder
from app.search.filters import filter_by_slots
from app.search.load_resources import ResourceLoader


class FashionSearchService:
    def __init__(
        self,
        catalog_path: str | Path,
    ):
        self.encoder = ClipTextEncoder()
        self.loader = ResourceLoader(catalog_path)

    def build_item_text(self, row: pd.Series) -> str:
        """
        Fashion-CLIP에 넣을 아이템 설명 문자열 생성.
        """
        parts: list[str] = []
        for col in [
            "title",
            "brand",
            "features_text",
            "description_text",
            "categories_text",
        ]:
            val = row.get(col)
            if isinstance(val, str) and val:
                parts.append(val)
        return " ".join(parts)

    def rank_with_clip(
        self,
        candidates: pd.DataFrame,
        query_for_embed: str,
        batch_size: int = 64,
    ) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        # 쿼리 임베딩
        q_vec = self.encoder.encode(query_for_embed)[0]

        # 아이템 임베딩
        texts = candidates.apply(self.build_item_text, axis=1).tolist()

        vec_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_vecs = self.encoder.encode(batch_texts)
            vec_list.append(batch_vecs)

        item_vecs = np.vstack(vec_list)

        scores = item_vecs @ q_vec  # (N,)

        ranked = candidates.copy()
        ranked["clip_score"] = scores
        return ranked.sort_values("clip_score", ascending=False)

    def search_fashion_items(
        self,
        slots: Mapping[str, Any],
        query_for_embed: str,
        top_k: int = 30,
        min_candidates: int = 30,
        max_candidates_for_clip: int = 2000,
        batch_size_for_clip: int = 64,
    ) -> List[Dict[str, Any]]:
        """
        입력: slots, query_for_embed
        출력: 추천 아이템 리스트
        """
        if self.loader is None:
            raise RuntimeError("ResourceLoader가 설정되지 않았습니다.")

        catalog = self.loader.get_catalog_df()

        # 1차: 슬롯 기반 필터
        candidates = filter_by_slots(catalog, slots)

        # 후보가 적으면 전체 카탈로그로 보완
        if len(candidates) < min_candidates:
            candidates = catalog

        if len(candidates) > max_candidates_for_clip:
            candidates = candidates.sample(
                n=max_candidates_for_clip,
                random_state=42,
            )

        # 2차: Fashion-CLIP 랭킹
        ranked = self.rank_with_clip(
            candidates,
            query_for_embed,
            batch_size=batch_size_for_clip,
        )

        # 리턴 컬럼 추리기
        cols = [
            col
            for col in [
                "item_id",
                "title",
                "brand",
                "image_url",
                "features_text",
                "description_text",
                "clip_score",
            ]
            if col in ranked.columns
        ]
        return ranked.head(top_k)[cols].to_dict(orient="records")
