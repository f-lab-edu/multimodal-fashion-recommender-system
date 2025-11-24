# fashion_core/multimodal_search_engine.py
# BM25 텍스트 + CLIP/FAISS 이미지 + RRF Fusion 엔진

from __future__ import annotations

from typing import Dict, List

from .bm25_text_engine import BM25TextSearchEngine
from .db_image_engine import DbImageSearchEngine
from .search_models import FusionHit


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
        stage1_k = max(top_k, top_k * stage1_factor)

        print(f"[Fusion] query={query!r}, stage1_k={stage1_k}, final_top_k={top_k}")

        bm25_hits = self.text_engine.search(query=query, top_k=stage1_k)
        print(f"[Fusion] BM25 hits(stage1) = {len(bm25_hits)}")

        image_hits = self.image_engine.search(query=query, top_k=stage1_k)
        print(f"[Fusion] Image hits(stage1) = {len(image_hits)}")

        fused: Dict[str, FusionHit] = {}

        # ----- BM25 반영 -----
        for h in bm25_hits:
            asin = h.asin or f"ITEM-{h.item_id}"
            if asin not in fused:
                fused[asin] = FusionHit(
                    asin=asin,
                    db_item_id=h.item_id,
                    title=h.title,
                    store=h.store,
                    image_url=h.image_url,
                    bm25_rank=h.rank,
                    bm25_score_raw=h.score,
                    image_rank=None,
                    image_score_raw=None,
                    rrf_score=0.0,
                )
            else:
                if fused[asin].bm25_rank is None or h.rank < fused[asin].bm25_rank:
                    fused[asin].bm25_rank = h.rank
                    fused[asin].bm25_score_raw = h.score

            fused[asin].rrf_score += self.w_text * self._rrf(
                fused[asin].bm25_rank, self.rrf_k
            )

        # ----- 이미지 반영 -----
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
                    image_rank=h.rank,
                    image_score_raw=h.score,
                    rrf_score=0.0,
                )
            else:
                if h.title:
                    fused[asin].title = h.title
                if h.store:
                    fused[asin].store = h.store
                if h.image_url:
                    fused[asin].image_url = h.image_url
                if fused[asin].db_item_id is None:
                    fused[asin].db_item_id = h.item_id

                if fused[asin].image_rank is None or h.rank < fused[asin].image_rank:
                    fused[asin].image_rank = h.rank
                    fused[asin].image_score_raw = h.score

            if fused[asin].image_rank is not None:
                fused[asin].rrf_score += self.w_image * self._rrf(
                    fused[asin].image_rank, self.rrf_k
                )

        results = list(fused.values())
        results.sort(key=lambda x: x.rrf_score, reverse=True)

        final = results[:top_k]
        for i, r in enumerate(final, start=1):
            r.rank = i

        return final
