# fashion_core/multimodal_search_engine.py
# BM25 텍스트 + CLIP/FAISS 이미지 + RRF Fusion 엔진

from __future__ import annotations

from typing import Dict, List
import logging
import sqlite3

from .bm25_text_engine import BM25TextSearchEngine
from .db_image_engine import DbImageSearchEngine
from .search_results import FusionHit

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _asin_key(hit) -> str:
        """SearchHit에서 asin 키를 생성 (asin 없으면 ITEM-id 사용)."""
        return hit.asin or f"ITEM-{hit.item_id}"

    def _apply_bm25_hits(
        self,
        fused: Dict[str, FusionHit],
        bm25_hits,
    ) -> None:
        """BM25 결과를 fused 딕셔너리에 반영."""
        for h in bm25_hits:
            asin = self._asin_key(h)
            fusion = fused.get(asin)

            if fusion is None:
                fusion = FusionHit(
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
                fused[asin] = fusion
            else:
                if fusion.bm25_rank is None or h.rank < fusion.bm25_rank:
                    fusion.bm25_rank = h.rank
                    fusion.bm25_score_raw = h.score

            # bm25_rank는 위에서 반드시 설정됨
            fusion.rrf_score += self.w_text * self._rrf(fusion.bm25_rank, self.rrf_k)

    def _get_or_create_fusion_for_image_hit(
        self,
        fused: Dict[str, FusionHit],
        asin: str,
        hit,  # 타입 엄밀히 하려면: hit: SearchHit
    ) -> FusionHit:
        fusion = fused.get(asin)
        if fusion is not None:
            return fusion

        # 새로 생성
        fusion = FusionHit(
            asin=asin,
            db_item_id=hit.item_id,
            title=hit.title,
            store=hit.store,
            image_url=hit.image_url,
            bm25_rank=None,
            bm25_score_raw=None,
            image_rank=hit.rank,
            image_score_raw=hit.score,
            rrf_score=0.0,
        )
        fused[asin] = fusion
        return fusion

    @staticmethod
    def _update_fusion_image_metadata(fusion: FusionHit, hit) -> None:
        """이미지 히트 기반으로 타이틀/스토어/이미지URL/아이템ID 보강."""
        if getattr(hit, "title", None):
            fusion.title = hit.title
        if getattr(hit, "store", None):
            fusion.store = hit.store
        if getattr(hit, "image_url", None):
            fusion.image_url = hit.image_url
        if fusion.db_item_id is None:
            fusion.db_item_id = hit.item_id

    @staticmethod
    def _update_image_rank_and_score(fusion: FusionHit, hit) -> None:
        """더 좋은 이미지 랭크일 경우 image_rank / image_score_raw 갱신."""
        if fusion.image_rank is None or hit.rank < fusion.image_rank:
            fusion.image_rank = hit.rank
            fusion.image_score_raw = hit.score

    def _add_image_rrf_score(self, fusion: FusionHit) -> None:
        """image_rank가 있을 경우 RRF 스코어 반영."""
        if fusion.image_rank is not None:
            fusion.rrf_score += self.w_image * self._rrf(fusion.image_rank, self.rrf_k)

    def _apply_image_hits(
        self,
        fused: Dict[str, FusionHit],
        image_hits,
    ) -> None:
        """이미지(FAISS) 결과를 fused 딕셔너리에 반영."""
        for h in image_hits:
            asin = self._asin_key(h)
            fusion = self._get_or_create_fusion_for_image_hit(fused, asin, h)

            # 기존 fusion이든 새 fusion이든 동일 처리 가능
            self._update_fusion_image_metadata(fusion, h)
            self._update_image_rank_and_score(fusion, h)
            self._add_image_rrf_score(fusion)

    @staticmethod
    def _finalize_results(
        fused: Dict[str, FusionHit],
        top_k: int,
    ) -> List[FusionHit]:
        """RRF 스코어 기준 정렬 + 최종 rank 부여."""
        results = list(fused.values())
        results.sort(key=lambda x: x.rrf_score, reverse=True)

        final = results[:top_k]
        for i, r in enumerate(final, start=1):
            r.rank = i
        return final

    def search(
        self,
        query: str,
        top_k: int = 10,
        stage1_factor: int = 3,
        conn: sqlite3.Connection = None,
    ) -> List[FusionHit]:
        stage1_k = max(top_k, top_k * stage1_factor)

        logger.info(
            "[Fusion] query=%r, stage1_k=%d, final_top_k=%d",
            query,
            stage1_k,
            top_k,
        )

        bm25_hits = self.text_engine.search(
            query=query,
            top_k=stage1_k,
            conn=conn,
        )
        logger.info("[Fusion] BM25 hits(stage1) = %d", len(bm25_hits))

        image_hits = self.image_engine.search(
            query=query,
            top_k=stage1_k,
            conn=conn,
        )
        logger.info("[Fusion] Image hits(stage1) = %d", len(image_hits))

        fused: Dict[str, FusionHit] = {}

        self._apply_bm25_hits(fused, bm25_hits)
        self._apply_image_hits(fused, image_hits)

        return self._finalize_results(fused, top_k)
