# fashion_core/als_rerank_multimodal_engine.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pathlib import Path

from fashion_core.multimodal_search_engine import MultiModalSearchEngine
from recommend.als_reranker import ALSReRanker

logger = logging.getLogger(__name__)


class PersonalizedSearchEngine:
    """
    MultiModalSearchEngine + ALSReRanker 를 결합한 개인맞춤 검색 엔진
    """

    def __init__(
        self,
        multimodal_engine: MultiModalSearchEngine,
        als_reranker: ALSReRanker,
    ) -> None:
        self.multimodal_engine = multimodal_engine
        self.als_reranker = als_reranker

    @classmethod
    def from_paths(
        cls,
        db_path: Path,
        bm25_path: Path,
        image_index_path: Path,
        als_config_path: Path,
        model_name: str = "patrickjohncyh/fashion-clip",
        device: Optional[str] = None,
        rrf_k: int = 60,
        w_text: float = 1.0,
        w_image: float = 1.0,
    ) -> "PersonalizedSearchEngine":
        """
        - 경로만 받아서 MultiModalSearchEngine + ALSReRanker를 한 번에 세팅하는 헬퍼.
        - 데모/CLI에서 편하게 쓰라고 제공.
        """
        base = MultiModalSearchEngine(
            db_path=db_path,
            bm25_path=bm25_path,
            image_index_path=image_index_path,
            model_name=model_name,
            device=device,
            rrf_k=rrf_k,
            w_text=w_text,
            w_image=w_image,
        )

        reranker = ALSReRanker(
            model_config_path=als_config_path,
            item_key="asin",  # MultiModalSearchEngine 결과의 키
            score_key="rrf_score",  # base score로 RRF 점수 사용
        )

        return cls(multimodal_engine=base, als_reranker=reranker)

    def search(
        self,
        query: str,
        top_k: int = 10,
        stage1_factor: int = 3,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        1) BM25 + CLIP + RRF로 fusion 결과 조회
        2) user_id가 있으면 ALS로 재랭킹
        3) 최종 결과 반환
        """
        logger.info(
            "PersonalizedSearchEngine.search(query=%r, top_k=%d, stage1_factor=%d, user_id=%s)",
            query,
            top_k,
            stage1_factor,
            user_id,
        )
        # 1) 멀티모달 검색
        fusion_results = self.multimodal_engine.search(
            query=query,
            top_k=top_k,
            stage1_factor=stage1_factor,
        )
        logger.debug("Base fusion results count=%d", len(fusion_results))

        # 2) user_id 없으면 그냥 fusion 결과만
        if not user_id:
            logger.info("No user_id supplied → skip ALS reranking.")
            return fusion_results

        # 3) ALS 재랭킹
        logger.info("Applying ALS reranking for user_id=%s", user_id)
        final_results = self.als_reranker.rerank(
            user_id=user_id,
            results=fusion_results,
            top_k=top_k,
        )
        logger.debug("Reranked results count=%d", len(final_results))

        return final_results

    def close(self) -> None:
        """
        내부 MultiModalSearchEngine 자원 해제
        """
        if getattr(self, "multimodal_engine", None) is not None:
            logger.info("Closing base MultiModalSearchEngine")
            self.multimodal_engine.close()
            self.multimodal_engine = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
