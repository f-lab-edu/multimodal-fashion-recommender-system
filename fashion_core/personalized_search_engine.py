# fashion_core/personalized_search_engine.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pathlib import Path

from fashion_core.multimodal_search_engine import MultiModalSearchEngine
from fashion_core.als_reranker import ALSReRanker

logger = logging.getLogger(__name__)


class PersonalizedSearchEngine:
    """
    MultiModalSearchEngine + ALSReRanker 를 결합한 개인맞춤 검색 엔진

    - 1단계: BM25 + CLIP + RRF 로 기본 fusion 검색
    - 2단계: ALS (user_id 또는 session_item_ids) 로 재랭킹 시도
    - 3단계: ALS 를 사용할 수 없으면 기본 fusion 결과 그대로 반환
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
        session_item_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        1) BM25 + CLIP + RRF로 fusion 결과 조회
        2) user_id 또는 session_item_ids를 이용해 ALS 재랭킹 시도
        3) ALS 사용 불가 시, fusion 결과 그대로 반환
        """
        logger.info(
            "PersonalizedSearchEngine.search(query=%r, top_k=%d, "
            "stage1_factor=%d, user_id=%s, session_len=%s)",
            query,
            top_k,
            stage1_factor,
            user_id,
            len(session_item_ids) if session_item_ids else 0,
        )

        # 1) 멀티모달 검색
        fusion_results = self.multimodal_engine.search(
            query=query,
            top_k=top_k,
            stage1_factor=stage1_factor,
        )
        logger.debug("Base fusion results count=%d", len(fusion_results))

        # user_id도 없고 session_item_ids도 없으면 → ALS로 할 수 있는 게 없음
        if user_id is None and not session_item_ids:
            logger.info(
                "No user_id and no session_item_ids supplied → skip ALS reranking."
            )
            # fusion_results 자체가 이미 top_k를 만족하도록 나왔을 수도 있지만,
            # 방어적으로 잘라준다.
            return fusion_results[:top_k]

        # 2) ALS 재랭킹 시도
        logger.info(
            "Applying ALS reranking (user_id=%s, session_len=%s)",
            user_id,
            len(session_item_ids) if session_item_ids else 0,
        )

        reranked = self.als_reranker.rerank(
            user_id=user_id,
            results=fusion_results,
            session_item_ids=session_item_ids,
            top_k=top_k,
        )

        # 3) ALS로 점수를 줄 수 없으면 (ALSReRanker에서 None 반환)
        if reranked is None:
            logger.info(
                "ALS reranking not applicable (cold-start user & empty/invalid session). "
                "Fallback to base fusion results."
            )
            return fusion_results[:top_k]

        logger.debug("Reranked results count=%d", len(reranked))

        return reranked

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
