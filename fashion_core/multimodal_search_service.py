# fashion_core/multimodal_search_service.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.text_normalization import tokenize  # 이미 쓰던 토크나이저

from fashion_core.bm25_text_engine import BM25TextSearchEngine
from fashion_core.db_image_engine import DbImageSearchEngine
from fashion_core.search_engine_rrf import BM25ClipFusionEngine
from fashion_core.search_results import FusionHit


class FashionSearchService:
    """
    - BM25TextSearchEngine + DbImageSearchEngine + BM25ClipFusionEngine
    를 한 번에 묶어서 사용하는 서비스 레이어.

    - 웹 API / gRPC / CLI 어디서든 이 클래스를 불러다 쓰면 됨.
    """

    def __init__(
        self,
        db_path: Path,
        bm25_path: Path,
        image_index_path: Path,
        model_name: str = "patrickjohncyh/fashion-clip",
        device: Optional[str] = None,
        rrf_k: int = 60,
        w_text: float = 1.0,
        w_image: float = 1.0,
    ):
        self.db_path = Path(db_path)
        self.bm25_path = Path(bm25_path)
        self.image_index_path = Path(image_index_path)

        # 1) 텍스트(BM25) 엔진
        self.text_engine = BM25TextSearchEngine(
            db_path=self.db_path,
            bm25_path=self.bm25_path,
            tokenizer=tokenize,
        )

        # 2) 이미지(CLIP + FAISS + DB) 엔진
        self.image_engine = DbImageSearchEngine(
            db_path=self.db_path,
            index_path=self.image_index_path,
            model_name=model_name,
            device=device,
            tokenizer=tokenize,
        )

        # 3) Fusion 엔진 (RRF)
        self.fusion_engine = BM25ClipFusionEngine(
            text_engine=self.text_engine,
            image_engine=self.image_engine,
            rrf_k=rrf_k,
            w_text=w_text,
            w_image=w_image,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        stage1_factor: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        서비스에서 바로 쓰기 좋은 JSON 스타일 결과 반환용 메서드.
        """
        hits: List[FusionHit] = self.fusion_engine.search(
            query=query,
            top_k=top_k,
            stage1_factor=stage1_factor,
        )

        results: List[Dict[str, Any]] = []
        for h in hits:
            # dataclass → dict
            data = asdict(h)
            # 키 이름 약간 정리해도 되고 그대로 써도 됨
            results.append(
                {
                    "rank": data["rank"],
                    "asin": data["asin"],
                    "item_id": data["db_item_id"],
                    "title": data["title"],
                    "store": data["store"],
                    "image_url": data["image_url"],
                    "bm25_rank": data["bm25_rank"],
                    "bm25_score": data["bm25_score_raw"],
                    "image_rank": data["image_rank"],
                    "image_score": data["image_score_raw"],
                    "rrf_score": data["rrf_score"],
                }
            )
        return results

    def close(self):
        # 자원 해제용(웹 서버 종료 시 등)
        if getattr(self, "text_engine", None) is not None:
            self.text_engine.close()
            self.text_engine = None
        if getattr(self, "image_engine", None) is not None:
            self.image_engine.close()
            self.image_engine = None

    def __del__(self):
        # 혹시라도 GC 시점에 정리
        try:
            self.close()
        except Exception:
            pass
