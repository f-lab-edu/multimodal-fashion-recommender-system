# service/service.py
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import bentoml
import torch

from fashion_core.personalized_search_engine import PersonalizedSearchEngine

DB_PATH = Path(os.getenv("DB_PATH", "data/fashion_items.db"))
BM25_PATH = Path(os.getenv("BM25_PATH", "data/bm25_from_item_docs.pkl"))
IMAGE_INDEX_PATH = Path(
    os.getenv("IMAGE_INDEX_PATH", "data/image_index_with_ids.faiss")
)
ALS_CONFIG_PATH = Path(os.getenv("ALS_CONFIG_PATH", "config/als_model.yaml"))


@bentoml.service(name="fashion_search_service")
class FashionSearchService:
    """
    멀티모달 + ALS 개인화 검색 서비스
    """

    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 엔진 초기화
        self.engine = PersonalizedSearchEngine.from_paths(
            db_path=DB_PATH,
            bm25_path=BM25_PATH,
            image_index_path=IMAGE_INDEX_PATH,
            als_config_path=ALS_CONFIG_PATH,
            model_name="patrickjohncyh/fashion-clip",
            device=device,
            rrf_k=60,
            w_text=1.0,
            w_image=1.0,
        )

    @bentoml.api
    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_item_ids: Optional[List[str]] = None,
        top_k: int = 10,
        stage1_factor: int = 3,
    ) -> Dict[str, Any]:
        """
        example request body:
        {
          "query": "black linen shirts",
          "user_id": "AE23BYWB52METWQVHSPN3MKN7AJA",      // optional
          "session_item_ids": ["B08FGCD1FC", "B09JB7H124"],      // optional
          "top_k": 50,
          "stage1_factor": 3
        }

        response:
        {
          "results": [...],
          "meta": {...}
        }
        """

        results = self.engine.search(
            query=query,
            top_k=top_k,
            stage1_factor=stage1_factor,
            user_id=user_id,
            session_item_ids=session_item_ids,
        )

        return {
            "results": results,
            "meta": {
                "query": query,
                "top_k": top_k,
                "stage1_factor": stage1_factor,
                "user_id": user_id,
                "session_len": len(session_item_ids) if session_item_ids else 0,
            },
        }
