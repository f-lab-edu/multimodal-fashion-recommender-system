# service/service.py
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import bentoml
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from fashion_core.personalized_search_engine import PersonalizedSearchEngine

DB_PATH = Path(os.getenv("DB_PATH", "data/fashion_items.db"))
BM25_PATH = Path(os.getenv("BM25_PATH", "data/bm25_from_item_docs.pkl"))
IMAGE_INDEX_PATH = Path(
    os.getenv("IMAGE_INDEX_PATH", "data/image_index_with_ids.faiss")
)
ALS_CONFIG_PATH = Path(os.getenv("ALS_CONFIG_PATH", "config/als_model.yaml"))

ui = FastAPI()
ui_dir = Path(__file__).resolve().parent / "ui"
ui.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")


@bentoml.service(name="fashion_search_service")
@bentoml.asgi_app(ui, path="/ui")
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

    @staticmethod
    def _validate_input(
        query: Any,
        top_k: Any,
        stage1_factor: Any,
        session_item_ids: Any,
    ) -> None:
        # 입력 검증
        if not isinstance(query, str) or not query.strip():
            raise ValueError("`query`를 입력해야 합니다.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("`top_k`는 1 이상의 정수여야 합니다.")

        if not isinstance(stage1_factor, int) or stage1_factor <= 0:
            raise ValueError("`stage1_factor`는 1 이상의 정수여야 합니다.")

        if session_item_ids is not None:
            if not isinstance(session_item_ids, list):
                raise ValueError("`session_item_ids`는 문자열 리스트여야 합니다.")
            for sid in session_item_ids:
                if not isinstance(sid, str) or not sid.strip():
                    raise ValueError(
                        "`session_item_ids`에는 빈 문자열을 넣을 수 없습니다."
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
          "top_k": 10,
          "stage1_factor": 3
        }
        """
        # 입력 검증
        self._validate_input(
            query=query,
            top_k=top_k,
            stage1_factor=stage1_factor,
            session_item_ids=session_item_ids,
        )

        try:
            results = self.engine.search(
                query=query,
                top_k=top_k,
                stage1_factor=stage1_factor,
                user_id=user_id,
                session_item_ids=session_item_ids,
            )
            success = True
            error: Optional[Dict[str, Any]] = None

        except Exception as e:
            results = []
            success = False
            error = {
                "code": "INTERNAL_ERROR",
                "message": "검색 도중 내부 에러가 발생했습니다.",
                "details": str(e),
            }

        return {
            "results": results,
            "success": success,
            "error": error,
            "meta": {
                "query": query,
                "top_k": top_k,
                "stage1_factor": stage1_factor,
                "user_id": user_id,
                "session_len": len(session_item_ids) if session_item_ids else 0,
            },
        }
