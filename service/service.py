# service/service.py
from pathlib import Path
from typing import Any, Dict, Optional, List

import bentoml
from bentoml.io import JSON

from fashion_core.personalized_search_engine import PersonalizedSearchEngine

DB_PATH = Path("data/fashion_items.db")
BM25_PATH = Path("data/bm25_from_item_docs.pkl")
IMAGE_INDEX_PATH = Path("data/image_index_with_ids.faiss")
ALS_CONFIG_PATH = Path("config/als_model.yaml")

engine = PersonalizedSearchEngine.from_paths(
    db_path=DB_PATH,
    bm25_path=BM25_PATH,
    image_index_path=IMAGE_INDEX_PATH,
    als_config_path=ALS_CONFIG_PATH,
    model_name="patrickjohncyh/fashion-clip",
    device="cuda" if bentoml.frameworks.torch.utils.is_cuda_available() else "cpu",
    rrf_k=60,
    w_text=1.0,
    w_image=1.0,
)

svc = bentoml.Service("fashion_search_service")


@svc.api(input=JSON(), output=JSON())
def search(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    example
    {
      "query": "black linen shirts",
      "user_id": "AE23BYWB52METWQVHSPN3MKN7AJA",  // optional
      "top_k": 50,
      "stage1_factor": 3
    }
    """
    query = body.get("query")
    if not query:
        raise ValueError("query is required")

    user_id: Optional[str] = body.get("user_id")
    session_item_ids_raw = body.get("session_item_ids")
    session_item_ids: Optional[List[str]] = None
    if isinstance(session_item_ids_raw, list):
        session_item_ids = [str(i) for i in session_item_ids_raw]

    top_k = int(body.get("top_k", 10))
    stage1_factor = int(body.get("stage1_factor", 3))

    results = engine.search(
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
