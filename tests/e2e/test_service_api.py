# tests/e2e/test_service_api.py
from __future__ import annotations

from pathlib import Path
import pytest

import service.service as svc_mod
from service.service import FashionSearchService


# Fixture: 서비스가 참조하는 전역 PATH를 테스트 아티팩트로 교체
@pytest.fixture
def service_instance(
    monkeypatch, mini_db, mini_bm25_pkl, mini_faiss_index, als_config_path
):
    # service/service.py의 전역 경로를 테스트용으로 교체
    monkeypatch.setattr(svc_mod, "DB_PATH", Path(mini_db))
    monkeypatch.setattr(svc_mod, "BM25_PATH", Path(mini_bm25_pkl))
    monkeypatch.setattr(svc_mod, "IMAGE_INDEX_PATH", Path(mini_faiss_index))
    monkeypatch.setattr(svc_mod, "ALS_CONFIG_PATH", Path(als_config_path))

    # torch cuda 체크도 고정
    monkeypatch.setattr(svc_mod.torch.cuda, "is_available", lambda: False)

    return FashionSearchService()


# E2E: API 메서드 테스트
def test_service_search_success_minimal(service_instance: FashionSearchService):
    resp = service_instance.search(query="red dress", top_k=2, stage1_factor=3)

    assert resp["success"] is True
    assert resp["error"] is None
    assert isinstance(resp["results"], list)
    assert len(resp["results"]) == 2

    meta = resp["meta"]
    assert meta["query"] == "red dress"
    assert meta["top_k"] == 2
    assert meta["stage1_factor"] == 3
    assert meta["session_len"] == 0


def test_service_search_applies_als_when_user_provided(
    service_instance: FashionSearchService,
):
    resp = service_instance.search(
        query="red dress", user_id="u1", top_k=2, stage1_factor=3
    )

    assert resp["success"] is True
    results = resp["results"]
    assert len(results) == 2

    # ALSReRanker가 붙이면 score_final/als_mode 필드가 생김
    assert "score_final" in results[0]
    assert results[0]["als_mode"] == "user"
    assert results[0]["als_model_version"] == "v-test"


def test_service_search_validation_errors(service_instance: FashionSearchService):
    # query 빈값
    with pytest.raises(ValueError):
        service_instance.search(query="   ")

    # top_k 잘못
    with pytest.raises(ValueError):
        service_instance.search(query="q", top_k=0)

    # stage1_factor 잘못
    with pytest.raises(ValueError):
        service_instance.search(query="q", stage1_factor=0)

    # session_item_ids 타입 잘못
    with pytest.raises(ValueError):
        service_instance.search(query="q", session_item_ids="A")  # type: ignore

    # session_item_ids 내부 값 잘못
    with pytest.raises(ValueError):
        service_instance.search(query="q", session_item_ids=["", "B"])  # 빈 문자열
