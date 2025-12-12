# tests/e2e/test_service_api.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pickle
import sqlite3

import numpy as np
import pytest
import yaml

import faiss
from rank_bm25 import BM25Okapi

import service.service as svc_mod
from service.service import FashionSearchService


# Fixtures: 최소 아티팩트들
@pytest.fixture
def mini_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "items.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            parent_asin TEXT,
            title TEXT,
            store TEXT,
            image_main_url TEXT
        )
        """
    )

    rows = [
        (1, "A", "Red Dress", "S", "url-A"),
        (2, "B", "Blue Shoes", "S", "url-B"),
        (3, "C", "Green Shirt", "S", "url-C"),
    ]

    cur.executemany(
        "INSERT INTO items (id, parent_asin, title, store, image_main_url) VALUES (?, ?, ?, ?, ?)",
        rows,
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mini_bm25_pkl(tmp_path: Path) -> Path:
    pkl_path = tmp_path / "bm25.pkl"
    docs = ["red dress summer", "blue shoes running", "green shirt casual"]
    tokenized = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    item_ids = [1, 2, 3]
    with pkl_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "item_ids": item_ids}, f)
    return pkl_path


@pytest.fixture
def mini_faiss_index(tmp_path: Path) -> Path:
    index_path = tmp_path / "index.faiss"

    d = 4
    base = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap(base)

    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # id=1(A)
            [0.0, 1.0, 0.0, 0.0],  # id=2(B)
            [0.0, 0.0, 1.0, 0.0],  # id=3(C)
        ],
        dtype=np.float32,
    )
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
    ids = np.array([1, 2, 3], dtype=np.int64)

    index.add_with_ids(vecs, ids)
    faiss.write_index(index, str(index_path))
    return index_path


@pytest.fixture
def als_config_path(tmp_path: Path) -> Path:
    model_dir = tmp_path / "als_model_dir"
    model_dir.mkdir(parents=True)

    user_factors = np.array([[1.0, 0.0]], dtype=np.float32)  # u1
    item_factors = np.array(
        [
            [2.0, 0.0],  # A
            [0.0, 1.0],  # B
            [0.5, 0.0],  # C
        ],
        dtype=np.float32,
    )
    fake_model = SimpleNamespace(user_factors=user_factors, item_factors=item_factors)
    mappings = {"user2idx": {"u1": 0}, "item2idx": {"A": 0, "B": 1, "C": 2}}

    with (model_dir / "als_model.pkl").open("wb") as f:
        pickle.dump(fake_model, f)
    with (model_dir / "mappings.pkl").open("wb") as f:
        pickle.dump(mappings, f)

    cfg_path = tmp_path / "als_config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"model_dir": str(model_dir), "w_als": 0.9, "model_version": "v-e2e"}, f
        )

    return cfg_path


# Fixtures: CLIP 다운로드 방지 + query embedding 고정
@pytest.fixture(autouse=True)
def patch_clip_and_embedding(monkeypatch):
    # service가 내부에서 쓰는 DbImageSearchEngine 모듈을 통해 patch해야 함
    import fashion_core.db_image_engine as img_mod

    class DummyModel:
        def to(self, device):
            return self

        def get_text_features(self, **kwargs):
            return None

    class DummyProcessor:
        def __call__(self, *args, **kwargs):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(
        img_mod.CLIPModel, "from_pretrained", lambda *_a, **_kw: DummyModel()
    )
    monkeypatch.setattr(
        img_mod.CLIPProcessor, "from_pretrained", lambda *_a, **_kw: DummyProcessor()
    )

    # 이미지 쿼리는 B(id=2)가 유리하게
    def fake_encode_text_query(self, query: str) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        img_mod.DbImageSearchEngine, "_encode_text_query", fake_encode_text_query
    )


# Fixture: 서비스가 참조하는 전역 PATH를 테스트 아티팩트로 교체
@pytest.fixture
def service_instance(
    monkeypatch, mini_db, mini_bm25_pkl, mini_faiss_index, als_config_path
):
    # service/service.py의 전역 경로를 테스트용으로 교체
    import fashion_core.multimodal_search_engine as mm_mod

    real_connect = sqlite3.connect

    def patched_connect(path, *args, **kwargs):
        conn = real_connect(path, *args, **kwargs)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(mm_mod.sqlite3, "connect", patched_connect)

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
    # if not resp["success"]:
    #     print(resp["error"])
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
