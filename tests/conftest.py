# tests/conftest.py
from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import faiss
from rank_bm25 import BM25Okapi


# ---- 공용 패치들 (CI/PR 안정화용) ----


@pytest.fixture(autouse=True)
def patch_sqlite_row_factory(monkeypatch):
    """
    엔진 내부에서 새로 여는 sqlite connection에도 row_factory=sqlite3.Row가 적용되게
    """
    real_connect = sqlite3.connect

    def patched_connect(path, *args, **kwargs):
        conn = real_connect(path, *args, **kwargs)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(sqlite3, "connect", patched_connect)


@pytest.fixture(autouse=True)
def patch_clip_no_download(monkeypatch):
    """
    transformers/torch 다운로드 방지 (PR/CI)
    """
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


@pytest.fixture(autouse=True)
def patch_default_image_query_embedding(monkeypatch):
    """
    기본 이미지 쿼리 임베딩을 고정해서 결과를 안정화.
    - B(id=2)가 1등
    - A(id=1)가 2등
    - C(id=3)는 밀리게
    """
    import fashion_core.db_image_engine as img_mod

    def fake_encode_text_query(self, query: str) -> np.ndarray:
        q = np.array([[0.2, 1.0, -0.1, 0.0]], dtype=np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
        return q

    monkeypatch.setattr(
        img_mod.DbImageSearchEngine, "_encode_text_query", fake_encode_text_query
    )


# ---- 공용 아티팩트 fixture들 ----


@pytest.fixture
def mini_db(tmp_path: Path) -> Path:
    """
    search_utils.load_item_meta_for_ids()가 기대하는 컬럼 스키마로 생성
    """
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

    with pkl_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "item_ids": [1, 2, 3]}, f)

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
    """
    ALSReRanker가 로드할 최소 파일셋 생성
    - u1이 A를 강하게 선호하도록 구성
    """
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
            {"model_dir": str(model_dir), "w_als": 0.9, "model_version": "v-test"}, f
        )

    return cfg_path
