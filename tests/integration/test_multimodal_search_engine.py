# tests/integration/test_multimodal_search_engine.py

from __future__ import annotations

from pathlib import Path
import pickle
import sqlite3

import numpy as np
import pytest

import faiss
from rank_bm25 import BM25Okapi

import fashion_core.db_image_engine as img_mod
from fashion_core.multimodal_search_engine import MultiModalSearchEngine


@pytest.fixture
def mini_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "items.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
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
    """
    {"bm25": BM25Okapi, "item_ids": [db_pk...]} 형태로 pkl 생성
    """
    pkl_path = tmp_path / "bm25.pkl"

    # 문서/아이템 매핑: doc i ↔ item_ids[i]
    docs = [
        "red dress summer",
        "blue shoes running",
        "green shirt casual",
    ]
    tokenized = [d.split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    item_ids = [1, 2, 3]

    with pkl_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "item_ids": item_ids}, f)

    return pkl_path


@pytest.fixture
def mini_faiss_index(tmp_path: Path) -> Path:
    """
    item_id를 그대로 FAISS의 id로 쓰는 index 생성 (IndexIDMap)
    """
    index_path = tmp_path / "index.faiss"

    d = 4
    base = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap(base)

    # (id=1..3) 벡터: 정규화해서 inner product 유사도 사용
    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # id=1
            [0.0, 1.0, 0.0, 0.0],  # id=2
            [0.0, 0.0, 1.0, 0.0],  # id=3
        ],
        dtype=np.float32,
    )
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
    ids = np.array([1, 2, 3], dtype=np.int64)

    index.add_with_ids(vecs, ids)
    faiss.write_index(index, str(index_path))
    return index_path


@pytest.fixture
def patch_clip(monkeypatch):
    """
    transformers/torch 모델 다운로드 방지용 더미 패치
    """

    class DummyModel:
        def to(self, device):
            return self

        def get_text_features(self, **kwargs):
            # 실제로는 쓰지 않게 _encode_text_query를 패치할 예정
            return None

    class DummyProcessor:
        def __call__(self, *args, **kwargs):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(
        img_mod.CLIPModel, "from_pretrained", lambda *_args, **_kw: DummyModel()
    )
    monkeypatch.setattr(
        img_mod.CLIPProcessor, "from_pretrained", lambda *_args, **_kw: DummyProcessor()
    )


def test_integration_multimodal_search_end_to_end(
    monkeypatch,
    patch_clip,
    mini_db: Path,
    mini_bm25_pkl: Path,
    mini_faiss_index: Path,
):
    """
    실제 구성요소 연결:
    - sqlite DB
    - bm25 pkl 로드 후 BM25 검색
    - faiss index 로드 후 이미지 검색(단, query embedding은 패치)
    - fusion(rrf) 후 MultiModalSearchEngine.search() 결과 dict 생성
    """

    # 이미지 쿼리 임베딩을 id=2가 가장 가깝게 나오도록 고정
    # (d=4, id=2 벡터가 [0,1,0,0])
    def fake_encode_text_query(self, query: str) -> np.ndarray:
        q = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        return q  # 이미 정규화된 형태

    monkeypatch.setattr(
        img_mod.DbImageSearchEngine, "_encode_text_query", fake_encode_text_query
    )

    eng = MultiModalSearchEngine(
        db_path=mini_db,
        bm25_path=mini_bm25_pkl,
        image_index_path=mini_faiss_index,
        model_name="dummy",
        device="cpu",
        rrf_k=60,
        w_text=1.0,
        w_image=1.0,
    )

    out = eng.search("red dress", top_k=2, stage1_factor=3)

    assert isinstance(out, list)
    assert len(out) == 2

    # 결과 dict 스키마 최소 확인
    for r in out:
        assert "rank" in r
        assert "asin" in r
        assert "item_id" in r
        assert "rrf_score" in r

    # rank는 1..top_k
    assert [r["rank"] for r in out] == [1, 2]

    # 만든 DB의 asin들 중에서 나와야 함
    asins = {r["asin"] for r in out}
    assert asins.issubset({"A", "B", "C"})

    assert "A" in asins
    assert "B" in asins

    eng.close()
