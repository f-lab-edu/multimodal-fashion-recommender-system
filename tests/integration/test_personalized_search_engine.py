# tests/integration/test_personalized_search_engine.py

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

import fashion_core.db_image_engine as img_mod
from fashion_core.personalized_search_engine import PersonalizedSearchEngine


@pytest.fixture
def mini_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "items.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # load_item_meta_for_ids()가 읽는 컬럼과 "반드시" 맞아야 함
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

    docs = [
        "red dress summer",  # -> id=1, asin=A
        "blue shoes running",  # -> id=2, asin=B
        "green shirt casual",  # -> id=3, asin=C
    ]
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
            [1.0, 0.0, 0.0, 0.0],  # id=1 (A)
            [0.0, 1.0, 0.0, 0.0],  # id=2 (B)
            [0.0, 0.0, 1.0, 0.0],  # id=3 (C)
        ],
        dtype=np.float32,
    )
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
    ids = np.array([1, 2, 3], dtype=np.int64)

    index.add_with_ids(vecs, ids)
    faiss.write_index(index, str(index_path))
    return index_path


# ALS 아티팩트
@pytest.fixture
def als_config_path(tmp_path: Path) -> Path:
    """
    ALSReRanker가 로드할 파일셋 만들기
    의도적으로 u1이 "A"를 점수 높게 만들어서 rerank 결과가 눈에 띄게 바뀌게 함.
    """
    model_dir = tmp_path / "als_model_dir"
    model_dir.mkdir(parents=True)

    # user 1명(u1), items 3개(A,B,C), dim=2
    user_factors = np.array([[1.0, 0.0]], dtype=np.float32)  # u1
    item_factors = np.array(
        [
            [2.0, 0.0],  # A (u1과 dot=2.0) -> ALS가 A를 강하게 밀도록
            [0.0, 1.0],  # B (dot=0.0)
            [0.5, 0.0],  # C (dot=0.5)
        ],
        dtype=np.float32,
    )

    fake_model = SimpleNamespace(user_factors=user_factors, item_factors=item_factors)
    mappings = {
        "user2idx": {"u1": 0},
        "item2idx": {"A": 0, "B": 1, "C": 2},
    }

    with (model_dir / "als_model.pkl").open("wb") as f:
        pickle.dump(fake_model, f)
    with (model_dir / "mappings.pkl").open("wb") as f:
        pickle.dump(mappings, f)

    cfg = {
        "model_dir": str(model_dir),
        "w_als": 0.9,  # ALS 영향 크게(순위 변화를 확실히)
        "model_version": "v-it",
    }
    cfg_path = tmp_path / "als_config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    return cfg_path


@pytest.fixture
def patch_clip(monkeypatch):
    """
    PR/CI에서 CLIP 모델을 실제로 다운로드/로딩하지 않게 막음
    """

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


@pytest.fixture
def patch_image_query_embedding(monkeypatch):
    # 이미지 검색이 B가 1등, A가 2등이 되도록 유사도 차이를 줌
    # 벡터들: A=[1,0,0,0], B=[0,1,0,0], C=[0,0,1,0]
    # q=[0.2, 1.0, -0.1, 0.0] => dot(B)=1.0, dot(A)=0.2, dot(C)=-0.1
    def fake_encode_text_query(self, query: str) -> np.ndarray:
        q = np.array([[0.2, 1.0, -0.1, 0.0]], dtype=np.float32)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
        return q

    monkeypatch.setattr(
        img_mod.DbImageSearchEngine, "_encode_text_query", fake_encode_text_query
    )


def make_engine(
    mini_db: Path,
    mini_bm25_pkl: Path,
    mini_faiss_index: Path,
    als_config_path: Path,
):
    # base는 image만 크게 보게 해서(B가 1등으로 뜨게) ALS로 순위가 뒤집히는 걸 확인하기 좋게 설정
    return PersonalizedSearchEngine.from_paths(
        db_path=mini_db,
        bm25_path=mini_bm25_pkl,
        image_index_path=mini_faiss_index,
        als_config_path=als_config_path,
        model_name="dummy",
        device="cpu",
        rrf_k=60,
        w_text=0.0,  # base에서 텍스트 영향 제거(의도적으로)
        w_image=1.0,
    )


## test 함수
def test_integration_no_user_no_session_returns_base(
    patch_clip,
    patch_image_query_embedding,
    mini_db: Path,
    mini_bm25_pkl: Path,
    mini_faiss_index: Path,
    als_config_path: Path,
):
    # user_id도 없고 session_item_ids도 없으면 ALS를 스킵하고 base fusion 결과 그대로 나오는지
    eng = make_engine(mini_db, mini_bm25_pkl, mini_faiss_index, als_config_path)

    out = eng.search("red dress", top_k=2, user_id=None, session_item_ids=None)
    assert len(out) == 2

    # ALS를 안 탔으니 score_final 같은 필드는 없어야 정상
    assert "score_final" not in out[0]
    assert "als_mode" not in out[0]

    eng.close()


def test_integration_user_id_applies_als_rerank(
    patch_clip,
    patch_image_query_embedding,
    mini_db: Path,
    mini_bm25_pkl: Path,
    mini_faiss_index: Path,
    als_config_path: Path,
):
    # user_id="u1"이면 ALS가 적용되어 결과가 재정렬되는지
    eng = make_engine(mini_db, mini_bm25_pkl, mini_faiss_index, als_config_path)

    base = eng.multimodal_engine.search("red dress", top_k=2)
    base_asins = [r["asin"] for r in base]
    assert "B" in base_asins  # ALS가 끌어올릴 후보에 있어야 함
    assert "A" in base_asins

    out = eng.search("red dress", top_k=2, user_id="u1", session_item_ids=None)
    assert len(out) == 2

    # ALS 필드가 붙어야 함
    for r in out:
        assert "score_final" in r
        assert r["als_model_version"] == "v-it"
        assert r["als_mode"] == "user"

    # u1은 A를 강하게 선호하도록 만들었으니 1등이 A가 되는 걸 기대
    assert out[0]["asin"] == "A"

    eng.close()


def test_integration_session_items_applies_als_when_user_unknown(
    patch_clip,
    patch_image_query_embedding,
    mini_db: Path,
    mini_bm25_pkl: Path,
    mini_faiss_index: Path,
    als_config_path: Path,
):
    # user_id가 학습셋에 없어도(session cold) session_item_ids가 있으면 2단계(item2item)로 ALS가 적용되는지.
    eng = make_engine(mini_db, mini_bm25_pkl, mini_faiss_index, als_config_path)

    out = eng.search(
        "red dress", top_k=2, user_id="no_such_user", session_item_ids=["A"]
    )
    assert len(out) == 2

    for r in out:
        assert "score_final" in r
        assert r["als_mode"] == "session_items"

    # session에 A가 있으니 A가 1등으로 오기 쉬움
    assert out[0]["asin"] == "A"

    eng.close()


def test_integration_fallback_when_session_invalid(
    patch_clip,
    patch_image_query_embedding,
    mini_db: Path,
    mini_bm25_pkl: Path,
    mini_faiss_index: Path,
    als_config_path: Path,
):
    # user도 cold이고 session_item_ids도 전부 매핑에 없으면 ALS가 None 반환
    eng = make_engine(mini_db, mini_bm25_pkl, mini_faiss_index, als_config_path)

    try:
        base = eng.multimodal_engine.search("red dress", top_k=2)
        out = eng.search(
            "red dress", top_k=2, user_id="no_such_user", session_item_ids=["X", "Y"]
        )

        # ALS 적용 불가 -> base 그대로 반환(상위 top_k)
        assert [r["asin"] for r in out] == [r["asin"] for r in base]
        assert "score_final" not in out[0]
    finally:
        eng.close()
