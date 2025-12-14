# tests/unit/test_db_image_engine.py

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import fashion_core.db_image_engine as img_mod
from fashion_core.db_image_engine import DbImageSearchEngine


@pytest.fixture
def tmp_index_path(tmp_path: Path) -> Path:
    """init에서 index_path.exists() 통과 목적 빈 파일"""
    p = tmp_path / "index.faiss"
    p.write_bytes(b"dummy")
    return p


@pytest.fixture
def engine(monkeypatch, mini_db: Path, tmp_index_path: Path) -> DbImageSearchEngine:
    """ """

    class FakeIndex:
        ntotal = 100

        def search(self, q_emb, k):
            # ids에 -1 섞어서 필터 동작도 같이 검증
            ids = np.array([[103, -1, 101, 102, 104]], dtype=np.int64)[:, :k]
            scores = np.array([[0.9, 0.85, 0.8, 0.7, 0.6]], dtype=np.float32)[:, :k]
            return scores, ids

    monkeypatch.setattr(img_mod.faiss, "read_index", lambda _: FakeIndex())

    eng = DbImageSearchEngine(
        db_path=mini_db,
        index_path=tmp_index_path,
        model_name="dummy",
        device="cpu",
    )

    # 텍스트 인코딩은 search 로직과 무관하니 고정 임베딩으로 대체
    monkeypatch.setattr(
        eng, "_encode_text_query", lambda q: np.ones((1, 4), dtype=np.float32)
    )
    return eng


def test_init_when_db_missing(tmp_path: Path, tmp_index_path: Path):
    with pytest.raises(FileNotFoundError):
        DbImageSearchEngine(
            db_path=tmp_path / "no.db",
            index_path=tmp_index_path,
            model_name="dummy",
            device="cpu",
        )


def test_init_when_index_missing(tmp_path: Path, mini_db: Path):
    with pytest.raises(FileNotFoundError):
        DbImageSearchEngine(
            db_path=mini_db,
            index_path=tmp_path / "no.faiss",
            model_name="dummy",
            device="cpu",
        )


# _normalize 테스트
def test_normalize_unit_length():
    x = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    y = DbImageSearchEngine._normalize(x)
    norms = np.linalg.norm(y, axis=1)
    assert np.allclose(norms, np.ones_like(norms), atol=1e-6)


# search() 테스트
def test_search_returns_hits_sorted_and_ranked(
    monkeypatch, engine: DbImageSearchEngine, mini_db: Path
):
    def fake_load_item_meta_for_ids(conn, item_ids):
        return {
            item_id: {
                "asin": f"ASIN-{item_id}",
                "title": f"title-{item_id}",
                "store": "S",
                "image_url": f"url-{item_id}",
            }
            for item_id in item_ids
        }

    monkeypatch.setattr(img_mod, "load_item_meta_for_ids", fake_load_item_meta_for_ids)
    monkeypatch.setattr(img_mod, "deduplicate_hits_by_asin", lambda hits, k: hits[:k])

    with sqlite3.connect(mini_db) as conn:
        out = engine.search("any query", conn=conn, top_k=3)

    assert isinstance(out, list)
    assert len(out) == 3

    # rank는 1부터 증가
    assert [h.rank for h in out] == [1, 2, 3]

    # score는 내림차순(동점 허용)
    scores = [h.score for h in out]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    # -1은 필터되어야 하므로 item_id에 -1이 없어야 함
    assert all(h.item_id >= 0 for h in out)


def test_search_respects_top_k(monkeypatch, engine: DbImageSearchEngine, mini_db: Path):
    monkeypatch.setattr(
        img_mod,
        "load_item_meta_for_ids",
        lambda conn, ids: {i: {"asin": str(i)} for i in ids},
    )
    monkeypatch.setattr(img_mod, "deduplicate_hits_by_asin", lambda hits, k: hits[:k])

    with sqlite3.connect(mini_db) as conn:
        out = engine.search("shoes", conn=conn, top_k=1)

    assert len(out) == 1


def test_search_returns_empty_when_no_valid_ids(
    monkeypatch, mini_db: Path, tmp_index_path: Path
):
    """index 결과가 전부 -1이면 [] 반환"""

    class FakeIndexAllInvalid:
        ntotal = 10

        def search(self, q_emb, k):
            ids = np.array([[-1, -1, -1]], dtype=np.int64)[:, :k]
            scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)[:, :k]
            return scores, ids

    class DummyModel:
        def to(self, device):
            return self

    class DummyProcessor:
        pass

    monkeypatch.setattr(img_mod.faiss, "read_index", lambda _: FakeIndexAllInvalid())
    monkeypatch.setattr(img_mod.CLIPModel, "from_pretrained", lambda _: DummyModel())
    monkeypatch.setattr(
        img_mod.CLIPProcessor, "from_pretrained", lambda _: DummyProcessor()
    )

    eng = DbImageSearchEngine(
        db_path=mini_db,
        index_path=tmp_index_path,
        model_name="dummy",
        device="cpu",
    )
    monkeypatch.setattr(
        eng, "_encode_text_query", lambda q: np.ones((1, 4), dtype=np.float32)
    )

    # 메타/디듭은 호출되기 전에 return [] 되어야 함
    conn = sqlite3.connect(mini_db)
    out = eng.search("q", conn=conn, top_k=5)
    conn.close()

    assert out == []


def test_search_applies_deduplicate_by_asin(
    monkeypatch, engine: DbImageSearchEngine, mini_db: Path
):
    # 101과 103이 같은 asin이라고 가정
    def fake_load_item_meta_for_ids(conn, item_ids):
        return {
            101: {"asin": "SAME"},
            102: {"asin": "UNIQ"},
            103: {"asin": "SAME"},
            104: {"asin": "UNIQ2"},
        }

    def fake_dedup(hits, k):
        seen = set()
        out = []
        for h in hits:
            if h.asin in seen:
                continue
            seen.add(h.asin)
            out.append(h)
        return out[:k]

    monkeypatch.setattr(img_mod, "load_item_meta_for_ids", fake_load_item_meta_for_ids)
    monkeypatch.setattr(img_mod, "deduplicate_hits_by_asin", fake_dedup)

    conn = sqlite3.connect(mini_db)
    out = engine.search("red", conn=conn, top_k=10)
    conn.close()

    asins = [h.asin for h in out]
    assert len(asins) == len(set(asins))
