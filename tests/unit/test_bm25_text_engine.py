# tests/unit/test_bm25_text_engine.py
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

import fashion_core.bm25_text_engine as bm25_mod
from fashion_core.bm25_text_engine import BM25TextSearchEngine


@pytest.fixture
def engine(mini_db: Path, mini_bm25_pkl: Path) -> BM25TextSearchEngine:
    """BM25TextSearchEngine 인스턴스 생성"""
    return BM25TextSearchEngine(
        db_path=mini_db,
        bm25_path=mini_bm25_pkl,
    )


def test_init_when_db_missing(tmp_path: Path, mini_bm25_pkl: Path):
    """DB 파일이 없을 때 FileNotFoundError 발생"""
    with pytest.raises(FileNotFoundError):
        BM25TextSearchEngine(
            db_path=tmp_path / "no.db",
            bm25_path=mini_bm25_pkl,
        )


def test_init_when_bm25_missing(tmp_path: Path, mini_db: Path):
    """BM25(pkl) 파일이 없을 때 FileNotFoundError 발생"""
    with pytest.raises(FileNotFoundError):
        BM25TextSearchEngine(
            db_path=mini_db,
            bm25_path=tmp_path / "no.pkl",
        )


def test_search_returns_hits_sorted_and_ranked(
    monkeypatch, engine: BM25TextSearchEngine, mini_db: Path
):
    """
    search() 가 정상 쿼리에서
    list 반환
    요청한 top-k만큼 나오는지
    hit rank가 순서대로인지
    score가 내림차순 인지
    item_id가 엔진이 가진 item_ids 매핑 범위 안에 있는지 확인.

    monkeypatch로 대체해서 BM25 검색 로직만 unit으로 검증
    """

    def fake_load_item_meta_for_ids(conn, item_ids):
        m = {}
        for item_id in item_ids:
            m[item_id] = {
                "asin": f"ASIN-{item_id}",
                "title": f"title-{item_id}",
                "store": "S",
                "image_url": f"url_{item_id}",
            }
        return m

    monkeypatch.setattr(bm25_mod, "load_item_meta_for_ids", fake_load_item_meta_for_ids)
    monkeypatch.setattr(bm25_mod, "deduplicate_hits_by_asin", lambda hits, k: hits[:k])

    conn = sqlite3.connect(mini_db)
    try:
        out = engine.search("Red dress", conn=conn, top_k=3)
    finally:
        conn.close()

    assert isinstance(out, list)
    assert len(out) == 3

    # rank는 1부터 증가
    assert [h.rank for h in out] == [1, 2, 3]

    scores = [h.score for h in out]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    assert out[0].item_id in engine.item_ids


def test_search_respects_top_k(
    monkeypatch, engine: BM25TextSearchEngine, mini_db: Path
):
    # top_k=1을 줬을 때 결과가 정확히 1개인지
    monkeypatch.setattr(
        bm25_mod,
        "load_item_meta_for_ids",
        lambda conn, ids: {i: {"asin": str(i)} for i in ids},
    )
    monkeypatch.setattr(bm25_mod, "deduplicate_hits_by_asin", lambda hits, k: hits[:k])

    conn = sqlite3.connect(mini_db)
    try:
        out = engine.search("shoes", conn=conn, top_k=1)
    finally:
        conn.close()

    assert len(out) == 1


def test_search_returns_empty_when_scores_empty(
    engine: BM25TextSearchEngine, mini_db: Path
):
    # 빈 배열을 반환하는 특이 상황에서 search()가 크래시 없이 빈 리스트 []를 반환하는지 확인.
    class FakeBM25:
        def get_scores(self, tokens):
            return np.array([])

    engine.bm25 = FakeBM25()

    conn = sqlite3.connect(mini_db)
    try:
        out = engine.search("anything", conn=conn, top_k=10)
    finally:
        conn.close()

    assert out == []


@pytest.fixture
def bm25_4items_pkl(tmp_path: Path) -> Path:
    """
    deduplicate 테스트용: item_ids가 101~104인 bm25.pkl 생성
    """
    pkl_path = tmp_path / "bm25_4.pkl"
    corpus = [
        ["red", "dress"],
        ["blue", "jeans"],
        ["red", "shoes"],
        ["black", "shoes"],
    ]
    bm25 = BM25Okapi(corpus)
    item_ids = [101, 102, 103, 104]
    with pkl_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "item_ids": item_ids}, f)
    return pkl_path


def test_search_applies_deduplicate_by_asin(
    monkeypatch, mini_db: Path, bm25_4items_pkl: Path
):
    eng = BM25TextSearchEngine(db_path=mini_db, bm25_path=bm25_4items_pkl)

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

    monkeypatch.setattr(bm25_mod, "load_item_meta_for_ids", fake_load_item_meta_for_ids)
    monkeypatch.setattr(bm25_mod, "deduplicate_hits_by_asin", fake_dedup)

    conn = sqlite3.connect(mini_db)
    try:
        out = eng.search("red", conn=conn, top_k=10)
    finally:
        conn.close()

    asins = [h.asin for h in out]
    assert len(asins) == len(set(asins))

    close = getattr(eng, "close", None)
    if callable(close):
        close()
