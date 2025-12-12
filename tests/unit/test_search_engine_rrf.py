# tests/unit/test_search_engine_rrf.py
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fashion_core.multimodal_search_engine import BM25ClipFusionEngine


class FakeTextEngine:
    def __init__(self, hits):
        self.hits = hits
        self.last_query = None
        self.last_top_k = None
        self.last_conn = None

    def search(self, query, top_k, conn):
        self.last_query = query
        self.last_top_k = top_k
        self.last_conn = conn
        return self.hits


class FakeImageEngine:
    def __init__(self, hits):
        self.hits = hits
        self.last_query = None
        self.last_top_k = None
        self.last_conn = None

    def search(self, query, top_k, conn):
        self.last_query = query
        self.last_top_k = top_k
        self.last_conn = conn
        return self.hits


def H(
    *,
    item_id: int,
    asin: str | None,
    rank: int,
    score: float,
    title: str | None = None,
    store: str | None = None,
    image_url: str | None = None,
):
    """SearchHit 흉내"""
    return SimpleNamespace(
        item_id=item_id,
        asin=asin,
        rank=rank,
        score=score,
        title=title,
        store=store,
        image_url=image_url,
    )


def test_rrf_formula():
    assert BM25ClipFusionEngine._rrf(rank=1, k=60) == pytest.approx(1.0 / 61.0)


def test_asin_key_fallback_to_item_id():
    hit = H(item_id=123, asin=None, rank=1, score=1.0)
    assert BM25ClipFusionEngine._asin_key(hit) == "ITEM-123"


def test_search_passes_conn_and_uses_stage1_k():
    text = FakeTextEngine(hits=[])
    img = FakeImageEngine(hits=[])

    eng = BM25ClipFusionEngine(text_engine=text, image_engine=img)

    dummy_conn = object()
    eng.search("q", top_k=5, stage1_factor=3, conn=dummy_conn)

    assert text.last_query == "q"
    assert img.last_query == "q"

    assert text.last_top_k == 15
    assert img.last_top_k == 15

    assert text.last_conn is dummy_conn
    assert img.last_conn is dummy_conn


def test_search_fuses_rrf_and_assigns_final_rank():
    bm25_hits = [
        H(item_id=10, asin="A", rank=1, score=9.0, title="bm25-A"),
        H(item_id=20, asin="B", rank=2, score=8.0, title="bm25-B"),
    ]
    image_hits = [
        H(item_id=10, asin="A", rank=5, score=0.9, title="img-A"),
        H(item_id=30, asin="C", rank=1, score=0.95, title="img-C"),
    ]

    text = FakeTextEngine(bm25_hits)
    img = FakeImageEngine(image_hits)

    fusion = BM25ClipFusionEngine(
        text_engine=text, image_engine=img, rrf_k=60, w_text=1.0, w_image=1.0
    )

    out = fusion.search("red dress", top_k=3, stage1_factor=3, conn=object())
    assert len(out) == 3

    # rank가 1..N으로 부여되는지
    assert [r.rank for r in out] == [1, 2, 3]

    # rrf_score 내림차순 정렬인지
    scores = [r.rrf_score for r in out]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    # 기대 순서 계산:
    # A: 1/(60+1) + 1/(60+5)
    # B: 1/(60+2)
    # C: 1/(60+1)
    # => A가 가장 큼 (C는 text가 없고 image 1등, B는 text 2등)
    def rrf(rank):  # helper
        return 1.0 / (60 + rank)

    expected = {
        "A": rrf(1) + rrf(5),
        "B": rrf(2),
        "C": rrf(1),
    }
    got = {r.asin: r.rrf_score for r in out}
    for k, v in expected.items():
        assert got[k] == pytest.approx(v)


def test_image_metadata_can_fill_missing_fields_from_bm25():
    # bm25에서 title이 비어있고, image에서 title이 있으면 채워져야 함
    bm25_hits = [
        H(
            item_id=10,
            asin="A",
            rank=1,
            score=9.0,
            title=None,
            store=None,
            image_url=None,
        ),
    ]
    image_hits = [
        H(
            item_id=10,
            asin="A",
            rank=1,
            score=0.9,
            title="img-title",
            store="img-store",
            image_url="img-url",
        ),
    ]

    fusion = BM25ClipFusionEngine(
        text_engine=FakeTextEngine(bm25_hits),
        image_engine=FakeImageEngine(image_hits),
        rrf_k=60,
    )

    out = fusion.search("q", top_k=1, conn=object())
    assert len(out) == 1
    r = out[0]
    assert r.asin == "A"
    assert r.title == "img-title"
    assert r.store == "img-store"
    assert r.image_url == "img-url"


def test_apply_image_hits_keeps_best_image_rank():
    # 같은 asin이 여러 번 들어와도 더 좋은(작은) rank로 갱신되는지
    fusion = BM25ClipFusionEngine(
        text_engine=FakeTextEngine([]), image_engine=FakeImageEngine([]), rrf_k=60
    )

    fused = {}
    hits = [
        H(item_id=10, asin="A", rank=5, score=0.1),
        H(item_id=10, asin="A", rank=2, score=0.9),  # 더 좋은 rank
    ]
    fusion._apply_image_hits(fused, hits)

    assert "A" in fused
    assert fused["A"].image_rank == 2
    assert fused["A"].image_score_raw == pytest.approx(0.9)
