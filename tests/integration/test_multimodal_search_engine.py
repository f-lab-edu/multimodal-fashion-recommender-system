# tests/integration/test_multimodal_search_engine.py

from __future__ import annotations

from pathlib import Path

from fashion_core.multimodal_search_engine import MultiModalSearchEngine


def test_integration_multimodal_search_end_to_end(
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
