# tests/integration/test_personalized_search_engine.py

from __future__ import annotations

from pathlib import Path

from fashion_core.personalized_search_engine import PersonalizedSearchEngine


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
        w_text=0.0,
        w_image=1.0,
    )


## test 함수
def test_integration_no_user_no_session_returns_base(
    mini_db: Path,
    mini_bm25_pkl: Path,
    mini_faiss_index: Path,
    als_config_path: Path,
):
    # user_id도 없고 session_item_ids도 없으면 ALS를 스킵하고 base fusion 결과 그대로 나오는지
    eng = make_engine(mini_db, mini_bm25_pkl, mini_faiss_index, als_config_path)
    try:
        out = eng.search("red dress", top_k=2, user_id=None, session_item_ids=None)
        assert len(out) == 2
        # ALS X -> score_final 같은 필드는 없어야 정상
        assert "score_final" not in out[0]
        assert "als_mode" not in out[0]
    finally:
        eng.close()


def test_integration_user_id_applies_als_rerank(
    mini_db: Path,
    mini_bm25_pkl: Path,
    mini_faiss_index: Path,
    als_config_path: Path,
):
    # user_id="u1"이면 ALS가 적용되어 결과가 재정렬되는지
    eng = make_engine(mini_db, mini_bm25_pkl, mini_faiss_index, als_config_path)
    try:
        base = eng.multimodal_engine.search("red dress", top_k=2)
        base_asins = [r["asin"] for r in base]
        assert "B" in base_asins  # ALS가 끌어올릴 후보에 있어야 함
        assert "A" in base_asins

        out = eng.search("red dress", top_k=2, user_id="u1", session_item_ids=None)
        assert len(out) == 2

        # ALS 필드가 붙어야 함
        for r in out:
            assert "score_final" in r
            assert r["als_model_version"] == "v-test"
            assert r["als_mode"] == "user"

        # u1은 A를 강하게 선호하도록 만들었으니 1등이 A가 되는 걸 기대
        assert out[0]["asin"] == "A"
    finally:
        eng.close()


def test_integration_session_items_applies_als_when_user_unknown(
    mini_db: Path,
    mini_bm25_pkl: Path,
    mini_faiss_index: Path,
    als_config_path: Path,
):
    # user_id가 학습셋에 없어도(session cold) session_item_ids가 있으면 2단계(item2item)로 ALS가 적용되는지.
    eng = make_engine(mini_db, mini_bm25_pkl, mini_faiss_index, als_config_path)
    try:
        out = eng.search(
            "red dress", top_k=2, user_id="no_such_user", session_item_ids=["A"]
        )
        assert len(out) == 2

        for r in out:
            assert "score_final" in r
            assert r["als_mode"] == "session_items"

        # session에 A가 있으니 A가 1등으로 오기 쉬움
        assert out[0]["asin"] == "A"
    finally:
        eng.close()


def test_integration_fallback_when_session_invalid(
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
