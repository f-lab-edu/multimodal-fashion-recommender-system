# tests/unit/test_als_reranker.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pickle
import pytest
import yaml

from fashion_core.als_reranker import ALSReRanker


@pytest.fixture
def als_artifacts(tmp_path: Path) -> Dict[str, Path]:
    """
    tmp_path 아래에
    - model_dir/als_model.pkl
    - model_dir/mappings.pkl
    - config.yaml
    를 만들어서 ALSReRanker가 실제로 로드하도록 구성
    """
    model_dir = tmp_path / "als_model_dir"
    model_dir.mkdir(parents=True)

    # 작은 행렬로 테스트
    # users 1명, items 4개, latent dim=2

    user_factors = np.array([[1.0, 0.0]], dtype=np.float32)  # user_idx=0
    item_factors = np.array(
        [
            [1.0, 0.0],  # idx 0 -> "A"
            [0.0, 1.0],  # idx 1 -> "B"
            [2.0, 0.0],  # idx 2 -> "C"
            [0.5, 0.5],  # idx 3 -> "D"
        ],
        dtype=np.float32,
    )

    fake_model = SimpleNamespace(user_factors=user_factors, item_factors=item_factors)
    mappings = {
        "user2idx": {"u1": 0},
        "item2idx": {"A": 0, "B": 1, "C": 2, "D": 3},
    }

    with (model_dir / "als_model.pkl").open("wb") as f:
        pickle.dump(fake_model, f)

    with (model_dir / "mappings.pkl").open("wb") as f:
        pickle.dump(mappings, f)

    cfg = {
        "model_dir": str(model_dir),
        "w_als": 0.5,
        "model_version": "v-test",
    }
    cfg_path = tmp_path / "als_config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    return {"cfg_path": cfg_path, "model_dir": model_dir}


@pytest.fixture
def reranker(als_artifacts: Dict[str, Path]) -> ALSReRanker:
    # config 경로로 ALSReRanker 인스턴스를 생성해 각 테스트에 제공
    return ALSReRanker(model_config_path=als_artifacts["cfg_path"])


# helpers
def test_minmax_norm_empty_returns_empty():
    # _minmax_norm([]) 입력이 빈 배열이면 빈 배열 그대로 반환하는지
    x = np.array([], dtype=np.float32)
    y = ALSReRanker._minmax_norm(x)
    assert y.size == 0


def test_minmax_norm_constant_returns_zeros():
    # 전부 동일한 배열이면 분모가 0에 가까워져서, 코드가 의도한 대로 전부 0으로 반환
    x = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    y = ALSReRanker._minmax_norm(x)
    assert np.allclose(y, np.zeros_like(x))


# rerank() 동작 테스트
def test_rerank_returns_empty_list_when_results_empty(reranker: ALSReRanker):
    # result=[] 이면 [] 반환하는지
    out = reranker.rerank(user_id="u1", results=[])
    assert out == []


def test_rerank_returns_none_when_no_user_and_no_session(reranker: ALSReRanker):
    # user_id=None이고 session_item_ids=None이면 ALS로 점수를 만들 수 없어서 None 반환 하는지
    results = [
        {"PARENT_ASIN": "A", "score": 0.9},
        {"PARENT_ASIN": "B", "score": 0.8},
    ]
    out = reranker.rerank(user_id=None, results=results, session_item_ids=None)
    assert out is None


def test_rerank_user_mode_adds_fields_and_sorts(reranker: ALSReRanker):
    # user 존재시 user->item 점수 계산, 필드 추가, 점수 정렬
    results = [
        {"PARENT_ASIN": "A", "score": 0.9},  # ALS dot = 1.0
        {"PARENT_ASIN": "B", "score": 0.8},  # ALS dot = 0.0
        {"PARENT_ASIN": "C", "score": 0.7},  # ALS dot = 2.0
    ]

    out = reranker.rerank(user_id="u1", results=results, top_k=None)
    assert out is not None
    assert len(out) == 3

    # 모든 결과에 부가 필드가 붙는지
    for r in out:
        assert "score_base_norm" in r
        assert "score_als_norm" in r
        assert "score_final" in r
        assert r["als_model_version"] == "v-test"
        assert r["als_mode"] == "user"

    # score_final 내림차순 정렬
    finals = [r["score_final"] for r in out]
    assert all(finals[i] >= finals[i + 1] for i in range(len(finals) - 1))

    # top item이 유효한 후보 중 하나인지
    assert out[0]["PARENT_ASIN"] in {"A", "B", "C"}


def test_rerank_user_mode_respects_top_k(reranker: ALSReRanker):
    # user 모드 top-k 만큼 결과가 만들어 지는지
    results = [
        {"PARENT_ASIN": "A", "score": 0.9},
        {"PARENT_ASIN": "B", "score": 0.8},
        {"PARENT_ASIN": "C", "score": 0.7},
    ]
    out = reranker.rerank(user_id="u1", results=results, top_k=2)
    assert out is not None
    assert len(out) == 2


def test_rerank_session_items_mode_when_user_unknown(reranker: ALSReRanker):
    # cold user, session_items_ids 는 유효 -> 2단계 로직 -> als_mode == "session_items" 인지
    # user 없음 → session_item_ids 유효 시 item2item(세션 평균 벡터) 모드로 동작
    results = [
        {"PARENT_ASIN": "A", "score": 0.2},
        {"PARENT_ASIN": "B", "score": 0.2},
        {"PARENT_ASIN": "D", "score": 0.2},
    ]
    # user_id는 학습셋에 없지만 session_item_ids는 유효 -> session_items 모드
    out = reranker.rerank(
        user_id="no_such_user", results=results, session_item_ids=["A", "C"]
    )
    assert out is not None
    assert len(out) == 3
    assert all(r["als_mode"] == "session_items" for r in out)


def test_rerank_session_items_returns_none_when_session_items_all_invalid(
    reranker: ALSReRanker,
):
    # user도 없고 session 아이템도 매핑에 없으면 2단계 실패 -> None 리턴
    results = [
        {"PARENT_ASIN": "A", "score": 0.9},
        {"PARENT_ASIN": "B", "score": 0.8},
    ]
    # session_item_ids가 전부 item2idx에 없음 -> 2단계도 실패 -> None
    out = reranker.rerank(
        user_id="no_such_user", results=results, session_item_ids=["X", "Y"]
    )
    assert out is None


def test_rerank_handles_cold_items_as_zero_als_score(reranker: ALSReRanker):
    """
    후보 중 ALS 매핑 없는 아이템(None)은 ALS 점수 0 처리하고 계속 진행
    """

    # "X"는 item2idx에 없어서 cold(None) 처리
    results = [
        {"PARENT_ASIN": "A", "score": 0.9},
        {"PARENT_ASIN": "X", "score": 0.1},
    ]
    out = reranker.rerank(user_id="u1", results=results)
    assert out is not None
    assert len(out) == 2
    # cold 아이템도 결과에 남고, 필드가 붙는지만 확인
    x = [r for r in out if r["PARENT_ASIN"] == "X"][0]
    assert "score_als_norm" in x


def test_rerank_raises_when_item_index_out_of_range(tmp_path: Path):
    """
    item2idx와 item_factors 불일치 시 _validate_item_idx가 IndexError를 던짐
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    fake_model = SimpleNamespace(
        user_factors=np.array([[1.0, 0.0]], dtype=np.float32),
        item_factors=np.array([[1.0, 0.0]], dtype=np.float32),  # item 1개뿐
    )
    mappings = {
        "user2idx": {"u1": 0},
        "item2idx": {"A": 999},  # 범위 밖
    }

    with (model_dir / "als_model.pkl").open("wb") as f:
        pickle.dump(fake_model, f)
    with (model_dir / "mappings.pkl").open("wb") as f:
        pickle.dump(mappings, f)

    cfg_path = tmp_path / "cfg.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"model_dir": str(model_dir), "w_als": 0.5, "model_version": "v"}, f
        )

    rr = ALSReRanker(model_config_path=cfg_path)

    results = [{"PARENT_ASIN": "A", "score": 1.0}]
    with pytest.raises(IndexError):
        rr.rerank(user_id="u1", results=results)
