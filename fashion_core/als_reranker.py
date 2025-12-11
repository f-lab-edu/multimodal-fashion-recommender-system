from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pickle
import yaml


class ALSReRanker:
    """
    ALS 모델을 사용하여 검색 결과를 개인맞춤 재정렬
    - 1단계: user_id가 ALS 학습셋에 있으면 user→item 기반 rerank
    - 2단계: user_id는 쓸 수 없지만 session_item_ids가 있으면 item2item 기반 rerank
    - 3단계: 둘 다 없으면 ALS로는 재정렬 불가 → None 반환
    """

    def __init__(
        self,
        model_config_path: Path,
        item_key: str = "PARENT_ASIN",
        score_key: str = "score",
    ) -> None:
        self.model_config_path = Path(model_config_path)
        self.item_key = item_key
        self.score_key = score_key

        self._load_from_config()

    def _load_from_config(self) -> None:
        with self.model_config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        model_dir = Path(cfg["model_dir"])
        self.w_als = float(cfg.get("w_als", 0.3))  # ALS 점수 가중치
        self.model_version = cfg.get("model_version", "unknown")

        # ALS 모델 로드
        with (model_dir / "als_model.pkl").open("rb") as f:
            self.model = pickle.load(f)
        with (model_dir / "mappings.pkl").open("rb") as f:
            mappings = pickle.load(f)

        self.user2idx: Dict[str, int] = mappings["user2idx"]
        self.item2idx: Dict[str, int] = mappings["item2idx"]

        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

    def reload_model(self) -> None:
        """
        - 새 ALS 모델을 다시 로드할 때
        """
        self._load_from_config()

    @staticmethod
    def _minmax_norm(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        if x.size == 0:
            return x
        x_min = float(x.min())
        x_max = float(x.max())
        if x_max - x_min < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return (x - x_min) / (x_max - x_min)

    def _validate_item_idx(
        self,
        idx: int,
        context: str,
    ) -> int:
        """
        item ALS index 검증 헬퍼
        - 범위 밖이면: 인덱스 에러
        (idx가 None인 경우는 호출부에서 미리 걸러야 함)
        """
        if idx < 0 or idx >= self.item_factors.shape[0]:
            raise IndexError(
                f"[ALSReRanker] item_idx out of range in {context}: "
                f"idx={idx}, item_factors.shape={self.item_factors.shape}"
            )
        return idx

    def _compute_user_item_scores(
        self,
        user_idx: int,
        item_indices: List[Optional[int]],
    ) -> np.ndarray:
        """
        1단계: user factor가 있는 경우 user→item 점수 계산
        - cold 후보 아이템(None)은 ALS 스코어 0
        - item2idx ↔ item_factors 범위 불일치는 에러
        """
        als_scores = np.zeros(len(item_indices), dtype=np.float32)
        u_vec = self.user_factors[user_idx]

        valid_positions: List[int] = []
        valid_indices: List[int] = []
        for pos, raw_idx in enumerate(item_indices):
            if raw_idx is None:
                # cold 아이템
                continue

            item_idx = self._validate_item_idx(
                raw_idx, context=f"user_item_index[i={pos}]"
            )
            valid_positions.append(pos)
            valid_indices.append(item_idx)

        if not valid_indices:
            return als_scores

        item_mat = self.item_factors[valid_indices]  # (N, K)
        scores_vec = item_mat @ u_vec  # (N,)

        for pos, s in zip(valid_positions, scores_vec):
            als_scores[pos] = float(s)

        return als_scores

    def _compute_session_item_scores(
        self,
        session_item_ids: List[str],
        item_indices: List[Optional[int]],
    ) -> Optional[np.ndarray]:
        """
        2단계: 장바구니 session_item_ids 기반 item2item 점수 계산
        - session_item_ids → item factor 평균 → 세션 벡터
        - 세션 벡터와 각 후보 아이템 factor의 dot-product
        - 유효한 세션 아이템이 하나도 없으면 None 반환
        """
        # 세션 아이템들을 ALS index로 변환
        session_indices: List[int] = []
        for sid in session_item_ids:
            idx = self.item2idx.get(sid)
            if idx is None:
                continue
            idx = self._validate_item_idx(idx, context=f"session_item_id={sid}")
            session_indices.append(idx)

        if not session_indices:
            # 사용할 수 있는 세션 아이템이 하나도 없음
            return None

        # 세션 벡터 = 세션 내 아이템 factor 평균
        session_vecs = self.item_factors[session_indices]  # (S, K)
        s_vec = session_vecs.mean(axis=0)  # (K,)

        als_scores = np.zeros(len(item_indices), dtype=np.float32)
        for i, item_idx in enumerate(item_indices):
            if item_idx is None:
                continue
            item_idx = self._validate_item_idx(
                item_idx, context=f"candidate_item_index[i={i}]"
            )
            i_vec = self.item_factors[item_idx]
            als_scores[i] = float(s_vec @ i_vec)

        return als_scores

    def rerank(
        self,
        user_id: Optional[str],
        results: List[Dict[str, Any]],
        session_item_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        검색 결과를 ALS 기반으로 재정렬.
        반환:
            - ALS를 사용한 경우: score_final 기준으로 정렬된 results
            - ALS로 점수를 줄 수 없는 경우(3단계): None
        """
        if not results:
            return []

        # 1) base 점수
        base_scores = np.array(
            [float(r.get(self.score_key, 0.0)) for r in results],
            dtype=np.float32,
        )

        # 2) 후보 아이템들의 ALS index
        item_indices: List[Optional[int]] = []
        for r in results:
            item_id = r.get(self.item_key)
            if item_id is None:
                item_indices.append(None)
            else:
                item_indices.append(self.item2idx.get(item_id))

        als_scores: Optional[np.ndarray] = None
        als_mode: Optional[str] = None

        # 1단계: user factor가 있는 경우
        user_idx: Optional[int] = None
        if user_id is not None:
            user_idx = self.user2idx.get(user_id)

        if user_idx is not None:
            als_scores = self._compute_user_item_scores(user_idx, item_indices)
            als_mode = "user"

        # 2단계: user는 못 쓰지만 session_item_ids가 있는 경우
        if als_scores is None and session_item_ids:
            session_scores = self._compute_session_item_scores(
                session_item_ids, item_indices
            )
            if session_scores is not None:
                als_scores = session_scores
                als_mode = "session_items"

        # 3단계: ALS로 점수를 줄 수 없는 경우
        if als_scores is None:
            return None

        # 4) base + ALS 스코어 결합
        base_norm = self._minmax_norm(base_scores)
        als_norm = self._minmax_norm(als_scores)

        w = float(self.w_als)
        final_scores = (1.0 - w) * base_norm + w * als_norm

        for r, b, a, f in zip(results, base_norm, als_norm, final_scores):
            r["score_base_norm"] = float(b)
            r["score_als_norm"] = float(a)
            r["score_final"] = float(f)
            r["als_model_version"] = self.model_version
            r["als_mode"] = als_mode  # "user" or "session_items"

        results_sorted = sorted(
            results,
            key=lambda x: x["score_final"],
            reverse=True,
        )

        if top_k is not None:
            results_sorted = results_sorted[:top_k]

        return results_sorted
