from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pickle
import yaml


class ALSReRanker:
    """
    ALS 모델을 사용하여 검색 결과를 개인맞춤 재정렬
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

    def _compute_als_scores(
        self,
        user_idx: int,
        item_indices: List[Optional[int]],
    ) -> np.ndarray:
        als_scores = np.zeros(len(item_indices), dtype=np.float32)
        u_vec = self.user_factors[user_idx]

        for i, item_idx in enumerate(item_indices):
            if item_idx is None:
                als_scores[i] = 0.0
                continue
            if item_idx < 0 or item_idx >= self.item_factors.shape[0]:
                als_scores[i] = 0.0
                continue
            i_vec = self.item_factors[item_idx]
            als_scores[i] = float(u_vec @ i_vec)

        return als_scores

    def rerank(
        self,
        user_id: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not results:
            return results

        user_idx = self.user2idx.get(user_id)
        if user_idx is None:
            # cold-start 유저 -> 원래 점수 그대로
            return results if top_k is None else results[:top_k]

        base_scores = np.array(
            [float(r.get(self.score_key, 0.0)) for r in results],
            dtype=np.float32,
        )

        item_indices: List[Optional[int]] = []
        for r in results:
            item_id = r.get(self.item_key)
            if item_id is None:
                item_indices.append(None)
            else:
                item_indices.append(self.item2idx.get(item_id))

        als_scores = self._compute_als_scores(user_idx, item_indices)

        base_norm = self._minmax_norm(base_scores)
        als_norm = self._minmax_norm(als_scores)

        w = float(self.w_als)
        final_scores = (1.0 - w) * base_norm + w * als_norm

        for r, b, a, f in zip(results, base_norm, als_norm, final_scores):
            r["score_base_norm"] = float(b)
            r["score_als_norm"] = float(a)
            r["score_final"] = float(f)
            r["als_model_version"] = self.model_version  # 디버깅 용

        results_sorted = sorted(
            results,
            key=lambda x: x["score_final"],
            reverse=True,
        )

        if top_k is not None:
            results_sorted = results_sorted[:top_k]

        return results_sorted
