# recommand/als_evaluator.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import logging
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import pickle

from common.als_io import load_als_jsonl

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

logger = logging.getLogger(__name__)


class ALSEvaluator:
    """
    - 학습된 ALS 모델(als_model.pkl)과 매핑(mappings.pkl)을 로드
    - train jsonl로 user_items_csr(유저 x 아이템 행렬) 생성
    - test jsonl로 Recall@k, Hit Rate, MAP@k 계산
    """

    def __init__(self, model_dir: Path) -> None:
        with (model_dir / "als_model.pkl").open("rb") as f:
            self.model = pickle.load(f)

        with (model_dir / "mappings.pkl").open("rb") as f:
            mappings = pickle.load(f)

        self.user2idx: Dict[str, int] = mappings["user2idx"]
        self.item2idx: Dict[str, int] = mappings["item2idx"]
        self.idx2user: Dict[int, str] = mappings["idx2user"]
        self.idx2item: Dict[int, str] = mappings["idx2item"]

        self.user_items_csr: Optional[csr_matrix] = None

        logger.info("[EVAL] 모델/매핑 로드 완료: %s", model_dir)
        logger.info(
            "[EVAL] 유저 수: %s, 아이템 수: %s",
            f"{len(self.user2idx):,}",
            f"{len(self.item2idx):,}",
        )

    def _load_df(self, path: Path) -> pd.DataFrame:
        df = load_als_jsonl(path)
        df = df.rename(columns={"PARENT_ASIN": "item_id"})
        return df

    def build_user_items_from_train(self, train_jsonl: Path) -> None:
        """
        train jsonl (학습에 사용한 데이터와 동일 스키마)를 읽어서
        users x items CSR 행렬을 생성.
        """
        df = self._load_df(train_jsonl)

        # 학습된 매핑에 존재하는 user/item만 사용
        df = df[
            df["user_id"].isin(self.user2idx) & df["item_id"].isin(self.item2idx)
        ].copy()

        if df.empty:
            raise RuntimeError("train 데이터에서 유효한 user/item이 없습니다.")

        df["user_idx"] = df["user_id"].map(self.user2idx).astype(np.int32)
        df["item_idx"] = df["item_id"].map(self.item2idx).astype(np.int32)

        user_idx = df["user_idx"].to_numpy()
        item_idx = df["item_idx"].to_numpy()

        data = np.ones(len(df), dtype=np.float32)

        num_users = self.model.user_factors.shape[0]
        num_items = self.model.item_factors.shape[0]

        mat = coo_matrix(
            (data, (user_idx, item_idx)),
            shape=(num_users, num_items),
            dtype=np.float32,
        ).tocsr()

        self.user_items_csr = mat
        logger.info(
            "[EVAL] user_items_csr shape: users=%s, items=%s",
            f"{num_users:,}",
            f"{num_items:,}",
        )

    # ----------------- 내부 헬퍼들 (복잡도 감소용) ----------------- #

    def _prepare_eval_targets(self, df_test: pd.DataFrame) -> Dict[int, Set[int]]:
        """
        df_test에서 평가 대상 유저별 정답 item_idx 집합을 생성.
        매핑에 없는 user/item은 제외.
        """
        df = df_test[
            df_test["user_id"].isin(self.user2idx)
            & df_test["item_id"].isin(self.item2idx)
        ].copy()

        if df.empty:
            logger.warning("[EVAL] 평가 대상 행이 없습니다.")
            return {}

        df["user_idx"] = df["user_id"].map(self.user2idx)
        df["item_idx"] = df["item_id"].map(self.item2idx)

        target_items_per_user: Dict[int, Set[int]] = (
            df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )
        logger.info(
            "[EVAL] 총 평가 대상 유저 수: %s",
            f"{len(target_items_per_user):,}",
        )
        return target_items_per_user

    def _recommend_for_user(self, user_idx: int, k: int) -> np.ndarray:
        """
        단일 유저에 대해 ALS 추천 item index 리스트 반환.
        """
        if self.user_items_csr is None:
            raise RuntimeError(
                "user_items_csr가 없습니다. build_user_items_from_train() 먼저 호출 필요."
            )

        user_row = self.user_items_csr[user_idx]

        # implicit.recommend()의 기본 반환은 (item_ids, scores)
        item_ids, _ = self.model.recommend(
            user_idx,
            user_row,
            N=k,
            filter_already_liked_items=True,
        )
        # ndarray 또는 리스트 모두 int로 캐스팅해서 사용
        return np.asarray(item_ids, dtype=int)

    def _evaluate_single_user(
        self,
        user_idx: int,
        pos_items: Set[int],
        k: int,
    ) -> Tuple[int, int, int, float, bool]:
        """
        단일 유저에 대한:
        - hits_user: 정답 중 맞춘 개수
        - total_pos: 정답 개수
        - hit_user_flag: 적어도 1개 맞췄으면 1, 아니면 0
        - ap_user: 해당 유저의 AP@k
        - has_pos_flag: 정답이 있는 유저이면 True
        """
        if not pos_items:
            return 0, 0, 0, 0.0, False

        rec_item_idxs = self._recommend_for_user(user_idx, k)
        rec_item_set = set(rec_item_idxs)

        total_pos = len(pos_items)
        inter_size = len(pos_items & rec_item_set)
        hit_user_flag = 1 if inter_size > 0 else 0

        # AP@k 계산
        hits_so_far = 0
        ap_user = 0.0
        for rank, item_id in enumerate(rec_item_idxs, start=1):
            if item_id in pos_items:
                hits_so_far += 1
                precision_at_rank = hits_so_far / rank
                ap_user += precision_at_rank

        ap_user /= min(total_pos, k)
        return inter_size, total_pos, hit_user_flag, ap_user, True

    # ---------------------- 메인 평가 함수들 ---------------------- #

    def evaluate_df(self, df_test: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        """
        df_test: user_id, item_id, rating, timestamp ...
        각 user_id에 대해 df_test 안의 item_id를 '정답 세트'로 보고
        ALS 추천 top-k 안에 들어가는 비율 및 MAP@k 계산.
        """
        if self.user_items_csr is None:
            raise RuntimeError(
                "user_items_csr가 없습니다. build_user_items_from_train() 먼저 호출 필요."
            )

        target_items_per_user = self._prepare_eval_targets(df_test)
        if not target_items_per_user:
            return {
                "recall_at_k": 0.0,
                "hit_rate": 0.0,
                "map_at_k": 0.0,
                "num_users": 0,
            }

        hits = 0
        total_positives = 0
        hit_users = 0
        num_users = 0

        ap_sum = 0.0
        num_users_with_pos = 0

        for user_idx, pos_items in target_items_per_user.items():
            (
                hits_user,
                total_pos_user,
                hit_user_flag,
                ap_user,
                has_pos,
            ) = self._evaluate_single_user(user_idx, pos_items, k)

            if not has_pos:
                continue

            num_users += 1
            hits += hits_user
            total_positives += total_pos_user
            hit_users += hit_user_flag

            ap_sum += ap_user
            num_users_with_pos += 1

        if num_users == 0 or total_positives == 0:
            recall_at_k = 0.0
            hit_rate = 0.0
        else:
            recall_at_k = hits / total_positives
            hit_rate = hit_users / num_users

        if num_users_with_pos == 0:
            map_at_k = 0.0
        else:
            map_at_k = ap_sum / num_users_with_pos

        return {
            "recall_at_k": float(recall_at_k),
            "hit_rate": float(hit_rate),
            "map_at_k": float(map_at_k),
            "num_users": int(num_users),
        }

    def evaluate_on_test(self, test_jsonl: Path, k: int = 10) -> Dict[str, float]:
        df_test = self._load_df(test_jsonl)
        metrics = self.evaluate_df(df_test, k=k)

        logger.info("[EVAL] num_users=%d", metrics["num_users"])
        logger.info("[EVAL] Recall@%d: %.4f", k, metrics["recall_at_k"])
        logger.info("[EVAL] HitRate@%d: %.4f", k, metrics["hit_rate"])
        logger.info("[EVAL] MAP@%d: %.4f", k, metrics["map_at_k"])

        return metrics
