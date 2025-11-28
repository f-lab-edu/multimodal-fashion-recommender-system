# recommand/als_evaluator.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import pickle

from common.als_io import load_als_jsonl

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


class ALSEvaluator:
    """
    - 학습된 ALS 모델(als_model.pkl)과 매핑(mappings.pkl)을 로드
    - train jsonl로 user_items_csr(유저 x 아이템 행렬) 생성
    - test jsonl로 Recall@k, Hit Rate 계산
    """

    def __init__(self, model_dir: Path) -> None:
        with (model_dir / "als_model.pkl").open("rb") as f:
            self.model = pickle.load(f)

        # 2) 매핑 로드
        with (model_dir / "mappings.pkl").open("rb") as f:
            mappings = pickle.load(f)

        self.user2idx: Dict[str, int] = mappings["user2idx"]
        self.item2idx: Dict[str, int] = mappings["item2idx"]
        self.idx2user: Dict[int, str] = mappings["idx2user"]
        self.idx2item: Dict[int, str] = mappings["idx2item"]

        self.user_items_csr: Optional[csr_matrix] = None

        print(f"[EVAL] 모델/매핑 로드 완료: {model_dir}")
        print(
            f"[EVAL] 유저 수: {len(self.user2idx):,}, 아이템 수: {len(self.item2idx):,}"
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
        print(f"[EVAL] user_items_csr shape: users={num_users:,}, items={num_items:,}")

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

        # 매핑에 없는 user/item은 평가 대상에서 제외
        df = df_test[
            df_test["user_id"].isin(self.user2idx)
            & df_test["item_id"].isin(self.item2idx)
        ].copy()

        if df.empty:
            print("[EVAL] 평가 대상 행이 없습니다.")
            return {
                "recall_at_k": 0.0,
                "hit_rate": 0.0,
                "map_at_k": 0.0,
                "num_users": 0,
            }

        df["user_idx"] = df["user_id"].map(self.user2idx)
        df["item_idx"] = df["item_id"].map(self.item2idx)

        # User별 정답 아이템 그룹화
        target_items_per_user: Dict[int, Set[int]] = (
            df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )

        hits = 0  # 전체 정답 중 맞춘 개수 (recall 계산용)
        total_positives = 0  # 전체 정답 개수 (recall 계산용)
        hit_users = 0  # 적어도 1개 이상 맞춘 유저 수 (hit rate 계산용)
        num_users = 0  # 평가에 사용된 유저 수

        ap_sum = 0.0  # MAP 계산용 (유저별 AP 합)
        num_users_with_pos = 0  # 정답이 1개 이상인 유저 수 (MAP 분모)

        eval_user_idxs = list(target_items_per_user.keys())
        print(f"[EVAL] 총 평가 대상 유저 수: {len(eval_user_idxs):,}")

        for user_idx, pos_items in target_items_per_user.items():
            if not pos_items:
                continue

            num_users += 1
            total_positives += len(pos_items)

            user_row = self.user_items_csr[user_idx]

            # implicit.recommend()는 [(item_id, score), ...] 리스트를 주는 게 기본이므로
            recs = self.model.recommend(
                user_idx,
                user_row,
                N=k,
                filter_already_liked_items=True,
            )

            # recs가 리스트[(item, score)]라고 가정
            # 혹시 (items, scores) 튜플이면 여기를 rec_item_idxs, _ = recs 로 바꾸면 됨
            item_ids, scores = recs
            rec_item_idxs = [int(i) for i in item_ids]
            rec_item_set = set(rec_item_idxs)

            # === Hit / Recall / HitRate 계산 ===
            inter = pos_items & rec_item_set
            if inter:
                hit_users += 1
                hits += len(inter)

            # === AP@k 계산 ===
            # ranked list 기준으로 순서대로 precision 계산
            hits_so_far = 0
            ap_user = 0.0

            for rank, item_id in enumerate(rec_item_idxs, start=1):
                if item_id in pos_items:
                    hits_so_far += 1
                    precision_at_rank = hits_so_far / rank
                    ap_user += precision_at_rank

            if len(pos_items) > 0:
                ap_user /= min(len(pos_items), k)
                ap_sum += ap_user
                num_users_with_pos += 1

        # 루프 끝난 뒤에 전체 기준으로 metric 계산
        if num_users == 0 or total_positives == 0:
            recall_at_k = 0.0
            hit_rate = 0.0
        else:
            recall_at_k = hits / total_positives if total_positives > 0 else 0.0
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
        df_test = self._load_df(test_jsonl)  # user_id, item_id, ...
        metrics = self.evaluate_df(df_test, k=k)
        print(f"[EVAL] num_users={metrics['num_users']}")
        print(f"[EVAL] Recall@{k}: {metrics['recall_at_k']:.4f}")
        print(f"[EVAL] HitRate@{k}: {metrics['hit_rate']:.4f}")
        print(f"[EVAL] MAP@{k}: {metrics['map_at_k']:.4f}")
        return metrics
