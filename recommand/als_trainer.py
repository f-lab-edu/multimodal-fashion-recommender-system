# recommand/als_trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, Set

import logging
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import implicit
import pickle

import sqlite3
import faiss

from common.als_io import load_als_jsonl

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

logger = logging.getLogger(__name__)


class ALSTrainer:
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 1e-3,
        iterations: int = 20,
        alpha: float = 40.0,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state

        self._rng = np.random.default_rng(self.random_state)

    def _load_df(self, path: Path) -> pd.DataFrame:
        df = load_als_jsonl(path)
        df = df.rename(columns={"PARENT_ASIN": "item_id"})

        logger.info("[TRAIN] 입력 df 행 수: %s", f"{len(df):,}")
        logger.info("[TRAIN] 유저 수: %s", f"{df['user_id'].nunique():,}")
        logger.info("[TRAIN] 아이템 수: %s", f"{df['item_id'].nunique():,}")
        return df

    def _build_confidence(self, df: pd.DataFrame) -> pd.Series:
        """각 (user, item) 상호작용을 implicit ALS에서 쓸 가중치로 변환."""
        base_rating = df["rating"].clip(lower=1.0, upper=5.0)
        verified_bonus = np.where(df["verified_purchase"], 1.2, 1.0)
        helpful_bonus = 1.0 + 0.1 * np.log1p(df["helpful_vote"])
        r_ui_confidence = base_rating * verified_bonus * helpful_bonus
        return r_ui_confidence.clip(upper=20.0).astype(np.float32)

    def _build_mappings(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
        # 문자열/긴 ID들을 0 ~ N-1 정수 인덱스로 변환
        users = df["user_id"].unique()
        items = df["item_id"].unique()

        user_to_index = {user_id: idx for idx, user_id in enumerate(users)}
        item_to_index = {item_id: idx for idx, item_id in enumerate(items)}

        df_mapped = df.copy()
        df_mapped["user_idx"] = df_mapped["user_id"].map(user_to_index).astype(np.int32)
        df_mapped["item_idx"] = df_mapped["item_id"].map(item_to_index).astype(np.int32)

        return df_mapped, user_to_index, item_to_index

    def _build_user_item_matrix(
        self, df_mapped: pd.DataFrame, r_ui_confidence: pd.Series
    ) -> csr_matrix:
        """(user_idx, item_idx, R_ui)에서 user–item 희소행렬 생성."""
        user_idx_array = df_mapped["user_idx"].to_numpy()
        item_idx_array = df_mapped["item_idx"].to_numpy()
        confidence_values = r_ui_confidence.to_numpy()

        num_users = int(df_mapped["user_idx"].max()) + 1
        num_items = int(df_mapped["item_idx"].max()) + 1

        confidence_matrix = coo_matrix(
            (confidence_values, (user_idx_array, item_idx_array)),
            shape=(num_users, num_items),
            dtype=np.float32,
        ).tocsr()

        logger.info(
            "[TRAIN] matrix shape: users=%s, items=%s",
            f"{num_users:,}",
            f"{num_items:,}",
        )
        return confidence_matrix

    def _build_item_init_from_faiss(
        self,
        item_to_index: Dict[str, int],
        image_index_path: Path,
        parent_asin_to_item_id: Dict[str, int],
    ) -> np.ndarray:
        """이미지 임베딩 기반 item factor 초기값 생성."""
        logger.info("[INIT] FAISS 인덱스 로드: %s", image_index_path)
        index = faiss.read_index(str(image_index_path))

        num_vectors = index.ntotal
        dim = index.d
        logger.info("[INIT] FAISS vectors: n=%s, dim=%s", f"{num_vectors:,}", dim)

        base_index = index
        if hasattr(index, "index"):
            base_index = index.index

        try:
            vectors = base_index.reconstruct_n(0, num_vectors)
        except Exception as exc:
            if hasattr(base_index, "xb"):
                vectors = faiss.vector_to_array(base_index.xb).reshape(num_vectors, dim)
            else:
                raise RuntimeError(
                    f"FAISS 인덱스 타입에서 벡터를 복원(reconstruct)할 수 없습니다: {type(base_index)}"
                ) from exc

        # FAISS 내부 id → 위치 인덱스 매핑
        faiss_ids = faiss.vector_to_array(index.id_map)
        faiss_id_to_position: Dict[int, int] = {
            int(faiss_id): pos for pos, faiss_id in enumerate(faiss_ids)
        }

        # 우리 ALS의 item_id → FAISS 벡터 매핑
        num_items = len(item_to_index)
        image_embedding_matrix = np.zeros((num_items, dim), dtype=np.float32)

        missing_count = 0
        matched_count = 0

        for item_id, col_idx in item_to_index.items():
            faiss_item_id = parent_asin_to_item_id.get(item_id)
            if faiss_item_id is None:
                missing_count += 1
                continue

            position = faiss_id_to_position.get(int(faiss_item_id))
            if position is None:
                missing_count += 1
                continue

            image_embedding_matrix[col_idx] = vectors[position]
            matched_count += 1

        logger.info(
            "[INIT] 이미지 임베딩 매칭 완료: matched=%s, missing=%s",
            f"{matched_count:,}",
            f"{missing_count:,}",
        )

        # 결측/정규화/차원 축소
        row_norms = np.linalg.norm(image_embedding_matrix, axis=1)
        zero_mask = row_norms < 1e-8
        if zero_mask.any():
            logger.info(
                "[INIT] 이미지 없음 → 랜덤 초기화 아이템 수: %s",
                f"{zero_mask.sum():,}",
            )
            image_embedding_matrix[zero_mask] = self._rng.normal(
                scale=0.01,
                size=(zero_mask.sum(), dim),
            )

        norms = np.linalg.norm(image_embedding_matrix, axis=1, keepdims=True) + 1e-8
        image_embedding_matrix = image_embedding_matrix / norms

        k = self.factors
        projection_matrix = self._rng.normal(scale=0.01, size=(dim, k)).astype(
            np.float32
        )

        item_factors_init = image_embedding_matrix @ projection_matrix

        item_norms = np.linalg.norm(item_factors_init, axis=1, keepdims=True) + 1e-8
        item_factors_init = item_factors_init / item_norms

        logger.info("[INIT] 이미지 기반 item 초기값(V_init) 생성 완료")
        return item_factors_init.astype(np.float32)

    def _als_step(
        self,
        confidence_matrix: csr_matrix,
        binary_interaction_matrix: csr_matrix,
        fixed_factors: np.ndarray,
    ) -> np.ndarray:
        """
        ALS 한 스텝 (user 또는 item 행렬 업데이트)

        - 유저 업데이트 시: fixed_factors = item_factors (Y)
        - 아이템 업데이트 시: fixed_factors = user_factors (X)
        """
        num_rows = confidence_matrix.shape[0]
        num_factors = fixed_factors.shape[1]

        other_factors = fixed_factors
        other_factors_t_other_factors = other_factors.T @ other_factors
        updated_factors = np.zeros((num_rows, num_factors), dtype=np.float32)

        identity = np.eye(num_factors, dtype=np.float32)
        reg = self.regularization

        c_indptr = confidence_matrix.indptr
        c_indices = confidence_matrix.indices
        c_data = confidence_matrix.data

        r_data = binary_interaction_matrix.data

        for row_idx in range(num_rows):
            start, end = c_indptr[row_idx], c_indptr[row_idx + 1]
            cols = c_indices[start:end]
            c_ui_vals = c_data[start:end]
            r_ui_vals = r_data[start:end]

            if len(cols) == 0:
                updated_factors[row_idx] = 0.0
                continue

            A = other_factors_t_other_factors.copy()
            b = np.zeros(num_factors, dtype=np.float32)

            for item_idx, c_ui, r_ui in zip(cols, c_ui_vals, r_ui_vals):
                y_i = other_factors[item_idx]
                A += (c_ui - 1.0) * np.outer(y_i, y_i)
                b += (c_ui * r_ui) * y_i

            A += reg * identity
            updated_factors[row_idx] = np.linalg.solve(A, b)

        return updated_factors

    def _load_parentasin_to_itemid(self, db_path: Path) -> Dict[str, int]:
        """
        DB의 items 테이블에서 (id, parent_asin)를 읽어서
        parent_asin(str) → id(int) 딕셔너리 생성.
        """
        logger.info("[INIT] DB에서 parent_asin → items.id 매핑 로드: %s", db_path)
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, parent_asin
                FROM items
                WHERE parent_asin IS NOT NULL AND parent_asin != ''
                """
            )
            rows = cur.fetchall()

        mapping: Dict[str, int] = {}
        for item_id, parent_asin in rows:
            mapping[str(parent_asin)] = int(item_id)

        logger.info("[INIT] 매핑 개수: %s", f"{len(mapping):,}")
        return mapping

    def _build_val_user_pos(
        self,
        df_val: pd.DataFrame,
        user_to_index: Dict[str, int],
        item_to_index: Dict[str, int],
    ) -> Dict[int, Set[int]]:
        """validation용 유저별 정답 아이템 집합 만들기."""
        df = df_val.copy()
        df = df[
            df["user_id"].isin(user_to_index) & df["item_id"].isin(item_to_index)
        ].copy()

        if df.empty:
            logger.warning("[EARLY] val 데이터에서 유효한 user/item이 없습니다.")
            return {}

        df["user_idx"] = df["user_id"].map(user_to_index).astype(int)
        df["item_idx"] = df["item_id"].map(item_to_index).astype(int)

        user_positive_items: Dict[int, Set[int]] = {}
        for user_idx, group in df.groupby("user_idx"):
            user_positive_items[int(user_idx)] = {
                int(item_idx) for item_idx in group["item_idx"]
            }

        logger.info(
            "[EARLY] val users for early stopping: %s",
            f"{len(user_positive_items):,}",
        )
        return user_positive_items

    def _hit_rate_at_k(
        self,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        val_user_positive_items: Dict[int, Set[int]],
        k: int,
    ) -> float:
        """주어진 user/item factors에 대해 HitRate@K 계산."""
        if not val_user_positive_items:
            return 0.0

        num_items = item_factors.shape[0]
        hit_users = 0
        num_users_eval = 0

        for user_idx, pos_items in val_user_positive_items.items():
            if not pos_items:
                continue

            scores_u = user_factors[user_idx] @ item_factors.T
            if k >= num_items:
                topk_idx = np.arange(num_items)
            else:
                topk_idx = np.argpartition(-scores_u, k)[:k]

            if pos_items & set(topk_idx):
                hit_users += 1
            num_users_eval += 1

        if num_users_eval == 0:
            return 0.0
        return hit_users / num_users_eval

    def _train_als_with_item_init(
        self,
        r_ui_csr: csr_matrix,
        item_init: np.ndarray,
        val_user_positive_items: Optional[Dict[int, Set[int]]] = None,
        eval_k: Optional[int] = 10,
        patience: Optional[int] = 3,
    ) -> implicit.als.AlternatingLeastSquares:
        """주어진 item_init을 사용해 커스텀 ALS 학습."""
        num_users, _ = r_ui_csr.shape
        k = self.factors

        logger.info(
            "[TRAIN] 커스텀 ALS 학습 시작 (item_init shape=%s)", item_init.shape
        )

        # confidence matrix
        confidence_matrix = r_ui_csr.copy()
        confidence_matrix.data = self.alpha * confidence_matrix.data
        confidence_matrix.data += 1.0
        confidence_matrix = confidence_matrix.tocsr()

        # binary interaction matrix
        binary_interaction_matrix = r_ui_csr.copy()
        binary_interaction_matrix.data[:] = 1.0

        user_factors = self._rng.normal(scale=0.01, size=(num_users, k)).astype(
            np.float32
        )
        item_factors = item_init.astype(np.float32)

        # === early stopping을 쓸지 여부 결정 ===
        use_early_stopping = (
            bool(val_user_positive_items)  # None 또는 {} 이면 False
            and eval_k is not None
            and patience is not None
        )

        best_hit = -1.0
        best_user_factors = None
        best_item_factors = None
        no_improve = 0

        for iteration in range(self.iterations):
            logger.info(
                "[TRAIN] ALS iter %d/%d - user 업데이트",
                iteration + 1,
                self.iterations,
            )
            user_factors = self._als_step(
                confidence_matrix, binary_interaction_matrix, item_factors
            )

            logger.info(
                "[TRAIN] ALS iter %d/%d - item 업데이트",
                iteration + 1,
                self.iterations,
            )
            item_factors = self._als_step(
                confidence_matrix.T.tocsr(),
                binary_interaction_matrix.T.tocsr(),
                user_factors,
            )

            # === early stopping 사용 시에만 HitRate 계산 ===
            if use_early_stopping:
                hit = self._hit_rate_at_k(
                    user_factors, item_factors, val_user_positive_items, k=eval_k
                )
                logger.info(
                    "[EARLY] iter %d - val HitRate@%d = %.4f",
                    iteration + 1,
                    eval_k,
                    hit,
                )

                if hit > best_hit + 1e-6:
                    best_hit = hit
                    best_user_factors = user_factors.copy()
                    best_item_factors = item_factors.copy()
                    no_improve = 0
                    logger.info("[EARLY] best 갱신 (HitRate=%.4f)", best_hit)
                else:
                    no_improve += 1
                    logger.info("[EARLY] 개선 없음 (%d/%d)", no_improve, patience)
                    if no_improve >= patience:
                        logger.info("[EARLY] Early stopping 발동")
                        break

        if (
            use_early_stopping
            and best_user_factors is not None
            and best_item_factors is not None
        ):
            logger.info("[EARLY] 최종 best HitRate@%d = %.4f", eval_k, best_hit)
            user_factors_final = best_user_factors
            item_factors_final = best_item_factors
        else:
            user_factors_final = user_factors
            item_factors_final = item_factors

        logger.info("[TRAIN] 커스텀 ALS 학습 완료")

        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        model.user_factors = user_factors_final.astype(np.float32)
        model.item_factors = item_factors_final.astype(np.float32)

        model._user_norms = None
        model._item_norms = None
        model._YtY = None
        model._XtX = None

        return model

    def train_and_save(
        self,
        jsonl_path: Path,
        out_dir: Path,
        image_index_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        val_jsonl_path: Optional[Path] = None,
        eval_k: Optional[int] = 10,
        patience: Optional[int] = 3,
    ) -> None:
        df = self._load_df(jsonl_path)
        r_ui_confidence = self._build_confidence(df)

        df_mapped, user_to_index, item_to_index = self._build_mappings(df)
        r_ui_csr = self._build_user_item_matrix(df_mapped, r_ui_confidence)

        val_user_positive_items: Optional[Dict[int, Set[int]]] = None
        if val_jsonl_path is not None:
            logger.info("[EARLY] val_jsonl 로드: %s", val_jsonl_path)
            df_val = self._load_df(val_jsonl_path)
            val_user_positive_items = self._build_val_user_pos(
                df_val, user_to_index, item_to_index
            )

        if image_index_path is not None:
            if db_path is None:
                raise ValueError(
                    "image_index_path를 쓰려면 db_path도 함께 넘겨줘야 합니다."
                )

            parent_asin_to_item_id = self._load_parentasin_to_itemid(db_path)

            item_init = self._build_item_init_from_faiss(
                item_to_index=item_to_index,
                image_index_path=image_index_path,
                parent_asin_to_item_id=parent_asin_to_item_id,
            )
            logger.info("[TRAIN] 이미지 임베딩 기반 item 초기값 사용")
        else:
            num_items = len(item_to_index)
            logger.info(
                "[TRAIN] 이미지 초기값 없음 → 랜덤 item 초기값으로 커스텀 ALS 사용 (num_items=%d)",
                num_items,
            )
            item_init = self._rng.normal(
                scale=0.01,
                size=(num_items, self.factors),
            ).astype(np.float32)

            norms = np.linalg.norm(item_init, axis=1, keepdims=True) + 1e-8
            item_init = item_init / norms

        # === 여기서 eval_k / patience를 그대로 넘겨줌
        #     (val_user_positive_items가 없으면 _train_als_with_item_init 안에서 무시됨)
        model = self._train_als_with_item_init(
            r_ui_csr=r_ui_csr,
            item_init=item_init,
            val_user_positive_items=val_user_positive_items,
            eval_k=eval_k,
            patience=patience,
        )

        index_to_user = {idx: user_id for user_id, idx in user_to_index.items()}
        index_to_item = {idx: item_id for item_id, idx in item_to_index.items()}

        out_dir.mkdir(parents=True, exist_ok=True)

        with (out_dir / "als_model.pkl").open("wb") as f:
            pickle.dump(model, f)

        with (out_dir / "mappings.pkl").open("wb") as f:
            pickle.dump(
                {
                    "user2idx": user_to_index,
                    "item2idx": item_to_index,
                    "idx2user": index_to_user,
                    "idx2item": index_to_item,
                },
                f,
            )

        logger.info("[TRAIN] 저장 완료: %s", out_dir)
