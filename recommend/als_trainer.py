# recommand/als_trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import logging
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares as ALS
import pickle

import sqlite3
import faiss

from common.als_io import load_als_jsonl

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
logger = logging.getLogger(__name__)


class ALSTrainer:
    def __init__(
        self,
        factors: int = 20,
        regularization: float = 1e-3,
        iterations: int = 20,
        alpha: float = 40.0,
        random_state: int = 42,
    ) -> None:
        """
         implicit.als.AlternatingLeastSquares 기반 ALS Trainer

        - image_index_path가 주어지면: 이미지 임베딩 → item_factors warm start
        - 아니면: 랜덤 item_factors로 warm start
        """
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        # 이미지 임베딩 랜덤 보정 등에 사용할 RNG
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
        # 문자열로된 ID들을 0 ~ N-1 정수 인덱스로 변환
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
        """(user_idx, item_idx, R_ui)에서 user–item 행렬 생성"""
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
            "[TRAIN] matrix shape: users=%d, items=%d",
            num_users,
            num_items,
        )

        return confidence_matrix

    def _build_item_init_from_images(
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
        if hasattr(index, "id_map"):
            faiss_ids = faiss.vector_to_array(index.id_map)
        else:
            # IDMap 이 아니면 0..n-1 이라고 가정
            faiss_ids = np.arange(num_vectors, dtype=np.int64)

        faiss_id_to_position: Dict[int, int] = {
            int(faiss_id): pos for pos, faiss_id in enumerate(faiss_ids)
        }

        # ALS 에서 item_id -> Faiss 벡터 매핑
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

        # 결측/정규화
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

        # 차원 축소(또는 회전) → self.factors 차원으로 프로젝션
        k = self.factors
        projection_matrix = self._rng.normal(scale=0.01, size=(dim, k)).astype(
            np.float32
        )

        item_factors_init = image_embedding_matrix @ projection_matrix

        item_norms = np.linalg.norm(item_factors_init, axis=1, keepdims=True) + 1e-8
        item_factors_init = item_factors_init / item_norms

        logger.info("[INIT] 이미지 기반 item 초기값(V_init) 생성 완료")
        return item_factors_init.astype(np.float32)

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

    def build_model(
        self,
        r_ui_csr: csr_matrix,
        item_to_index: Dict[str, int],
        image_index_path: Optional[Path],
        db_path: Optional[Path],
    ) -> ALS:
        """
        - r_ui_csr: (users x items) user–item 가중치 행렬
        - image_index_path가 있으면:
            FAISS → 이미지 임베딩 → item_factors 초기값으로 사용 (warm start)
        - 없으면:
            item_factors / user_factors 랜덤 초기화
        """
        num_users, num_items = r_ui_csr.shape
        model = ALS(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            dtype=np.float32,
            random_state=self.random_state,
        )
        if image_index_path is not None:
            if db_path is None:
                raise ValueError(
                    "image_index_path를 쓰려면 db_path도 함께 넘겨줘야 합니다."
                )

            parent_asin_to_item_id = self._load_parentasin_to_itemid(db_path)

            item_init = self._build_item_init_from_images(
                item_to_index=item_to_index,
                image_index_path=image_index_path,
                parent_asin_to_item_id=parent_asin_to_item_id,
            )

            if item_init.shape != (num_items, self.factors):
                raise ValueError(
                    f"item_init shape mismatch: expected ({num_items}, {self.factors}), "
                    f"got {item_init.shape}"
                )
            norms = np.linalg.norm(item_init, axis=1, keepdims=True) + 1e-8
            item_init = (item_init / norms) * 0.01
            model.item_factors = item_init.astype(np.float32)

        logger.info(
            "[TRAIN] implicit ALS fit() 시작 (users=%s, items=%s)",
            f"{num_users:,}",
            f"{num_items:,}",
        )
        model.fit(r_ui_csr, show_progress=True)
        logger.info("[TRAIN] implicit ALS fit() 완료")

        return model

    def train_and_save(
        self,
        jsonl_path: Path,
        out_dir: Path,
        image_index_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        df = self._load_df(jsonl_path)
        r_ui_confidence = self._build_confidence(df)

        df_mapped, user_to_index, item_to_index = self._build_mappings(df)
        r_ui_csr = self._build_user_item_matrix(df_mapped, r_ui_confidence)

        # implicit ALS warm start
        model = self.build_model(
            r_ui_csr=r_ui_csr,
            item_to_index=item_to_index,
            image_index_path=image_index_path,
            db_path=db_path,
        )

        # 매핑 저장
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
