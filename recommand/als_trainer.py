# recommand/als_trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import implicit
import pickle

import sqlite3
import faiss

from common.als_io import load_als_jsonl

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


class ALSTrainer:
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 1e-3,
        iterations: int = 20,
        alpha: float = 40.0,
        use_gpu: bool = False,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.use_gpu = use_gpu
        self.random_state = random_state

        self._rng = np.random.default_rng(self.random_state)

    def _load_df(self, path: Path) -> pd.DataFrame:
        df = load_als_jsonl(path)
        df = df.rename(columns={"PARENT_ASIN": "item_id"})

        print(f"[TRAIN] 입력 df 행 수: {len(df):,}")
        print(f"[TRAIN] 유저 수: {df['user_id'].nunique():,}")
        print(f"[TRAIN] 아이템 수: {df['item_id'].nunique():,}")
        return df

    def _build_confidence(self, df: pd.DataFrame) -> pd.Series:
        base = df["rating"].clip(lower=1.0, upper=5.0)
        verified_bonus = np.where(df["verified_purchase"], 1.2, 1.0)
        helpful_bonus = 1.0 + 0.1 * np.log1p(df["helpful_vote"])
        R_ui = base * verified_bonus * helpful_bonus
        return R_ui.clip(upper=20.0).astype(np.float32)

    def _build_mappings(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
        users = df["user_id"].unique()
        items = df["item_id"].unique()

        user2idx = {u: i for i, u in enumerate(users)}
        item2idx = {it: i for i, it in enumerate(items)}

        df_mapped = df.copy()
        df_mapped["user_idx"] = df_mapped["user_id"].map(user2idx).astype(np.int32)
        df_mapped["item_idx"] = df_mapped["item_id"].map(item2idx).astype(np.int32)

        return df_mapped, user2idx, item2idx

    def _build_user_item_matrix(self, df_mapped: pd.DataFrame, R_ui: pd.Series):
        user_idx = df_mapped["user_idx"].to_numpy()
        item_idx = df_mapped["item_idx"].to_numpy()
        data = R_ui.to_numpy()

        num_users = int(df_mapped["user_idx"].max()) + 1
        num_items = int(df_mapped["item_idx"].max()) + 1

        mat = coo_matrix(
            (data, (user_idx, item_idx)),
            shape=(num_users, num_items),
            dtype=np.float32,
        ).tocsr()
        print(f"[TRAIN] matrix shape: users={num_users:,}, items={num_items:,}")
        return mat

    def _build_item_init_from_faiss(
        self,
        item2idx: Dict[str, int],
        image_index_path: Path,
        parentasin_to_itemid: Dict[str, int],
    ) -> np.ndarray:
        print(f"[INIT] FAISS 인덱스 로드: {image_index_path}")
        index = faiss.read_index(str(image_index_path))

        n = index.ntotal
        d = index.d
        print(f"[INIT] FAISS vectors: n={n}, dim={d}")

        base_index = index
        if hasattr(index, "index"):
            base_index = index.index

        try:
            xb = base_index.reconstruct_n(0, n)
        except Exception as e:
            if hasattr(base_index, "xb"):
                xb = faiss.vector_to_array(base_index.xb).reshape(n, d)
            else:
                raise RuntimeError(
                    f"FAISS 인덱스 타입에서 벡터를 복원(reconstruct)할 수 없습니다: {type(base_index)}"
                ) from e

        ids = faiss.vector_to_array(index.id_map)

        faiss_id_to_pos: Dict[int, int] = {int(fid): pos for pos, fid in enumerate(ids)}

        num_items = len(item2idx)
        img_emb_mat = np.zeros((num_items, d), dtype=np.float32)

        missing = 0
        matched = 0

        for item_id, col_idx in item2idx.items():
            faiss_id = parentasin_to_itemid.get(item_id)
            if faiss_id is None:
                missing += 1
                continue

            pos = faiss_id_to_pos.get(int(faiss_id))
            if pos is None:
                missing += 1
                continue

            img_emb_mat[col_idx] = xb[pos]
            matched += 1

        print(
            f"[INIT] 이미지 임베딩 매칭 완료: matched={matched:,}, missing={missing:,}"
        )

        row_norms = np.linalg.norm(img_emb_mat, axis=1)
        zero_mask = row_norms < 1e-8
        if zero_mask.any():
            print(f"[INIT] 이미지 없음 → 랜덤 초기화 아이템 수: {zero_mask.sum():,}")
            img_emb_mat[zero_mask] = self._rng.normal(
                scale=0.01,
                size=(zero_mask.sum(), d),
            )

        norms = np.linalg.norm(img_emb_mat, axis=1, keepdims=True) + 1e-8
        img_emb_mat = img_emb_mat / norms

        K = self.factors
        W = self._rng.normal(scale=0.01, size=(d, K)).astype(np.float32)

        V_init = img_emb_mat @ W

        V_norms = np.linalg.norm(V_init, axis=1, keepdims=True) + 1e-8
        V_init = V_init / V_norms

        print("[INIT] 이미지 기반 item 초기값(V_init) 생성 완료")
        return V_init.astype(np.float32)

    def _als_step(
        self,
        Cui: csr_matrix,
        Rui: csr_matrix,
        fixed_factors: np.ndarray,
    ) -> np.ndarray:
        n_rows = Cui.shape[0]
        K = fixed_factors.shape[1]

        Y = fixed_factors
        YtY = Y.T @ Y
        X_new = np.zeros((n_rows, K), dtype=np.float32)

        eye = np.eye(K, dtype=np.float32)
        reg = self.regularization

        C_indptr = Cui.indptr
        C_indices = Cui.indices
        C_data = Cui.data

        R_data = Rui.data

        for u in range(n_rows):
            start, end = C_indptr[u], C_indptr[u + 1]
            cols = C_indices[start:end]
            C_ui_vals = C_data[start:end]
            R_ui_vals = R_data[start:end]

            if len(cols) == 0:
                X_new[u] = 0.0
                continue

            A = YtY.copy()
            b = np.zeros(K, dtype=np.float32)

            for i, c_ui, r_ui in zip(cols, C_ui_vals, R_ui_vals):
                y_i = Y[i]
                A += (c_ui - 1.0) * np.outer(y_i, y_i)
                b += (c_ui * r_ui) * y_i

            A += reg * eye
            X_new[u] = np.linalg.solve(A, b)

        return X_new

    def _load_parentasin_to_itemid(self, db_path: Path) -> Dict[str, int]:
        print(f"[INIT] DB에서 parent_asin → items.id 매핑 로드: {db_path}")
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

        print(f"[INIT] 매핑 개수: {len(mapping):,}")
        return mapping

    def _build_val_user_pos(
        self,
        df_val: pd.DataFrame,
        user2idx: Dict[str, int],
        item2idx: Dict[str, int],
    ) -> Dict[int, Set[int]]:
        df = df_val.copy()
        df = df[df["user_id"].isin(user2idx) & df["item_id"].isin(item2idx)].copy()

        if df.empty:
            print("[EARLY] val 데이터에서 유효한 user/item이 없습니다.")
            return {}

        df["user_idx"] = df["user_id"].map(user2idx).astype(int)
        df["item_idx"] = df["item_id"].map(item2idx).astype(int)

        user_pos: Dict[int, Set[int]] = {}
        for u, g in df.groupby("user_idx"):
            user_pos[int(u)] = set(int(i) for i in g["item_idx"].tolist())

        print(f"[EARLY] val users for early stopping: {len(user_pos):,}")
        return user_pos

    def _hit_rate_at_k(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        val_user_pos: Dict[int, Set[int]],
        k: int,
    ) -> float:
        if not val_user_pos:
            return 0.0

        num_items = Y.shape[0]
        hit_users = 0
        num_users_eval = 0

        for u, pos_items in val_user_pos.items():
            if not pos_items:
                continue

            scores_u = X[u] @ Y.T
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
        R_ui_csr: csr_matrix,
        item_init: np.ndarray,
        val_user_pos: Optional[Dict[int, Set[int]]] = None,
        eval_k: int = 10,
        patience: int = 3,
    ) -> implicit.als.AlternatingLeastSquares:
        num_users, num_items = R_ui_csr.shape
        K = self.factors

        print("[TRAIN] 커스텀 ALS (이미지 초기값) 학습 시작")

        Cui = R_ui_csr.copy()
        Cui.data = self.alpha * Cui.data
        Cui.data += 1.0
        Cui = Cui.tocsr()

        Rui = R_ui_csr.copy()
        Rui.data[:] = 1.0

        X = self._rng.normal(scale=0.01, size=(num_users, K)).astype(np.float32)
        Y = item_init.astype(np.float32)

        best_hit = -1.0
        best_X = None
        best_Y = None
        no_improve = 0

        for it in range(self.iterations):
            print(f"[TRAIN] ALS iter {it + 1}/{self.iterations} - user 업데이트")
            X = self._als_step(Cui, Rui, Y)

            print(f"[TRAIN] ALS iter {it + 1}/{self.iterations} - item 업데이트")
            Y = self._als_step(Cui.T.tocsr(), Rui.T.tocsr(), X)

            if val_user_pos is not None:
                hit = self._hit_rate_at_k(X, Y, val_user_pos, k=eval_k)
                print(f"[EARLY] iter {it + 1} - val HitRate@{eval_k} = {hit:.4f}")

                if hit > best_hit + 1e-6:
                    best_hit = hit
                    best_X = X.copy()
                    best_Y = Y.copy()
                    no_improve = 0
                    print(f"[EARLY] best 갱신 (HitRate={best_hit:.4f})")
                else:
                    no_improve += 1
                    print(f"[EARLY] 개선 없음 ({no_improve}/{patience})")
                    if no_improve >= patience:
                        print("[EARLY] Early stopping 발동")
                        break

        if val_user_pos is not None and best_X is not None and best_Y is not None:
            print(f"[EARLY] 최종 best HitRate@{eval_k} = {best_hit:.4f}")
            X_final = best_X
            Y_final = best_Y
        else:
            X_final = X
            Y_final = Y

        print("[TRAIN] 커스텀 ALS 학습 완료")

        model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
        )
        model.user_factors = X_final.astype(np.float32)
        model.item_factors = Y_final.astype(np.float32)

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
        eval_k: int = 10,
        patience: int = 3,
    ) -> None:
        df = self._load_df(jsonl_path)
        R_ui = self._build_confidence(df)

        df_mapped, user2idx, item2idx = self._build_mappings(df)
        R_ui_csr = self._build_user_item_matrix(df_mapped, R_ui)

        val_user_pos: Optional[Dict[int, Set[int]]] = None
        if val_jsonl_path is not None:
            print(f"[EARLY] val_jsonl 로드: {val_jsonl_path}")
            df_val = self._load_df(val_jsonl_path)
            val_user_pos = self._build_val_user_pos(df_val, user2idx, item2idx)

        if image_index_path is not None:
            if db_path is None:
                raise ValueError(
                    "image_index_path를 쓰려면 db_path도 함께 넘겨줘야 합니다."
                )

            parentasin_to_itemid = self._load_parentasin_to_itemid(db_path)

            item_init = self._build_item_init_from_faiss(
                item2idx=item2idx,
                image_index_path=image_index_path,
                parentasin_to_itemid=parentasin_to_itemid,
            )

            model = self._train_als_with_item_init(
                R_ui_csr=R_ui_csr,
                item_init=item_init,
                val_user_pos=val_user_pos,
                eval_k=eval_k,
                patience=patience,
            )
        else:
            print("[TRAIN] 이미지 초기값 없음 → 기본 implicit ALS 사용")
            model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                use_gpu=self.use_gpu,
                random_state=self.random_state,
            )
            print("[TRAIN] ALS 모델 학습 시작 (implicit.fit)")
            model.fit(R_ui_csr * self.alpha, show_progress=True)
            print("[TRAIN] ALS 모델 학습 완료 (implicit.fit)")

        idx2user = {idx: u for u, idx in user2idx.items()}
        idx2item = {idx: it for it, idx in item2idx.items()}

        out_dir.mkdir(parents=True, exist_ok=True)

        with (out_dir / "als_model.pkl").open("wb") as f:
            pickle.dump(model, f)

        with (out_dir / "mappings.pkl").open("wb") as f:
            pickle.dump(
                {
                    "user2idx": user2idx,
                    "item2idx": item2idx,
                    "idx2user": idx2user,
                    "idx2item": idx2item,
                },
                f,
            )

        print(f"[TRAIN] 저장 완료: {out_dir}")
