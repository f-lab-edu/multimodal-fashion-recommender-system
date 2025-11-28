# recommand/two_tower_trainer.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import sqlite3
import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from common.als_io import load_als_jsonl
from recommand.two_tower_model import TwoTowerModel

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


class InteractionDataset(Dataset):
    """
    - 각 샘플: (user_idx, pos_item_pos)
    - neg sampling은 collate_fn or training loop에서 처리
    """

    def __init__(self, user_idxs: np.ndarray, item_positions: np.ndarray) -> None:
        assert len(user_idxs) == len(item_positions)
        self.user_idxs = user_idxs
        self.item_positions = item_positions

    def __len__(self) -> int:
        return len(self.user_idxs)

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return int(self.user_idxs[idx]), int(self.item_positions[idx])


class TwoTowerTrainer:
    def __init__(
        self,
        db_path: Path,
        image_index_path: Path,
        train_jsonl: Path,
        clip_dim: int = 512,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        batch_size: int = 1024,
        num_epochs: int = 10,
        lr: float = 1e-3,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.db_path = db_path
        self.image_index_path = image_index_path
        self.train_jsonl = train_jsonl

        self.clip_dim = clip_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.random_state = random_state

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.rng = np.random.default_rng(random_state)

        # 아래에서 채워질 것들
        self.user2idx: Dict[str, int] = {}
        self.item2pos: Dict[str, int] = {}
        self.user_raw_embs: np.ndarray | None = None  # (num_users, clip_dim)
        self.item_clip_embs: np.ndarray | None = None  # (num_faiss_items, clip_dim)
        self.user_pos_items: Dict[int, set[int]] = {}

        self.model: TwoTowerModel | None = None

    # ---------- 1) 데이터 로드 + 매핑 ----------

    def _load_train_df(self) -> pd.DataFrame:
        df = load_als_jsonl(self.train_jsonl)
        df = df.rename(columns={"PARENT_ASIN": "item_id"})
        print(f"[TT] train df rows = {len(df):,}")
        return df

    def _load_parentasin_to_itemid(self) -> Dict[str, int]:
        print(f"[TT] DB에서 parent_asin → items.id 매핑 로드: {self.db_path}")
        with sqlite3.connect(str(self.db_path)) as conn:
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

        print(f"[TT] parentasin_to_itemid size = {len(mapping):,}")
        return mapping

    def _load_faiss_index(self) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
        """
        FAISS IndexIDMap에서:
        - xb (n, clip_dim)  # CLIP 임베딩
        - ids (n,)          # items.id
        - id → row_pos 매핑
        """
        print(f"[TT] FAISS 인덱스 로드: {self.image_index_path}")
        index = faiss.read_index(str(self.image_index_path))
        n = index.ntotal
        d = index.d
        print(f"[TT] FAISS vectors: n={n}, dim={d}")

        # 실제 벡터 들고 있는 base index
        base_index = index
        if hasattr(index, "index"):
            base_index = index.index

        try:
            xb = base_index.reconstruct_n(0, n)  # (n, d)
        except Exception as e:
            if hasattr(base_index, "xb"):
                xb = faiss.vector_to_array(base_index.xb).reshape(n, d)
            else:
                raise RuntimeError(
                    f"벡터 복원이 안 되는 FAISS 타입입니다: {type(base_index)}"
                ) from e

        ids = faiss.vector_to_array(index.id_map)  # (n,)

        id_to_pos: Dict[int, int] = {int(fid): pos for pos, fid in enumerate(ids)}
        return xb.astype("float32"), ids.astype("int64"), id_to_pos

    def _build_mappings_and_user_embs(self) -> Tuple[InteractionDataset, int, int]:
        df = self._load_train_df()
        parentasin_to_itemid = self._load_parentasin_to_itemid()
        item_clip_embs, faiss_ids, id_to_pos = self._load_faiss_index()

        # 1) 유효한 (user, parent_asin)만 남기기
        #    - parent_asin → items.id 매핑
        df["item_db_id"] = df["item_id"].map(parentasin_to_itemid)
        df = df.dropna(subset=["item_db_id"]).copy()
        df["item_db_id"] = df["item_db_id"].astype(int)

        #    - items.id → FAISS row_pos 매핑
        df["item_pos"] = df["item_db_id"].map(id_to_pos)
        df = df.dropna(subset=["item_pos"]).copy()
        df["item_pos"] = df["item_pos"].astype(int)

        print(f"[TT] train df (after join with DB+FAISS) rows = {len(df):,}")

        # 2) user2idx 생성
        users = df["user_id"].unique()
        self.user2idx = {u: i for i, u in enumerate(users)}

        # item2pos: parent_asin → faiss row_pos (나중에 필요하면 사용)
        self.item2pos = {
            row["item_id"]: int(row["item_pos"])
            for _, row in df[["item_id", "item_pos"]].drop_duplicates().iterrows()
        }

        df["user_idx"] = df["user_id"].map(self.user2idx)
        df["user_idx"] = df["user_idx"].astype(int)
        df["item_pos"] = df["item_pos"].astype(int)

        num_users = len(self.user2idx)
        num_items = item_clip_embs.shape[0]  # 전체 FAISS 아이템 수

        print(f"[TT] num_users={num_users:,}, num_faiss_items={num_items:,}")

        # 3) ⭐ 리뷰 기반 confidence 가중치 계산
        #    rating, verified_purchase, helpful_vote가 df에 있다고 가정 (ALS와 동일 스키마)
        base = df["rating"].clip(lower=1.0, upper=5.0).astype(np.float32)
        verified_bonus = np.where(df["verified_purchase"], 1.2, 1.0).astype(np.float32)
        helpful_bonus = (
            1.0 + 0.1 * np.log1p(df["helpful_vote"].astype(np.float32))
        ).astype(np.float32)

        conf = base * verified_bonus * helpful_bonus
        conf = np.clip(conf, a_min=0.0, a_max=20.0).astype(np.float32)
        df["conf"] = conf

        # 4) 유저별 본 아이템 set + 유저 raw 임베딩(가중 평균) 계산
        self.user_pos_items = {u_idx: set() for u_idx in range(num_users)}

        sum_embs = np.zeros((num_users, self.clip_dim), dtype=np.float32)
        sum_w = np.zeros(num_users, dtype=np.float32)

        for _, row in df[["user_idx", "item_pos", "conf"]].iterrows():
            u = int(row["user_idx"])
            pos = int(row["item_pos"])
            w = float(row["conf"])

            self.user_pos_items[u].add(pos)
            sum_embs[u] += w * item_clip_embs[pos]
            sum_w[u] += w

        # 가중 평균: sum(w * e) / sum(w)
        # sum_w가 0인 경우(이론상 거의 없지만)를 위해 작은 값으로 나눔
        denom = np.maximum(sum_w[:, None], 1e-6)
        user_raw_embs = sum_embs / denom

        self.user_raw_embs = user_raw_embs  # (num_users, clip_dim)
        self.item_clip_embs = item_clip_embs  # (num_items_in_faiss, clip_dim)

        # 5) 학습용 InteractionDataset (각 리뷰 = (user_idx, item_pos))
        user_idxs = df["user_idx"].to_numpy(dtype=np.int64)
        item_positions = df["item_pos"].to_numpy(dtype=np.int64)
        dataset = InteractionDataset(user_idxs, item_positions)

        return dataset, num_users, num_items

    # ---------- 2) 학습 루프 ----------

    def _sample_negative_items(
        self, user_idx: int, num_items: int, num_neg: int = 3
    ) -> List[int]:
        """
        해당 user가 보지 않은 아이템에서 negative 샘플링
        """
        pos_set = self.user_pos_items[user_idx]
        negs: List[int] = []
        while len(negs) < num_neg:
            j = int(self.rng.integers(low=0, high=num_items))
            if j in pos_set:
                continue
            negs.append(j)
        return negs

    def train(self, out_path: Path) -> None:
        dataset, num_users, num_items = self._build_mappings_and_user_embs()
        assert self.user_raw_embs is not None
        assert self.item_clip_embs is not None

        model = TwoTowerModel(
            clip_dim=self.clip_dim,
            hidden_dim=self.hidden_dim,
            emb_dim=self.emb_dim,
        ).to(self.device)
        self.model = model

        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=self.lr)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        for epoch in range(1, self.num_epochs + 1):
            model.train()
            total_loss = 0.0
            total_steps = 0

            for user_idxs, pos_item_pos in loader:
                user_idxs = user_idxs.numpy()
                pos_item_pos = pos_item_pos.numpy()

                # user_raw_embs / pos_item_raw / neg_item_raw 가져오기
                user_raw = self.user_raw_embs[user_idxs]  # (B, clip_dim)
                pos_raw = self.item_clip_embs[pos_item_pos]  # (B, clip_dim)

                neg_positions = []
                for u in user_idxs:
                    neg_positions.extend(
                        self._sample_negative_items(int(u), num_items, num_neg=1)
                    )
                neg_positions = np.array(neg_positions, dtype=np.int64)
                neg_raw = self.item_clip_embs[neg_positions]  # (B, clip_dim)

                user_raw_t = torch.from_numpy(user_raw).to(self.device)
                pos_raw_t = torch.from_numpy(pos_raw).to(self.device)
                neg_raw_t = torch.from_numpy(neg_raw).to(self.device)

                out = model(
                    user_raw_embs=user_raw_t,
                    pos_item_raw_embs=pos_raw_t,
                    neg_item_raw_embs=neg_raw_t,
                )

                loss = TwoTowerModel.bpr_loss(out["pos_scores"], out["neg_scores"])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_steps += 1

            avg_loss = total_loss / max(total_steps, 1)
            print(
                f"[TT] epoch {epoch}/{self.num_epochs} - train BPR loss={avg_loss:.4f}"
            )

        # 최종 저장: 모델 + user/item 매핑, raw_embs 등
        out_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": model.state_dict(),
            "user2idx": self.user2idx,
            "item2pos": self.item2pos,
            "user_raw_embs": self.user_raw_embs,
            # item_clip_embs는 너무 크면 별도 npy로 저장하는 것도 고려 가능
        }
        torch.save(state, out_path)
        print(f"[TT] 저장 완료: {out_path}")
