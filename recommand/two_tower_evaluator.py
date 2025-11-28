# recommand/two_tower_evaluator.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import faiss
import numpy as np
import pandas as pd
import torch

from common.als_io import load_als_jsonl
from recommand.two_tower_model import TwoTowerModel

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


class TwoTowerEvaluator:
    """
    - 학습된 Two-Tower 모델(pt) + 저장된 매핑(state)을 로드
    - FAISS 이미지 인덱스에서 임베딩을 꺼내고,
      그 중 'train에 등장한 아이템(item2pos에 있는 애들)'만 후보로 사용
    - train.jsonl로 user_pos_items(이미 본 아이템) 구성
    - test.jsonl로 Recall@K / HitRate@K 계산
    """

    def __init__(
        self,
        model_path: Path,
        image_index_path: Path,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model_path = model_path
        self.image_index_path = image_index_path

        # ---------- 1) state 로드 ----------
        state = torch.load(model_path, map_location=self.device)
        model_state = state["model_state_dict"]

        self.user2idx: Dict[str, int] = state["user2idx"]
        self.item2pos: Dict[str, int] = state[
            "item2pos"
        ]  # item_id(str) -> item_pos(int)
        self.user_raw_embs: np.ndarray = state["user_raw_embs"]  # (num_users, clip_dim)

        # hyperparam (학습 때와 맞춰야 함)
        clip_dim = self.user_raw_embs.shape[1]
        hidden_dim = model_state["item_mlp.0.weight"].shape[0]
        emb_dim = model_state["item_mlp.3.weight"].shape[0]

        # ---------- 2) 모델 복원 ----------
        self.model = TwoTowerModel(
            clip_dim=clip_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
        ).to(self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

        # ---------- 3) FAISS 전체 로드 ----------
        self.item_clip_embs_full, self.num_items_full = self._load_item_clip_embs_full()

        # ---------- 4) 후보 아이템: train에 등장한 item_pos만 사용 ----------
        all_candidate_positions = sorted(set(self.item2pos.values()))
        self.candidate_positions: np.ndarray = np.array(
            all_candidate_positions, dtype=np.int64
        )  # (Nc,)

        # item_pos -> candidate index 매핑 (필터링/마스킹에 사용)
        self.pos_to_cand_idx: Dict[int, int] = {
            int(pos): idx for idx, pos in enumerate(all_candidate_positions)
        }

        # ---------- 5) 후보 아이템에 대한 latent 계산 ----------
        self.item_latents = self._build_item_latents_for_candidates()
        self.num_candidates = self.item_latents.shape[0]

        # user_pos_items: train에서 채울 것
        self.user_pos_items: Dict[int, set[int]] = {}

        print("[TT-EVAL] 모델/상태 로드 완료")
        print(f"[TT-EVAL] 유저 수: {len(self.user2idx):,}")
        print(f"[TT-EVAL] FAISS 전체 아이템 수: {self.num_items_full:,}")
        print(f"[TT-EVAL] 후보 아이템 수(Train 등장 아이템): {self.num_candidates:,}")

    # ---------- FAISS 로딩 ----------

    def _load_item_clip_embs_full(self) -> Tuple[np.ndarray, int]:
        """
        - image_index_path에서 FAISS IndexIDMap 로드
        - 전체 아이템 임베딩(이미지 CLIP)을 numpy로 꺼내온다.
        """
        print(f"[TT-EVAL] FAISS 인덱스 로드: {self.image_index_path}")
        index = faiss.read_index(str(self.image_index_path))
        n = index.ntotal
        d = index.d
        print(f"[TT-EVAL] FAISS vectors: n={n}, dim={d}")

        # base index 추출 (IndexIDMap일 경우 내부 index 사용)
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

        xb = xb.astype("float32")
        # user_raw_embs와 차원 일치 확인
        assert (
            xb.shape[1] == self.user_raw_embs.shape[1]
        ), f"FAISS dim {xb.shape[1]} != user_raw_embs dim {self.user_raw_embs.shape[1]}"

        return xb, n

    def _build_item_latents_for_candidates(self, batch_size: int = 4096) -> np.ndarray:
        """
        - candidate_positions에 해당하는 item_clip_embs만 뽑아서
        - TwoTowerModel의 item 타워로 통과시켜 latent 생성
        """
        print("[TT-EVAL] candidate item_latents 계산 시작 (Train 등장 아이템만)")
        idx = self.candidate_positions  # (Nc,)
        all_embs = self.item_clip_embs_full[idx]  # (Nc, clip_dim)
        n, d = all_embs.shape

        item_latents_list: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = torch.from_numpy(all_embs[start:end]).to(self.device)
                v = self.model.encode_items(batch)  # (B, emb_dim)
                item_latents_list.append(v.cpu().numpy())

        item_latents = np.concatenate(item_latents_list, axis=0).astype("float32")
        print(f"[TT-EVAL] candidate item_latents shape: {item_latents.shape}")
        return item_latents

    # ---------- train.jsonl 기반 user_pos_items 구성 ----------

    def build_user_pos_from_train(self, train_jsonl: Path) -> None:
        """
        - train.jsonl를 읽어서, 각 user가 train에서 본 아이템(item_pos) 집합을 만든다.
        - ALS evaluator의 filter_already_liked_items와 같은 역할.
        - 후보는 item2pos(=Train에서 등장한 아이템) 안에 있는 것들만 고려.
        """
        df = load_als_jsonl(train_jsonl)
        df = df.rename(columns={"PARENT_ASIN": "item_id"})

        # 매핑에 존재하는 user / item만 사용
        df = df[
            df["user_id"].isin(self.user2idx) & df["item_id"].isin(self.item2pos)
        ].copy()

        if df.empty:
            raise RuntimeError(
                "train 데이터에서 유효한 user/item이 없습니다 (Two-Tower 평가)."
            )

        df["user_idx"] = df["user_id"].map(self.user2idx).astype(int)
        df["item_pos"] = df["item_id"].map(self.item2pos).astype(int)

        self.user_pos_items = {u_idx: set() for u_idx in range(len(self.user2idx))}

        for _, row in df[["user_idx", "item_pos"]].iterrows():
            u = int(row["user_idx"])
            pos = int(row["item_pos"])
            # pos가 candidate_positions 안에 있어야만 사용
            if pos in self.pos_to_cand_idx:
                self.user_pos_items[u].add(pos)

        print("[TT-EVAL] user_pos_items 구축 완료")
        example_len = len(self.user_pos_items.get(0, []))
        print(f"[TT-EVAL] 예시 유저 0의 본 아이템 수: {example_len}")

    # ---------- 평가 ----------

    def _load_test_df(self, test_jsonl: Path) -> pd.DataFrame:
        df = load_als_jsonl(test_jsonl)
        df = df.rename(columns={"PARENT_ASIN": "item_id"})
        return df

    def evaluate_on_test(self, test_jsonl: Path, k: int = 10) -> Dict[str, float]:
        """
        - test jsonl 기준으로 Recall@K / HitRate@K 측정
        - ALS evaluator와 비슷한 방식:
          * 평가 유저: train/test 둘 다에 등장 + 매핑 존재
          * 추천 시: train에서 본 아이템은 후보에서 제외
          * 후보 아이템: train에 등장한 item2pos (item_pos)만 사용
        """
        if not self.user_pos_items:
            raise RuntimeError(
                "user_pos_items가 비어 있습니다. build_user_pos_from_train()을 먼저 호출하세요."
            )

        df_test = self._load_test_df(test_jsonl)

        # 평가 대상: 매핑에 존재하는 user + item만
        df = df_test[
            df_test["user_id"].isin(self.user2idx)
            & df_test["item_id"].isin(self.item2pos)
        ].copy()

        if df.empty:
            print("[TT-EVAL] 평가 대상 행이 없습니다.")
            return {"recall_at_k": 0.0, "hit_rate": 0.0, "num_users": 0}

        df["user_idx"] = df["user_id"].map(self.user2idx).astype(int)
        df["item_pos"] = df["item_id"].map(self.item2pos).astype(int)

        # test 정답 중에서도, 후보(candidate_positions)에 있는 것만 사용
        df = df[df["item_pos"].isin(self.pos_to_cand_idx)].copy()
        if df.empty:
            print("[TT-EVAL] 후보 안에 들어오는 정답 아이템이 없습니다.")
            return {"recall_at_k": 0.0, "hit_rate": 0.0, "num_users": 0}

        # Torch로 item_latents / candidate_positions 가져오기
        item_latents_t = torch.from_numpy(self.item_latents).to(
            self.device
        )  # (Nc, emb_dim)
        candidate_positions_np = self.candidate_positions  # (Nc,)

        hits = 0
        total_positives = 0
        hit_users = 0
        num_users = 0

        # 유저별 groupby
        for user_idx, g in df.groupby("user_idx"):
            user_idx = int(user_idx)
            pos_items = set(int(x) for x in g["item_pos"].tolist())
            if not pos_items:
                continue

            # train에서 본 아이템 (item_pos 기준)
            seen_items = self.user_pos_items.get(user_idx, set())

            # 유저 latent 계산
            user_raw = self.user_raw_embs[user_idx : user_idx + 1]  # (1, clip_dim)
            user_raw_t = torch.from_numpy(user_raw).to(self.device)

            with torch.no_grad():
                u = self.model.encode_users(user_raw_t)  # (1, emb_dim)
                u_vec = u.squeeze(0)  # (emb_dim,)

                scores = item_latents_t @ u_vec  # (Nc,)

            # train에서 본 아이템은 추천 후보에서 제외 (candidate index 기준으로 마스킹)
            if seen_items:
                cand_seen_idx = [
                    self.pos_to_cand_idx[pos]
                    for pos in seen_items
                    if pos in self.pos_to_cand_idx
                ]
                if cand_seen_idx:
                    seen_idx_t = torch.tensor(
                        cand_seen_idx, dtype=torch.long, device=self.device
                    )
                    scores[seen_idx_t] = -1e9

            # 유효한 후보 개수
            valid_items = self.num_candidates - len(seen_items)
            if valid_items <= 0:
                continue

            top_k = min(k, valid_items)
            top_scores, top_cand_idx = torch.topk(scores, k=top_k)  # candidate index들
            top_cand_idx_np = top_cand_idx.cpu().numpy()

            # candidate index -> 실제 item_pos 매핑
            rec_item_pos = set(int(candidate_positions_np[i]) for i in top_cand_idx_np)

            inter = pos_items & rec_item_pos

            num_users += 1
            total_positives += len(pos_items)

            if inter:
                hit_users += 1
                hits += len(inter)

        if num_users == 0 or total_positives == 0:
            recall_at_k = 0.0
            hit_rate = 0.0
        else:
            recall_at_k = hits / total_positives
            hit_rate = hit_users / num_users

        print(f"[TT-EVAL] num_users={num_users}")
        print(f"[TT-EVAL] Recall@{k}: {recall_at_k:.4f}")
        print(f"[TT-EVAL] HitRate@{k}: {hit_rate:.4f}")

        return {
            "recall_at_k": float(recall_at_k),
            "hit_rate": float(hit_rate),
            "num_users": int(num_users),
        }
