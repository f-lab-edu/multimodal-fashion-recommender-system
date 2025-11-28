# recommand/two_tower_model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """
    - item_tower: CLIP 이미지 임베딩(512차원)을 입력 받아 K차원 latent v_i로 투영
    - user_tower: 유저의 "raw" 임베딩(아이템 CLIP 평균, 512차원)을 입력 받아 K차원 latent u로 투영
    """

    def __init__(
        self,
        clip_dim: int = 512,
        hidden_dim: int = 256,
        emb_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 아이템 타워
        self.item_mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        # 유저 타워
        self.user_mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def encode_items(self, item_raw_embs: torch.Tensor) -> torch.Tensor:
        """
        item_raw_embs: (B, clip_dim)  # CLIP 이미지 임베딩
        return: (B, emb_dim)
        """
        v = self.item_mlp(item_raw_embs)
        v = F.normalize(v, dim=-1)  # cosine 기반 추천을 위해 정규화
        return v

    def encode_users(self, user_raw_embs: torch.Tensor) -> torch.Tensor:
        """
        user_raw_embs: (B, clip_dim)  # 유저의 mean CLIP 임베딩
        return: (B, emb_dim)
        """
        u = self.user_mlp(user_raw_embs)
        u = F.normalize(u, dim=-1)
        return u

    def forward(
        self,
        user_raw_embs: torch.Tensor,  # (B, clip_dim)
        pos_item_raw_embs: torch.Tensor,  # (B, clip_dim)
        neg_item_raw_embs: torch.Tensor,  # (B, clip_dim)
    ) -> dict[str, torch.Tensor]:
        """
        한 배치에 대해 BPR-style loss 계산을 위한 점수 반환.
        """
        u = self.encode_users(user_raw_embs)  # (B, K)
        v_pos = self.encode_items(pos_item_raw_embs)  # (B, K)
        v_neg = self.encode_items(neg_item_raw_embs)  # (B, K)

        # dot product scores
        pos_scores = (u * v_pos).sum(dim=-1)  # (B,)
        neg_scores = (u * v_neg).sum(dim=-1)  # (B,)

        return {
            "u": u,
            "v_pos": v_pos,
            "v_neg": v_neg,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
        }

    @staticmethod
    def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        BPR loss: -log σ(s_pos - s_neg)
        """
        x = pos_scores - neg_scores
        return -F.logsigmoid(x).mean()
