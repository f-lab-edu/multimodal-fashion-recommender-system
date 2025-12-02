# common/item_clip_text_embeds.py
# npz -> id2embded 딕셔너리 로드 유틸
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_item_text_embeddings(npz_path: Path) -> Tuple[Dict]:
    """
    - id2emb: {PARENT_ASIN(str) -> embedding(np.ndarray, float32)}
    - dim: 임베딩 차원
    """
    data = np.load(npz_path, allow_pickle=True)
    item_ids = data["item_ids"].tolist()
    embs = data["embeddings"].astype("float32")

    id2emb: Dict[str, np.ndarray] = dict(zip(item_ids, embs))
    dim = embs.shape[1]
    print(f"[EMB] loaded {len(id2emb):,} item text embeddings, dim={dim}")
    return id2emb, dim
