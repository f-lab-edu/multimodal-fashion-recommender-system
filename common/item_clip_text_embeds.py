# common/item_clip_text_embeds.py
# npz -> id2embded 딕셔너리 로드 유틸
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import logging
import numpy as np

logger = logging.getLogger(__name__)


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
    logger.info("[EMB] loaded %s item text embeddings, dim=%d", f"{len(id2emb):,}", dim)
    return id2emb, dim
