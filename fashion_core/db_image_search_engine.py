# fashion_core/db_image_search_engine.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import sqlite3
import torch
import faiss
from transformers import CLIPModel, CLIPProcessor

FASHION_CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"


class DbImageSearchEngine:
    def __init__(
        self,
        db_path: Path,
        index_path: Path,
        model_name: str = FASHION_CLIP_MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        self.model_name = model_name

        if not self.db_path.exists():
            raise FileNotFoundError(f"DB not found: {self.db_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[INFO] Using device = {self.device}")
        print(f"[INFO] DB PATH      = {self.db_path}")
        print(f"[INFO] FAISS INDEX  = {self.index_path}")
        print(f"[INFO] MODEL        = {self.model_name}")

        # 1) DB 연결
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # 2) FAISS 인덱스 로드
        self.index = faiss.read_index(str(self.index_path))
        print(f"[INFO] index.ntotal = {self.index.ntotal}")

        # 3) CLIP 모델 로드
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
