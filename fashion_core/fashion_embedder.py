# fashion_core/fashion_embedder.py
# Fashion-CLIP 모델을 이용해서
# 패션 아이템 이미지 + 텍스트 임베딩을 계산하는 유틸

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPModel, CLIPProcessor
import faiss


class FashionEmbeddingBuilder:
    """
    - 아마존 패션 JSONL + 로컬 이미지 → Fashion-CLIP 임베딩
    - FAISS 인덱스 + 메타데이터(JSONL)로 저장
    """

    def __init__(
        self,
        jsonl_path: Path,
        image_root: Path,
        index_path: Path,
        meta_path: Path,
        model_name: str = "patrickjohncyh/fashion-clip",
        text_batch_size: int = 64,
        image_batch_size: int = 32,
        device: Optional[str] = None,
        shard_id: Optional[int] = None,
        num_shards: Optional[int] = None,
        embedding_mode: str = "text",
    ):
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.index_path = index_path
        self.meta_path = meta_path
        self.model_name = model_name
        self.text_batch_size = text_batch_size
        self.image_batch_size = image_batch_size
        self.device = device

        # embedding_mode 유효성 체크
        if embedding_mode not in ("text", "image"):
            raise ValueError(
                f"embedding_mode must be one of 'text', 'image', got: {embedding_mode}"
            )
        self.embedding_mode = embedding_mode

        self.use_image = embedding_mode == "image"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 샤드 설정
        self.shard_id = shard_id
        self.num_shards = num_shards

        print(f"[INFO] Using device = {self.device}")
        print(f"[INFO] JSONL = {self.jsonl_path}")
        print(f"[INFO] IMAGE_ROOT = {self.image_root}")
        print(f"[INFO] INDEX_PATH = {self.index_path}")
        print(f"[INFO] META_PATH = {self.meta_path}")
        print(f"[INFO] MODEL = {self.model_name}")
        print(f"[INFO] USE_IMAGE = {self.use_image}")
        if self.num_shards is not None:
            print(f"[INFO] SHARD = {self.shard_id} / {self.num_shards}")
        else:
            print("[INFO] SHARD = (no sharding)")

        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

    def build_text_for_embedding(self, item: Dict[str, Any]) -> str:
        """
        JSONL 아이템에서 텍스트 필드들을 조합해서
        임베딩에 쓸 문자열 생성
        """
        fields = ["title", "store", "description", "features"]
        parts = []

        for field in fields:
            value = item.get(field)

            if not value:
                continue

            if isinstance(value, list):
                value = " ".join(str(v) for v in value if v)

            text = str(value).strip()
            if text:
                parts.append(text)

        combined_text = " . ".join(parts)
        return combined_text

    def resolve_image_path(self, item: Dict[str, Any]) -> Optional[Path]:
        """
        JSONL 아이템에서 사용할 로컬 이미지 경로 결정.

          - 파일명이 "{asin}.jpg" 형식으로 저장되어 있음
          - asin은 parent_asin > asin 순으로 사용
        """
        if not self.use_image:
            return None

        key = item.get("parent_asin") or item.get("asin") or item.get("id")
        if not key:
            return None

        img_path = self.image_root / f"{key}.jpg"
        if img_path.exists():
            return img_path

        img_path = self.image_root / f"{key}.png"
        if img_path.exists():
            return img_path

        return None

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
        return x / norms

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        embs = []
        for i in tqdm(
            range(0, len(texts), self.text_batch_size), desc="Encoding texts"
        ):
            batch = texts[i : i + self.text_batch_size]
            inputs = self.processor(
                text=batch,
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            with torch.no_grad():
                out = self.model.get_text_features(**inputs)
            embs.append(out.cpu().numpy())
        embs = np.concatenate(embs, axis=0)
        embs = self._normalize(embs)
        return embs

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        embs = []
        for i in tqdm(
            range(0, len(images), self.image_batch_size), desc="Encoding images"
        ):
            batch = images[i : i + self.image_batch_size]
            inputs = self.processor(
                text=None,
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                out = self.model.get_image_features(**inputs)
            embs.append(out.cpu().numpy())
        embs = np.concatenate(embs, axis=0)
        embs = self._normalize(embs)
        return embs

    def _process_batch(self, items: List[Dict[str, Any]]) -> None:
        """
        JSONL 일부 배치(items)에 대해:
        - 텍스트/이미지 경로 준비
        - 임베딩 계산
        - FAISS 인덱스에 add
        - metadata에 메타데이터 추가
        """
        metas: List[Dict[str, Any]] = []

        if self.embedding_mode == "text":
            # 텍스트 전용 인덱스
            texts: List[str] = []
            for item in items:
                text = self.build_text_for_embedding(item)
                if not text.strip():
                    continue
                asin = item.get("parent_asin") or item.get("asin") or item.get("id")
                img_path_str: Optional[str] = None
                if asin:
                    jpg_path = self.image_root / f"{asin}.jpg"
                    png_path = self.image_root / f"{asin}.png"
                    if jpg_path.exists():
                        img_path_str = str(jpg_path)
                    elif png_path.exists():
                        img_path_str = str(png_path)

                meta = {
                    "asin": item.get("parent_asin")
                    or item.get("asin")
                    or item.get("id"),
                    "text": text,
                    "image_path": img_path_str,
                }
                texts.append(text)
                metas.append(meta)

            if not texts:
                return

            embs = self.encode_texts(texts).astype("float32")
        else:
            # 이미지 전용 인덱스
            image_paths: List[Path] = []
            metas_tmp: List[Dict[str, Any]] = []

            for item in items:
                img_path = self.resolve_image_path(item)
                if img_path is None:
                    continue
                meta = {
                    "asin": item.get("parent_asin")
                    or item.get("asin")
                    or item.get("id"),
                    "text": None,
                    "image_path": str(img_path),
                }
                image_paths.append(img_path)
                metas_tmp.append(meta)

            if not image_paths:
                return

            # 실제 열리는 이미지만 사용
            pil_images: List[Image.Image] = []
            metas = []
            for img_path, meta in zip(image_paths, metas_tmp):
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                pil_images.append(img)
                metas.append(meta)

            if not pil_images:
                return

            embs = self.encode_images(pil_images).astype("float32")

        if self.index is None:
            dim = embs.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embs)
        self.metadata.extend(metas)

    # 인덱스 빌드
    def build_index(self, max_items: Optional[int] = None, chunk_size: int = 1000):
        """
         JSONL을 읽어서:
        - 텍스트/이미지 정보 준비
        - Fashion-CLIP 임베딩 계산
        - FAISS 인덱스(self.index)와 metadata(self.metadata) 구성
        """
        self.index = None
        self.metadata = []

        shard_id = self.shard_id
        num_shards = self.num_shards

        if (shard_id is None) != (num_shards is None):
            raise ValueError(
                "shard_id와 num_shards는 둘 다 설정하거나 둘 다 None이어야 합니다."
            )

        shard_item_count = 0  # 이 shard에서 처리한 아이템 수

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            batch_items: List[Dict[str, Any]] = []

            for line_idx, line in enumerate(f):
                if num_shards is not None and shard_id is not None:
                    if line_idx % num_shards != shard_id:
                        continue

                item = json.loads(line)
                batch_items.append(item)
                shard_item_count += 1

                # max_items 제한 (이 shard 기준)
                if max_items is not None and shard_item_count >= max_items:
                    break

                # chunk_size마다 처리
                if len(batch_items) >= chunk_size:
                    self._process_batch(batch_items)
                    batch_items = []

            # 마지막 남은 배치 처리
            if batch_items:
                self._process_batch(batch_items)

        if self.index is None:
            raise RuntimeError(
                "No embeddings were added. Check your data/filtering/shard settings."
            )
        print(
            f"[INFO] Index built with {self.index.ntotal} vectors "
            f"(shard_items={shard_item_count})."
        )

    # 저장
    def save(self):
        if self.index is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")

        # 인덱스 저장
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        print(f"[INFO] Saved index to {self.index_path}")

        # 메타데이터 저장(JSONL)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"[INFO] Saved metadata to {self.meta_path}")
