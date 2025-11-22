# scripts/build_image_index_from_db.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import sqlite3
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPModel, CLIPProcessor


FASHION_CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"


def parse_args():
    p = argparse.ArgumentParser(
        description="SQLite DB(items.id) + images_by_id/{id}.jpg → "
        "Fashion-CLIP 이미지 임베딩 + FAISS(IndexIDMap, id=item_id)"
    )
    p.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="SQLite DB 경로 (예: data/fashion_items.db)",
    )
    p.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/images_by_id"),
        help="이미지 파일들이 있는 디렉토리 (예: data/images_by_id)",
    )
    p.add_argument(
        "--index-path",
        type=Path,
        required=True,
        help="저장할 FAISS 인덱스 파일 경로 (예: data/image_index.faiss)",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=FASHION_CLIP_MODEL_NAME,
        help="사용할 CLIP 모델 이름 (기본: patrickjohncyh/fashion-clip)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu (기본: 자동 감지)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="이미지 임베딩 배치 크기 (기본: 32)",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="앞에서 N개만 처리 (테스트용, 기본: 전체)",
    )
    return p.parse_args()


def encode_image_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    pil_images: List[Image.Image],
) -> np.ndarray:
    inputs = processor(
        text=None,
        images=pil_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        out = model.get_image_features(**inputs)

    embs = out.cpu().numpy().astype("float32")
    # L2 normalize → IP == cosine
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    embs = embs / norms
    return embs


def main():
    args = parse_args()

    db_path: Path = args.db_path
    image_root: Path = args.image_root
    index_path: Path = args.index_path
    model_name: str = args.model_name

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[INFO] DB         = {db_path}")
    print(f"[INFO] IMAGE_ROOT = {image_root}")
    print(f"[INFO] INDEX_PATH = {index_path}")
    print(f"[INFO] MODEL      = {model_name}")
    print(f"[INFO] DEVICE     = {device}")

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")

    # 1) DB에서 image_main_url 있는 item_id들만 가져오기
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, image_main_url
        FROM items
        WHERE image_main_url IS NOT NULL AND image_main_url != ''
        ORDER BY id
        """
    )
    rows = cur.fetchall()
    conn.close()

    total_candidates = len(rows)
    print(f"[INFO] items with image_main_url = {total_candidates}")

    if args.max_items is not None:
        rows = rows[: args.max_items]
        print(
            f"[INFO] limiting to first {len(rows)} items (max_items={args.max_items})"
        )

    # 2) CLIP 모델 로드
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 🔹 IndexFlatIP + IndexIDMap → DB의 id를 그대로 Faiss id로 사용
    index: Optional[faiss.IndexIDMap] = None

    batch_ids: List[int] = []
    batch_images: List[Image.Image] = []

    added_count = 0

    for row in tqdm(rows, desc="Preparing images"):
        item_id = int(row["id"])
        img_path = image_root / f"{item_id}.jpg"

        if not img_path.exists():
            # 이미지 파일이 없는 경우 스킵
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] failed to open image for item_id={item_id}: {e}")
            continue

        batch_ids.append(item_id)
        batch_images.append(img)

        if len(batch_images) >= args.batch_size:
            embs = encode_image_batch(model, processor, device, batch_images)

            if index is None:
                dim = embs.shape[1]
                base_index = faiss.IndexFlatIP(dim)
                index = faiss.IndexIDMap(base_index)

            ids = np.array(batch_ids, dtype="int64")
            index.add_with_ids(embs, ids)

            added_count += len(batch_images)
            batch_ids = []
            batch_images = []

    # 마지막 남은 배치 처리
    if batch_images:
        embs = encode_image_batch(model, processor, device, batch_images)

        if index is None:
            dim = embs.shape[1]
            base_index = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap(base_index)

        ids = np.array(batch_ids, dtype="int64")
        index.add_with_ids(embs, ids)

        added_count += len(batch_images)

    if index is None or index.ntotal == 0:
        raise RuntimeError(
            "No image embeddings were added. Check image_root / DB filtering."
        )

    print(
        f"[INFO] Built image index with {index.ntotal} vectors (added_count={added_count})."
    )

    # 3) 인덱스 저장
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"[INFO] Saved FAISS index to {index_path}")


if __name__ == "__main__":
    main()
