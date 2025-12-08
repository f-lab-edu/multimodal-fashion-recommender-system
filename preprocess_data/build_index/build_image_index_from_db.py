# scripts/build_image_index_from_db.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Iterable, Tuple
import logging

import faiss
import numpy as np
import sqlite3
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPModel, CLIPProcessor


FASHION_CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"

logger = logging.getLogger(__name__)


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


def _resolve_device(arg_device: Optional[str]) -> str:
    if arg_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg_device


def _log_config(
    db_path: Path,
    image_root: Path,
    index_path: Path,
    model_name: str,
    device: str,
) -> None:
    logger.info("[INFO] DB         = %s", db_path)
    logger.info("[INFO] IMAGE_ROOT = %s", image_root)
    logger.info("[INFO] INDEX_PATH = %s", index_path)
    logger.info("[INFO] MODEL      = %s", model_name)
    logger.info("[INFO] DEVICE     = %s", device)


def _validate_paths(db_path: Path, image_root: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")


def _load_items_with_images(
    db_path: Path, max_items: Optional[int]
) -> List[sqlite3.Row]:
    with sqlite3.connect(str(db_path)) as conn:
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

    total_candidates = len(rows)
    logger.info("[INFO] items with image_main_url = %s", total_candidates)

    if max_items is not None:
        rows = rows[:max_items]
        logger.info(
            "[INFO] limiting to first %s items (max_items=%s)",
            len(rows),
            max_items,
        )

    return rows


def _init_faiss_index(dim: int) -> faiss.IndexIDMap:
    base_index = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap(base_index)


def _add_batch_to_index(
    index: Optional[faiss.IndexIDMap],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    batch_ids: List[int],
    batch_images: List[Image.Image],
) -> Tuple[Optional[faiss.IndexIDMap], int]:
    """배치 하나를 임베딩 후 인덱스에 추가하고, 추가된 개수를 반환."""
    if not batch_images:
        return index, 0

    embs = encode_image_batch(model, processor, device, batch_images)

    if index is None:
        dim = embs.shape[1]
        index = _init_faiss_index(dim)

    ids = np.array(batch_ids, dtype="int64")
    index.add_with_ids(embs, ids)
    return index, len(batch_images)


def _build_index_from_rows(
    rows: Iterable[sqlite3.Row],
    image_root: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    batch_size: int,
) -> faiss.IndexIDMap:
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
            logger.warning("[WARN] failed to open image for item_id=%s: %s", item_id, e)
            continue

        batch_ids.append(item_id)
        batch_images.append(img)

        if len(batch_images) >= batch_size:
            index, added = _add_batch_to_index(
                index, model, processor, device, batch_ids, batch_images
            )
            added_count += added
            batch_ids = []
            batch_images = []

    # 마지막 남은 배치 처리
    index, added = _add_batch_to_index(
        index, model, processor, device, batch_ids, batch_images
    )
    added_count += added

    if index is None or index.ntotal == 0:
        raise RuntimeError(
            "No image embeddings were added. Check image_root / DB filtering."
        )

    logger.info(
        "[INFO] Built image index with %s vectors (added_count=%s).",
        index.ntotal,
        added_count,
    )
    return index


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    args = parse_args()

    db_path: Path = args.db_path
    image_root: Path = args.image_root
    index_path: Path = args.index_path
    model_name: str = args.model_name

    device = _resolve_device(args.device)
    _log_config(db_path, image_root, index_path, model_name, device)
    _validate_paths(db_path, image_root)

    rows = _load_items_with_images(db_path, args.max_items)

    # 2) CLIP 모델 로드
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 3) 인덱스 빌드
    index = _build_index_from_rows(
        rows=rows,
        image_root=image_root,
        model=model,
        processor=processor,
        device=device,
        batch_size=args.batch_size,
    )

    # 4) 인덱스 저장
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("[INFO] Saved FAISS index to %s", index_path)


if __name__ == "__main__":
    main()
