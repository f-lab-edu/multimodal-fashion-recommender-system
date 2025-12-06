# scripts/search_multimodal_service_demo.py

"""
python -m fashion_core.search_multimodal_service_demo  \
      --db-path data/fashion_items.db   \
        --bm25-path data/bm25_from_item_docs.pkl \
            --image-index-path data/image_index_with_ids.faiss   \
                --query "black shirts"   --top-k 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fashion_core.multimodal_search_engine import MultiModalSearchEngine

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="BM25 + CLIP/FAISS Fusion 검색 (서비스 레이어 데모)"
    )
    p.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="SQLite DB 경로 (예: data/fashion_items.db)",
    )
    p.add_argument(
        "--bm25-path",
        type=Path,
        required=True,
        help="BM25 pickle 경로 (예: data/bm25_from_item_docs.pkl)",
    )
    p.add_argument(
        "--image-index-path",
        type=Path,
        required=True,
        help="이미지 FAISS 인덱스 경로 (예: data/fashion_image_index.faiss)",
    )
    p.add_argument(
        "--query",
        type=str,
        required=True,
        help='검색 쿼리 (예: "black shirts")',
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="최종 top-k (기본: 10)",
    )
    p.add_argument(
        "--stage1-factor",
        type=int,
        default=3,
        help="1차 후보군 배수 (기본: 3 → 각 모달리티에서 top_k*3 검색)",
    )
    p.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF 공식의 k (기본: 60)",
    )
    p.add_argument(
        "--w-text",
        type=float,
        default=1.0,
        help="텍스트(BM25) 가중치 (기본: 1.0)",
    )
    p.add_argument(
        "--w-image",
        type=float,
        default=1.0,
        help="이미지(CLIP) 가중치 (기본: 1.0)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu (기본: 자동)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    service = MultiModalSearchEngine(
        db_path=args.db_path,
        bm25_path=args.bm25_path,
        image_index_path=args.image_index_path,
        model_name="patrickjohncyh/fashion-clip",
        device=args.device,
        rrf_k=args.rrf_k,
        w_text=args.w_text,
        w_image=args.w_image,
    )

    try:
        results = service.search(
            query=args.query,
            top_k=args.top_k,
            stage1_factor=args.stage1_factor,
        )
    finally:
        service.close()

    logger.info("")
    logger.info("[QUERY] %r", args.query)
    logger.info("[INFO] top-%d fusion results:", args.top_k)
    logger.info("%s", "=" * 80)

    for r in results:
        title = (r["title"] or "")[:80] if r["title"] else "(no title)"
        logger.info("#%s", r["rank"])
        logger.info("  asin       : %s", r["asin"])
        logger.info("  item_id    : %s", r["item_id"])
        logger.info("  title      : %s", title)
        logger.info("  store      : %s", r["store"] or "")
        logger.info("  image_url  : %s", r["image_url"] or "")
        logger.info(
            "  bm25_rank  : %s  bm25_score=%s",
            r["bm25_rank"],
            r["bm25_score"],
        )
        logger.info(
            "  image_rank : %s  image_score=%s",
            r["image_rank"],
            r["image_score"],
        )
        logger.info("  rrf_score  : %.6f", r["rrf_score"])
        logger.info("%s", "-" * 80)


if __name__ == "__main__":
    main()
