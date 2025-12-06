# fashion_core/search_multimodal_personally_demo.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fashion_core.personalized_search_engine import PersonalizedSearchEngine

"""
python -m fashion_core.search_multimodal_personally_demo \
  --db-path data/fashion_items.db \
  --bm25-path data/bm25_from_item_docs.pkl \
  --image-index-path data/image_index_with_ids.faiss \
  --query "black linen shirts" \
  --top-k 10 \
  --user-id AE23BYWB52METWQVHSPN3MKN7AJA \
  --als-config config/als_model.yaml \
  --log-level INFO

"""
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="BM25 + CLIP/FAISS Fusion + (optional) ALS 재랭킹 데모"
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
        help="이미지 FAISS 인덱스 경로 (예: data/image_index_with_ids.faiss)",
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
    p.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="ALS 재랭킹에 사용할 user_id. 지정하지 않으면 개인화 없이 fusion 결과만 사용.",
    )
    p.add_argument(
        "--als-config",
        type=Path,
        default=Path("config/als_model.yaml"),
        help="ALS 모델 설정 YAML 경로 (기본: config/als_model.yaml)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    return p.parse_args()


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def main():
    args = parse_args()
    setup_logging(args.log_level)

    engine = PersonalizedSearchEngine.from_paths(
        db_path=args.db_path,
        bm25_path=args.bm25_path,
        image_index_path=args.image_index_path,
        als_config_path=args.als_config,
        model_name="patrickjohncyh/fashion-clip",
        device=args.device,
        rrf_k=args.rrf_k,
        w_text=args.w_text,
        w_image=args.w_image,
    )

    try:
        results = engine.search(
            query=args.query,
            top_k=args.top_k,
            stage1_factor=args.stage1_factor,
            user_id=args.user_id,  # None이면 ALS 스킵, 있으면 재랭킹
        )
    finally:
        engine.close()

    logger.info("Query: %r", args.query)
    if args.user_id:
        logger.info(
            "Showing top-%d personalised results for user_id=%s",
            args.top_k,
            args.user_id,
        )
    else:
        logger.info("Showing top-%d fusion results (no ALS)", args.top_k)

    for i, r in enumerate(results, start=1):
        title = (r["title"] or "")[:80] if r["title"] else "(no title)"
        image_url = r.get("image_url") or ""
        store = r.get("store") or ""

        logger.info("#%d", i)
        logger.info("  asin      = %s", r["asin"])
        logger.info("  title     = %s", title)
        logger.info("  store     = %s", store)
        logger.info("  image_url = %s", image_url)
        logger.info(
            "  rrf=%.4f final=%.4f",
            float(r["rrf_score"]),
            float(r.get("score_final", r["rrf_score"])),
        )


if __name__ == "__main__":
    main()
