# scripts/search_fusion.py
from __future__ import annotations

import argparse
from pathlib import Path

from fashion_core.fashion_search_engine import FusionSearchEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fusion search over text/image sharded Fashion-CLIP indices (RRF)."
    )
    parser.add_argument(
        "--text-prefix",
        type=Path,
        required=True,
        help="텍스트 인덱스 prefix (예: data/fashion_index_text)",
    )
    parser.add_argument(
        "--image-prefix",
        type=Path,
        required=True,
        help="이미지 인덱스 prefix (예: data/fashion_index_image)",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="검색 쿼리 (한글/영어 모두 가능)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="최종 반환할 결과 개수",
    )
    parser.add_argument(
        "--per-shard-k-factor",
        type=int,
        default=2,
        help="각 샤드에서 top_k * factor 만큼 더 많이 뽑아서 fusion",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF 공식의 k 값 (1/(k+rank))",
    )
    parser.add_argument(
        "--w-text",
        type=float,
        default=1.0,
        help="텍스트 랭킹의 가중치",
    )
    parser.add_argument(
        "--w-image",
        type=float,
        default=1.0,
        help="이미지 랭킹의 가중치",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu (기본: 자동 감지)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="patrickjohncyh/fashion-clip",
        help="Fashion-CLIP 모델 이름",
    )
    parser.add_argument(
        "--stage1-factor",
        type=int,
        default=3,
        help="1차 후보군 배수 (텍스트/이미지 각각 top_k * factor까지 가져옴)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    engine = FusionSearchEngine.from_prefixes(
        text_base_prefix=args.text_prefix,
        image_base_prefix=args.image_prefix,
        model_name=args.model_name,
        device=args.device,
        rrf_k=args.rrf_k,
        w_text=args.w_text,
        w_image=args.w_image,
    )

    results = engine.search(
        query=args.query,
        top_k=args.top_k,
        per_shard_k_factor=args.per_shard_k_factor,
        stage1_factor=args.stage1_factor,
    )

    print("Global fused top-{} results".format(args.top_k))
    print("=" * 80)
    for r in results:
        print(
            f"[{r['rank']:02d}] RRF={r['rrf_score']:.4f}  "
            f"asin={r['asin']}  "
            f"text_rank={r['rank_text']}  img_rank={r['rank_image']}"
        )
        print(f"     text      : {r.get('text')}")
        print(f"     image_path: {r.get('image_path')}")
        print(
            f"     raw_scores: text={r['score_text_raw']}  img={r['score_image_raw']}"
        )
        print("-" * 80)


if __name__ == "__main__":
    main()
