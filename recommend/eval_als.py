# recommand/eval_als.py
from __future__ import annotations

from pathlib import Path
import argparse
import logging

from recommend.als_evaluator import ALSEvaluator


"""
사용 예시:

# 이미지 임베딩 초기화 ALS 평가
python -m recommend.eval_als \
  --train-jsonl data/als_splits_v3/train.jsonl \
  --test-jsonl  data/als_splits_v3/test.jsonl \
  --model-dir   models/als_warm_split_v3 \
  --k 50

"""

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Implicit ALS 평가 스크립트 " "(train/test jsonl → Precision@K / MAP@K)"
        ),
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        required=True,
        help="훈련에 사용한 ALS jsonl 경로 (예: data/als_splits/train.jsonl)",
    )
    p.add_argument(
        "--test-jsonl",
        type=Path,
        required=True,
        help="테스트용 ALS jsonl 경로 (예: data/als_splits/test.jsonl)",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="학습된 ALS 모델 디렉토리 (예: models/als_v1)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="상위 K개 추천에 대해 Precision@K / MAP@K를 계산 (기본값: 10)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    args = parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {args.model_dir}")

    logger.info("[EVAL] 설정")
    logger.info("- train_jsonl: %s", args.train_jsonl)
    logger.info("- test_jsonl : %s", args.test_jsonl)
    logger.info("- model_dir  : %s", args.model_dir)
    logger.info("- K          : %d", args.k)

    evaluator = ALSEvaluator(model_dir=args.model_dir)

    # 1) train.jsonl로 user_items_csr 생성
    evaluator.build_user_items_from_train(args.train_jsonl)

    # 2) test.jsonl로 평가
    metrics = evaluator.evaluate_on_test(args.test_jsonl, k=args.k)

    logger.info("=== 최종 평가 결과 ===")
    logger.info("- 평가 유저 수        : %s", f"{metrics['num_users']:,}")
    logger.info("- Recall@%2d          : %.4f", args.k, metrics["recall_at_k"])
    logger.info("- HitRate@%2d         : %.4f", args.k, metrics["hit_rate"])
    logger.info("- MAP@%2d             : %.4f", args.k, metrics["map_at_k"])


if __name__ == "__main__":
    main()
