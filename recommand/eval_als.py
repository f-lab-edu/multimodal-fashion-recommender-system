# recommand/eval_als.py
from __future__ import annotations

from pathlib import Path
import argparse

from recommand.als_evaluator import ALSEvaluator


"""
사용 예시:

# 베이스라인 ALS 평가
python -m recommand.eval_als \
  --train-jsonl data/als_splits/train.jsonl \
  --test-jsonl  data/als_splits/test.jsonl \
  --model-dir   models/als_v1 \
  --k 10

# 이미지 임베딩 초기화 ALS 평가
python -m recommand.eval_als \
  --train-jsonl data/als_splits_v2/train.jsonl \
  --test-jsonl  data/als_splits_v2/test.jsonl \
  --model-dir   models/als_v2_es \
  --k 10
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Implicit ALS 평가 스크립트 (train/test jsonl → Recall@K / HitRate@K)"
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
        help="상위 K개 추천에 대해 Recall@K / HitRate@K를 계산 (기본값: 10)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {args.model_dir}")

    print("[EVAL] 설정")
    print(f"- train_jsonl: {args.train_jsonl}")
    print(f"- test_jsonl : {args.test_jsonl}")
    print(f"- model_dir  : {args.model_dir}")
    print(f"- K          : {args.k}")

    evaluator = ALSEvaluator(model_dir=args.model_dir)

    # 1) train.jsonl로 user_items_csr 생성
    evaluator.build_user_items_from_train(args.train_jsonl)

    # 2) test.jsonl로 평가
    metrics = evaluator.evaluate_on_test(args.test_jsonl, k=args.k)

    print("\n=== 최종 평가 결과 ===")
    print(f"- 평가 유저 수      : {metrics['num_users']:,}")
    print(f"- Recall@{args.k:>2}       : {metrics['recall_at_k']:.4f}")
    print(f"- HitRate@{args.k:>2}      : {metrics['hit_rate']:.4f}")
    print(f"- MAP@{args.k:>2}          : {metrics['map_at_k']:.4f}")


if __name__ == "__main__":
    main()
