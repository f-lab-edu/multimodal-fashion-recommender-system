# recommand/eval_two_tower.py
from __future__ import annotations

from pathlib import Path
import argparse

from recommand.two_tower_evaluator import TwoTowerEvaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-Tower (CLIP 기반) 추천 모델 평가 스크립트"
    )
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="학습된 Two-Tower 모델 경로 (예: models/two_tower_v1.pt)",
    )
    p.add_argument(
        "--image-index-path",
        type=Path,
        required=True,
        help="이미지 FAISS 인덱스 경로 (예: data/image_index_with_ids.faiss)",
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
        "--k",
        type=int,
        default=10,
        help="상위 K개 추천에 대해 Recall@K / HitRate@K를 계산 (default: 10)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("[TT-EVAL] 설정")
    print(f"- model_path      : {args.model_path}")
    print(f"- image_index_path: {args.image_index_path}")
    print(f"- train_jsonl     : {args.train_jsonl}")
    print(f"- test_jsonl      : {args.test_jsonl}")
    print(f"- K               : {args.k}")

    evaluator = TwoTowerEvaluator(
        model_path=args.model_path,
        image_index_path=args.image_index_path,
    )

    # 1) train.jsonl 기반으로 user_pos_items 구성 (이미 본 아이템 필터용)
    evaluator.build_user_pos_from_train(args.train_jsonl)

    # 2) test.jsonl로 평가
    metrics = evaluator.evaluate_on_test(args.test_jsonl, k=args.k)

    print("\n=== Two-Tower 최종 평가 결과 ===")
    print(f"- 평가 유저 수      : {metrics['num_users']:,}")
    print(f"- Recall@{args.k:2d} : {metrics['recall_at_k']:.4f}")
    print(f"- HitRate@{args.k:2d}: {metrics['hit_rate']:.4f}")


if __name__ == "__main__":
    main()
