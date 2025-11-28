# recommand/train_als.py
from __future__ import annotations

from pathlib import Path
import argparse

from recommand.als_trainer import ALSTrainer

"""
예시:

python -m recommand.train_als \
    --train-jsonl data/als_splits_v2/train.jsonl \
    --val-jsonl   data/als_splits_v2/test.jsonl \
    --model-dir   models/als_v2_es \
    --image-index-path data/image_index_with_ids.faiss \
    --db-path data/fashion_items.db \
    --eval-k 200 \
    --patience 3
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Implicit ALS 학습 스크립트 (train jsonl → ALS 모델 저장, 선택적으로 early stopping)"
    )
    p.add_argument(
        "--train-jsonl",
        type=Path,
        required=True,
        help="훈련용 ALS jsonl 경로 (예: data/als_splits/train.jsonl)",
    )
    p.add_argument(
        "--val-jsonl",
        type=Path,
        default=None,
        help="early stopping에 사용할 validation ALS jsonl 경로 (예: data/als_splits/test.jsonl)",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="ALS 모델 및 매핑을 저장할 디렉토리 (예: models/als_v1)",
    )
    p.add_argument(
        "--image-index-path",
        type=Path,
        default=None,
        help="이미지 임베딩 FAISS 인덱스 경로 (예: data/image_index.faiss)",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite DB 경로 (예: data/fashion_items.db, items.id/parent_asin 매핑용)",
    )
    p.add_argument(
        "--eval-k",
        type=int,
        default=10,
        help="early stopping 시 HitRate@K 계산에 사용할 K (기본: 10)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=3,
        help="HitRate@K가 개선되지 않을 때 허용할 epoch 수 (기본: 3)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    train_path: Path = args.train_jsonl
    val_path: Path | None = args.val_jsonl
    model_dir: Path = args.model_dir
    image_index_path: Path | None = args.image_index_path
    db_path: Path | None = args.db_path

    trainer = ALSTrainer(
        factors=64,
        regularization=1e-2,
        iterations=20,
        alpha=40.0,
        use_gpu=False,
        random_state=42,
    )

    trainer.train_and_save(
        jsonl_path=train_path,
        out_dir=model_dir,
        image_index_path=image_index_path,
        db_path=db_path,
        val_jsonl_path=val_path,
        eval_k=args.eval_k,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
