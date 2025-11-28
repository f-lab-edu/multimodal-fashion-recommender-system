# recommand/train_two_tower.py
from __future__ import annotations

from pathlib import Path
import argparse

from recommand.two_tower_trainer import TwoTowerTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-Tower (CLIP 기반) 추천 모델 학습 스크립트"
    )
    p.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="SQLite DB 경로 (예: data/fashion_items.db)",
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
        help="훈련용 ALS jsonl 경로 (예: data/als_splits/train.jsonl)",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help="Two-tower 모델을 저장할 경로 (예: models/two_tower_v1.pt)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    trainer = TwoTowerTrainer(
        db_path=args.db_path,
        image_index_path=args.image_index_path,
        train_jsonl=args.train_jsonl,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
    )
    trainer.train(out_path=args.out_path)


if __name__ == "__main__":
    main()
