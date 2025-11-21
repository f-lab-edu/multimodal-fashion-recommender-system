# scripts/build_index.py
from pathlib import Path
import argparse

from fashion_core.fashion_embedder import FashionEmbeddingBuilder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Fashion-CLIP embeddings and FAISS index from JSONL and images."
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        required=True,
        help="meta_Amazon_Fashion_v1.jsonl 경로",
    )
    parser.add_argument(
        "--image-root", type=Path, required=True, help="로컬 이미지 폴더 경로"
    )
    parser.add_argument(
        "--index-path", type=Path, required=True, help="저장할 FAISS 인덱스 파일 경로"
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        required=True,
        help="생성할 메타데이터(JSONL) 경로",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="patrickjohncyh/fashion-clip",
        help="사용할 CLIP 기반 패션 모델 이름 (HuggingFace)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="테스트용으로 앞에서 N개만 인덱싱 (기본: 전체)",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="이미지 임베딩 없이 텍스트만 사용하려면 지정",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu 중 지정 (기본: 자동 감지)",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=32,
        help="텍스트 임베딩 배치 크기",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=16,
        help="이미지 임베딩 배치 크기",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="샤드 인덱스 (0-based). num-shards와 함께 사용",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="전체 샤드 개수. shard-id와 함께 사용",
    )
    parser.add_argument(
        "--embedding-mode",
        type=str,
        choices=["text", "image"],
        default="text",
        help="인덱싱 모드 선택: text(텍스트 인덱스) / image(이미지 인덱스)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    builder = FashionEmbeddingBuilder(
        jsonl_path=args.jsonl_path,
        image_root=args.image_root,
        index_path=args.index_path,
        meta_path=args.meta_path,
        model_name=args.model_name,
        text_batch_size=args.text_batch_size,
        image_batch_size=args.image_batch_size,
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        embedding_mode=args.embedding_mode,
    )
    builder.build_index(max_items=args.max_items)
    builder.save()


if __name__ == "__main__":
    main()
