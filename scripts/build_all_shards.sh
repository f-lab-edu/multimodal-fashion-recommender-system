#!/usr/bin/env bash
set -e

NUM_SHARDS_TEXT=4
NUM_SHARDS_IMAGE=8

JSONL_PATH="data/meta_Amazon_Fashion_v1.jsonl"
IMAGE_ROOT="data/images"
BASE_PREFIX="data/embeddings/fashion_index"

# 텍스트 임베딩
for SID in $(seq 0 $((NUM_SHARDS_TEXT - 1))); do
  echo "=== Building TEXT shard $SID / $NUM_SHARDS_TEXT ==="

  python -m scripts.build_index \
    --jsonl-path "$JSONL_PATH" \
    --image-root "$IMAGE_ROOT" \
    --index-path "${BASE_PREFIX}_text_shard_${SID}.faiss" \
    --meta-path  "${BASE_PREFIX}_text_shard_${SID}_meta.jsonl" \
    --shard-id "${SID}" \
    --num-shards "${NUM_SHARDS_TEXT}" \
    --text-batch-size 32 \
    --image-batch-size 16 \
    --embedding-mode text
done

# 이미지 임베딩
# for SID in $(seq 0 $((NUM_SHARDS_IMAGE - 1))); do
#   echo "=== Building IMAGE shard $SID / $NUM_SHARDS_IMAGE ==="

#   python -m scripts.build_index \
#     --jsonl-path "$JSONL_PATH" \
#     --image-root "$IMAGE_ROOT" \
#     --index-path "${BASE_PREFIX}_image_shard_${SID}.faiss" \
#     --meta-path  "${BASE_PREFIX}_image_shard_${SID}_meta.jsonl" \
#     --shard-id "${SID}" \
#     --num-shards "${NUM_SHARDS_IMAGE}" \
#     --text-batch-size 32 \
#     --image-batch-size 16 \
#     --embedding-mode image
# done

# echo "=== All shards done ==="
