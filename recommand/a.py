import json
import numpy as np

# 1) 텍스트 임베딩 npz 로드
npz = np.load(
    "/home/lsy/multimodal-fashion-recommender-system/data/item_clip_text_embeds.npz",
    allow_pickle=True,
)

text_item_ids = set(npz["item_ids"].tolist())
print(f"[TEXT] 임베딩 아이템 수: {len(text_item_ids):,}")

# 2) ALS 학습에 쓴 jsonl 로드 (Amazon_Fashion_v1_for_als.jsonl or train.jsonl)
rows = []
with open(
    "/home/lsy/multimodal-fashion-recommender-system/data/Amazon_Fashion_v1_for_als.jsonl",
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        pa = obj.get("PARENT_ASIN") or obj.get("parent_asin")
        if pa is not None:
            rows.append(pa)

als_items = set(rows)
print(f"[ALS] 아이템 수: {len(als_items):,}")

# 3) 교집합
inter = als_items & text_item_ids
print(f"[COVERAGE] ALS 아이템 중 텍스트 임베딩이 있는 아이템 수: {len(inter):,}")
print(f"[COVERAGE] 비율: {len(inter) / len(als_items):.3f}")
