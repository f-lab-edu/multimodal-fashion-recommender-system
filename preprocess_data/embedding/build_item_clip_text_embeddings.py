# preprocess_data/embedding/build_item_clip_text_embeddings.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd

import torch
from fashion_clip.fashion_clip import FashionCLIP  # 너가 이미 쓰고 있는 패키지 기준

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


# --------------------------
# 1) 메타 데이터 로드
# --------------------------
def load_meta(jsonl_path: Path) -> pd.DataFrame:
    """
    meta_Amazon_Fashion_v1.jsonl 형식 가정 (대략):
      - parent_asin 또는 PARENT_ASIN
      - asin
      - title
      - (있다면) garment_label, color, material 등
    """
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            parent_asin = obj.get("parent_asin")
            if parent_asin is None:
                continue

            title = obj.get("title") or ""

            rows.append(
                {
                    "PARENT_ASIN": str(parent_asin),
                    "title": str(title),
                }
            )

    df = pd.DataFrame(rows)
    print(f"[META] 로드 행 수: {len(df):,}")
    print(f"[META] PARENT_ASIN 수: {df['PARENT_ASIN'].nunique():,}")
    return df


# --------------------------
# 2) 아이템별 대표 텍스트 만들기
# --------------------------
def build_item_texts(df: pd.DataFrame) -> Dict[str, str]:
    """
    PARENT_ASIN 기준으로 그룹핑해서,
    아이템마다 CLIP에 넣을 한 줄짜리 설명 텍스트를 만든다.
    """
    item_texts: Dict[str, str] = {}

    for parent_asin, g in df.groupby("PARENT_ASIN"):
        row = g.iloc[0]
        title = (row.get("title") or row.get("product_title") or "").strip()
        if not title:
            continue

        item_texts[parent_asin] = title

    print(f"[META] 텍스트가 만들어진 아이템 수: {len(item_texts):,}")
    return item_texts


# --------------------------
# 3) Fashion-CLIP으로 인코딩
# --------------------------
def encode_texts_with_fclip(
    item_texts: Dict[str, str],
    batch_size: int = 64,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fashion-CLIP 텍스트 인코더로 item_texts를 임베딩.
    반환:
      - item_ids: (N,) PARENT_ASIN 문자열 배열
      - embeddings: (N, D) float32 (L2 정규화 완료)
    """
    print(f"[FCLIP] device: {device}")
    model = FashionCLIP("fashion-clip")

    # 1) 아이템 ID / 텍스트 리스트 생성
    item_ids: List[str] = list(item_texts.keys())
    texts: List[str] = [item_texts[iid] for iid in item_ids]
    print(f"[FCLIP] 텍스트 개수: {len(texts):,}, batch_size={batch_size}")

    # 2) 전체 텍스트를 한 번에 encode_text에 넣고,
    #    내부에서 batch_size 단위로 처리하도록 맡긴다.
    with torch.no_grad():
        emb = model.encode_text(
            texts,
            batch_size=batch_size,
        )
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()

    # 3) float32 + L2 정규화
    emb = emb.astype(np.float32)
    norms = np.linalg.norm(emb, ord=2, axis=-1, keepdims=True) + 1e-10
    emb = emb / norms

    embeddings = emb
    item_ids_arr = np.array(item_ids, dtype=object)

    print(f"[FCLIP] 최종 임베딩 shape: {embeddings.shape}")
    return item_ids_arr, embeddings


# --------------------------
# 4) main + CLI
# --------------------------
def main():
    import argparse

    p = argparse.ArgumentParser(description="Fashion-CLIP 텍스트 임베딩 생성 스크립트")
    p.add_argument(
        "--meta-jsonl",
        type=Path,
        default=Path("data/meta_Amazon_Fashion_v1.jsonl"),
        help="메타 JSONL 경로 (기본: data/meta_Amazon_Fashion_v1.jsonl)",
    )
    p.add_argument(
        "--out-npz",
        type=Path,
        default=Path("data/item_clip_text_embeds.npz"),
        help="임베딩 npz 저장 경로 (기본: data/item_clip_text_embeds.npz)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="인코딩 배치 크기",
    )
    args = p.parse_args()

    meta_path: Path = args.meta_jsonl
    out_path: Path = args.out_npz

    # 1) 메타 로드
    df_meta = load_meta(meta_path)

    # 2) 아이템별 텍스트 구성
    item_texts = build_item_texts(df_meta)

    # 3) Fashion-CLIP으로 임베딩
    item_ids, embeddings = encode_texts_with_fclip(
        item_texts,
        batch_size=args.batch_size,
    )

    # 4) 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        item_ids=item_ids,
        embeddings=embeddings,
    )
    print(f"[SAVE] saved text embeddings to {out_path}")


if __name__ == "__main__":
    main()
