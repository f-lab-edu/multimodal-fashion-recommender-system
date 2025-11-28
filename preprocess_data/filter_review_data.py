# preprocess_data/filter_review_data.py
# 검색 기능에 사용했던 데이터 필터링

from __future__ import annotations

import json
from pathlib import Path
from typing import Set


def load_valid_parent_asins(meta_path: Path) -> Set[str]:
    valid = set()
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            # meta 파일에서 PARENT_ASIN 키 이름이 뭔지에 따라 조합
            parent_asin = obj.get("parent_asin")

            if parent_asin:
                valid.add(parent_asin)

    print(f"[meta] 유효한 PARENT_ASIN 수: {len(valid):,}")
    return valid


def filter_reviews_by_meta(
    reviews_path: Path,
    output_path: Path,
    valid_parent_asins: Set[str],
) -> None:
    total = 0
    kept = 0
    skipped_no_parent = 0
    skipped_not_in_meta = 0

    with reviews_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1
            obj = json.loads(line)

            parent_asin = (
                obj.get("PARENT_ASIN")
                or obj.get("parent_asin")
                or obj.get("parentAsin")
            )

            if not parent_asin:
                skipped_no_parent += 1
                continue

            if parent_asin not in valid_parent_asins:
                skipped_not_in_meta += 1
                continue

            # 조건 통과 → 출력 파일에 그대로 기록
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print("=== 리뷰 필터링 결과 ===")
    print(f"- 전체 리뷰 수           : {total:,}")
    print(f"- meta에 있는 아이템 리뷰: {kept:,}")
    print(f"- parent_asin 없는 행    : {skipped_no_parent:,}")
    print(f"- meta에 없는 아이템 리뷰: {skipped_not_in_meta:,}")
    print(f"- 최종 남은 비율         : {kept/total*100:.2f}%")


def main():
    meta_path = Path("data/meta_Amazon_Fashion_v1.jsonl")
    reviews_path = Path("data/Amazon_Fashion.jsonl")
    output_path = Path("data/Amazon_Fashion_v1.jsonl")

    # 1) meta에서 유효한 PARENT_ASIN set 만들기
    valid_parent_asins = load_valid_parent_asins(meta_path)

    # 2) 리뷰에서 meta에 있는 아이템만 필터링
    filter_reviews_by_meta(
        reviews_path=reviews_path,
        output_path=output_path,
        valid_parent_asins=valid_parent_asins,
    )


if __name__ == "__main__":
    main()
