# scripts/download_images.py
# meta_with_garment_label.jsonl을 읽어 Fashion 이미지들을 로컬에 캐시

import json
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_JSONL = PROJECT_ROOT / "data" / "meta_with_garment_label.jsonl"
IMAGE_ROOT = PROJECT_ROOT / "data" / "images"

IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update(
    {
        # 대충 브라우저 흉내만 내줌 (User-Agent 없으면 가끔 막히기도 해서)
        "User-Agent": "Mozilla/5.0 (compatible; FashionCLIP-Downloader/1.0)"
    }
)


def pick_best_image_url(item: dict) -> Optional[str]:
    images = item.get("images") or []
    if not images:
        return None

    # 1순위: MAIN + hi_res
    for img in images:
        if img.get("variant") == "MAIN" and img.get("hi_res"):
            return img["hi_res"]

    # 2순위: MAIN + large
    for img in images:
        if img.get("variant") == "MAIN" and img.get("large"):
            return img["large"]

    # 3순위: 아무거나 hi_res
    for img in images:
        if img.get("hi_res"):
            return img["hi_res"]

    # 4순위: 아무거나 large
    for img in images:
        if img.get("large"):
            return img["large"]

    return None


def download_one_image(asin: str, url: str, dest: Path) -> bool:
    if dest.exists():
        # 이미 받아둔 경우
        return False

    try:
        resp = SESSION.get(url, timeout=10)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"[WARN] failed to download {asin} from {url}: {e}")
        return False


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="주어진 JSONL에서 이미지 URL을 읽어 로컬로 캐시합니다."
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        required=False,
        default=PROJECT_ROOT / "data" / "meta_Amazon_Fashion_v1.jsonl",
        help="이미지 정보를 포함한 JSONL 경로 (기본: data/meta_Amazon_Fashion_v1.jsonl)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=False,
        default=PROJECT_ROOT / "data" / "images",
        help="이미지를 저장할 디렉토리 (기본: data/images)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_jsonl: Path = args.jsonl_path
    image_root: Path = args.image_root

    image_root.mkdir(parents=True, exist_ok=True)

    total_lines = sum(1 for _ in input_jsonl.open("r", encoding="utf-8"))
    print(f"[download_images] reading {total_lines} lines from {input_jsonl}")
    print(f"[download_images] saving images under {image_root}")

    seen_keys = set()
    downloaded = 0
    skipped_no_url = 0

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines):
            item = json.loads(line)

            # parent_asin이 있으면 그걸 우선, 없으면 asin/id 사용
            key = item.get("parent_asin") or item.get("asin") or item.get("id")
            if not key:
                continue

            if key in seen_keys:
                continue
            seen_keys.add(key)

            url = pick_best_image_url(item)
            if not url:
                skipped_no_url += 1
                continue

            dest = image_root / f"{key}.jpg"
            ok = download_one_image(key, url, dest)
            if ok:
                downloaded += 1

    print(
        f"[download_images] downloaded={downloaded}, "
        f"skipped_no_url={skipped_no_url}"
    )


if __name__ == "__main__":
    main()
