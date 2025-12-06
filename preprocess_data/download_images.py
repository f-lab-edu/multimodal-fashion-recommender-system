# scripts/download_images.py
# meta_with_garment_label.jsonl을 읽어 Fashion 이미지들을 로컬에 캐시
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
from pathlib import Path
from typing import Optional, Tuple
import logging

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

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


def _score_image(img: dict) -> Tuple[int, Optional[str]]:
    """
    한 이미지에 대해 우선순위와 선택할 URL을 계산한다.
    점수가 높을수록 더 좋은 품질/우선순위.
    """
    is_main = img.get("variant") == "MAIN"

    # 1순위: MAIN + hi_res
    if is_main and img.get("hi_res"):
        return 4, img["hi_res"]

    # 2순위: MAIN + large
    if is_main and img.get("large"):
        return 3, img["large"]

    # 3순위: 아무거나 hi_res
    if img.get("hi_res"):
        return 2, img["hi_res"]

    # 4순위: 아무거나 large
    if img.get("large"):
        return 1, img["large"]

    # 후보 없음
    return 0, None


def pick_best_image_url(item: dict) -> Optional[str]:
    images = item.get("images") or []
    if not images:
        return None

    best_score = 0
    best_url: Optional[str] = None

    for img in images:
        score, url = _score_image(img)
        # 동일 점수면 먼저 나온 이미지를 유지하기 위해 '>'만 사용
        if url is not None and score > best_score:
            best_score = score
            best_url = url

    return best_url


def download_one_image(asin: str, url: str, dest: Path) -> bool:
    if dest.exists():
        # 이미 받아둔 경우
        return False

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        logger.warning("[WARN] failed to download %s from %s: %s", asin, url, e)
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

    logging.basicConfig(  # ✅ 기본 로깅 설정
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    input_jsonl: Path = args.jsonl_path
    image_root: Path = args.image_root

    image_root.mkdir(parents=True, exist_ok=True)

    total_lines = sum(1 for _ in input_jsonl.open("r", encoding="utf-8"))
    logger.info(
        "[download_images] reading %d lines from %s",
        total_lines,
        input_jsonl,
    )
    logger.info(
        "[download_images] saving images under %s",
        image_root,
    )

    seen_keys = set()
    jobs: list[tuple[str, str]] = []
    skipped_no_url = 0

    # 1단계: 다운로드 대상(job) 수집
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Scanning JSONL"):
            item = json.loads(line)

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

            jobs.append((key, url))

    logger.info(
        "[download_images] total jobs=%d, skipped_no_url=%d",
        len(jobs),
        skipped_no_url,
    )

    # 2단계: 병렬 다운로드
    downloaded = 0

    def _job_wrapper(job: tuple[str, str]) -> bool:
        key, url = job
        dest = image_root / f"{key}.jpg"
        return download_one_image(key, url, dest)

    max_workers = 16

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_job_wrapper, job) for job in jobs]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            try:
                if fut.result():
                    downloaded += 1
            except Exception as e:
                # 혹시라도 download_one_image에서 처리 안 된 예외가 튀어나오면
                logger.warning("[WARN] unexpected error in worker: %s", e)

    logger.info(
        "[download_images] downloaded=%d, skipped_no_url=%d",
        downloaded,
        skipped_no_url,
    )


if __name__ == "__main__":
    main()
