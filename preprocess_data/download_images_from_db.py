# scripts/download_images_from_items.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sqlite3
from pathlib import Path
import logging

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "Mozilla/5.0 (compatible; FashionCLIP-Downloader/1.0)"}
)


def download_one_image(item_id: int, url: str, dest: Path) -> bool:
    if dest.exists():
        # 이미 받아둔 경우
        return False

    try:
        resp = SESSION.get(url, timeout=10)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        logger.warning(
            "[WARN] failed to download item_id=%s from %s: %s",
            item_id,
            url,
            e,
        )
        return False


def parse_args():
    p = argparse.ArgumentParser(
        description="items.image_main_url 를 이용해 대표 이미지를 id.jpg 로 다운로드"
    )
    p.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="SQLite DB 경로 (예: data/fashion_items.db)",
    )
    p.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/images_by_id"),
        help="이미지를 저장할 디렉토리 (기본: data/images_by_id)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="동시 다운로드 스레드 수 (기본: 16)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(  # ✅ 기본 로깅 설정
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    db_path: Path = args.db_path
    image_root: Path = args.image_root
    max_workers: int = args.max_workers

    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    image_root.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, image_main_url
        FROM items
        WHERE image_main_url IS NOT NULL AND image_main_url != ''
        """
    )
    rows = cur.fetchall()
    conn.close()

    jobs: list[tuple[int, str]] = [(row["id"], row["image_main_url"]) for row in rows]

    logger.info("[download_images_items] download_jobs=%d", len(jobs))

    downloaded = 0

    def _job_wrapper(job: tuple[int, str]) -> bool:
        item_id, url = job
        dest = image_root / f"{item_id}.jpg"
        return download_one_image(item_id, url, dest)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_job_wrapper, job) for job in jobs]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading",
        ):
            try:
                if fut.result():
                    downloaded += 1
            except Exception as e:
                logger.warning("[WARN] unexpected error in worker: %s", e)

    logger.info(
        "[download_images_items] downloaded=%d, skipped=%d",
        downloaded,
        len(jobs) - downloaded,
    )


if __name__ == "__main__":
    main()
