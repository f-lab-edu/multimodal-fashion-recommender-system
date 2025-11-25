# fashion_core/search_utils.py
from __future__ import annotations

from typing import Any, Dict, List

from .search_results import SearchHit
import sqlite3


def load_item_meta_for_ids(
    conn: sqlite3.Connection,
    item_ids: List[int],
) -> Dict[int, Dict[str, Any]]:
    """items 테이블에서 주어진 id들의 메타 정보를 딕셔너리로 반환."""
    if not item_ids:
        return {}

    placeholders = ",".join("?" for _ in item_ids)
    rows = conn.execute(
        f"""
        SELECT
            id,
            parent_asin,
            title,
            store,
            image_main_url
        FROM items
        WHERE id IN ({placeholders})
        """,
        item_ids,
    ).fetchall()

    meta_map: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        iid = int(r["id"])
        meta_map[iid] = {
            "item_id": iid,
            "asin": r["parent_asin"],
            "title": r["title"],
            "store": r["store"],
            "image_url": r["image_main_url"],
        }
    return meta_map


def deduplicate_hits_by_asin(
    hits: List[SearchHit],
    top_k: int,
) -> List[SearchHit]:
    """
    asin 기준으로 중복 제거 + top_k 개까지만 남기고,
    남은 결과에 대해 rank를 1부터 다시 부여.
    """
    deduped: List[SearchHit] = []
    seen_asin: set[str] = set()

    for h in hits:
        key = h.asin or f"ITEM-{h.item_id}"
        if key in seen_asin:
            continue
        seen_asin.add(key)
        deduped.append(h)
        if len(deduped) >= top_k:
            break

    for i, h in enumerate(deduped, start=1):
        h.rank = i

    return deduped
