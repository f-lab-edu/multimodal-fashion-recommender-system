# fashion_core/search_results.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence


TokenizeFn = Callable[[str], Sequence[str]]


@dataclass
class SearchHit:
    item_id: int
    asin: Optional[str]
    score: float
    rank: int
    title: Optional[str] = None
    store: Optional[str] = None
    image_url: Optional[str] = None  # DB에 저장된 대표 이미지(or 로컬 경로)


@dataclass
class FusionHit:
    asin: str
    db_item_id: Optional[int]
    title: Optional[str]
    store: Optional[str]
    image_url: Optional[str]
    bm25_rank: Optional[int]
    bm25_score_raw: Optional[float]
    image_rank: Optional[int]
    image_score_raw: Optional[float]
    rrf_score: float
    rank: Optional[int] = None  # 최종 순위
