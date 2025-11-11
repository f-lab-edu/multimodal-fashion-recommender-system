from __future__ import annotations
import re
from typing import Dict, List, Tuple


class Normalizer:
    """한→영 동의어 치환 + 소문자화 + 공백 정리"""

    def __init__(self, synonyms: Dict[str, str]):
        # 긴 키워드 우선 치환
        items = sorted(synonyms.items(), key=lambda kv: len(kv[0]), reverse=True)
        self.patterns: List[Tuple[re.Pattern, str]] = [
            (
                re.compile(
                    rf"(?<![가-힣A-Za-z0-9]){re.escape(k)}(?![가-힣A-Za-z0-9])",
                    re.IGNORECASE,
                ),
                v,
            )
            for k, v in items
        ]

    def __call__(self, text: str) -> str:
        t = text
        for pat, val in self.patterns:
            t = pat.sub(val, t)
        return re.sub(r"\s+", " ", t).strip().lower()
