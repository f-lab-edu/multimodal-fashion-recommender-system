from __future__ import annotations
from typing import Set


class StopwordFilter:
    def __init__(self, stopwords: Set[str]):
        self.stop = set(stopwords)

    def __call__(self, text: str) -> str:
        return " ".join(tok for tok in text.split() if tok not in self.stop)
