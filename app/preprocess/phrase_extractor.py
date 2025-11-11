from __future__ import annotations
import re
from typing import Set


class PhraseExtractor:
    def __init__(self, phrases: Set[str]):
        self.phrases = set(phrases)

    def __call__(self, text: str) -> Set[str]:
        found = set()
        for ph in self.phrases:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(ph)}(?![A-Za-z0-9])", text):
                found.add(ph)
        return found
