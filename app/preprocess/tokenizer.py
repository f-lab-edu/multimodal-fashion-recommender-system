from __future__ import annotations
import re
from typing import Set


class Tokenizer:
    def __call__(self, text: str) -> Set[str]:
        parts = re.split(r"\s+", text.strip())
        toks: Set[str] = set()
        for p in parts:
            if not p:
                continue
            toks.add(p)
            if "-" in p:
                toks.update(p.split("-"))
        return toks
