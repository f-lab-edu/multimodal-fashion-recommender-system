from __future__ import annotations
from typing import Set


class RulesEngine:
    """경량 규칙: high+rise, 혼방(material-blend) 등"""

    def __init__(self, materials: Set[str]):
        self.materials = set(materials)

    def __call__(self, tokens: Set[str]) -> Set[str]:
        toks = set(t.lower() for t in tokens if t)
        # high + rise → high-rise
        if "high" in toks and "rise" in toks:
            toks.add("high-rise")
        # 혼방: material + 'blend' → '{material}-blend'
        if "blend" in toks:
            for m in self.materials:
                if m != "blend" and m in toks:
                    toks.add(f"{m}-blend")
        return toks
