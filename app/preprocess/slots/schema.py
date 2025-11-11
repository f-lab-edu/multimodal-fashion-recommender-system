from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, List, Iterable
import json


@dataclass
class Slot:
    name: str
    allowed: Set[str] = field(default_factory=set)
    aliases: Dict[str, str] = field(default_factory=dict)
    values: List[str] = field(default_factory=list)

    def normalize(self, token: str) -> str | None:
        t = (token or "").strip().lower()
        if not t:
            return None
        t = self.aliases.get(t, t)
        if self.allowed and t not in self.allowed:
            return None
        return t

    def add(self, token: str) -> bool:
        v = self.normalize(token)
        if v is None:
            return False
        if v not in self.values:
            self.values.append(v)
        return True

    def update(self, tokens: Iterable[str]) -> int:
        cnt = 0
        for t in tokens:
            cnt += int(self.add(t))
        return cnt

    def clear(self) -> None:
        self.values.clear()

    def to_dict(self) -> Dict:
        return {"name": self.name, "values": self.values}


@dataclass
class Slots:
    color: Slot
    material: Slot
    garment: Slot
    fit: Slot
    length: Slot
    season: Slot
    style: Slot

    def empty_like(self) -> "Slots":
        def clone(s: Slot):
            return Slot(s.name, set(s.allowed), dict(s.aliases), [])

        return Slots(*(clone(getattr(self, k)) for k in self.__dict__))

    def to_json(self) -> str:
        return json.dumps(
            {k: getattr(self, k).to_dict() for k in self.__dict__}, ensure_ascii=False
        )
