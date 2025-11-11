from __future__ import annotations
import re
from typing import Dict, Set
from app.preprocess.slots.schema import Slots
from app.preprocess.slots.registry import load_all, load_default_all


# --------------------------
# 구성요소들(간단 구현)
# --------------------------
class Normalizer:
    """한→영 동의어 치환 + 소문자화 + 공백 정리"""

    def __init__(self, synonyms: Dict[str, str]):
        # 긴 키워드 먼저 치환되도록 정렬
        items = sorted(synonyms.items(), key=lambda kv: len(kv[0]), reverse=True)
        self._patterns = [
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
        for pat, val in self._patterns:
            t = pat.sub(val, t)
        return re.sub(r"\s+", " ", t).strip().lower()


class StopwordFilter:
    def __init__(self, stopwords: Set[str]):
        self._stop = set(stopwords)

    def __call__(self, text: str) -> str:
        return " ".join(tok for tok in text.split() if tok not in self._stop)


class PhraseExtractor:
    def __init__(self, phrases: Set[str]):
        self._phrases = set(phrases)

    def __call__(self, text: str) -> Set[str]:
        found = set()
        for ph in self._phrases:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(ph)}(?![A-Za-z0-9])", text):
                found.add(ph)
        return found


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


class RulesEngine:
    """경량 규칙: high+rise, 혼방(material-blend) 등"""

    def __init__(self, materials: Set[str]):
        self._materials = set(materials)

    def __call__(self, tokens: Set[str]) -> Set[str]:
        toks = set(t.lower() for t in tokens if t)
        # high + rise → high-rise
        if "high" in toks and "rise" in toks:
            toks.add("high-rise")
        # 혼방: material + 'blend' → '{material}-blend'
        if "blend" in toks:
            for m in self._materials:
                if m != "blend" and m in toks:
                    toks.add(f"{m}-blend")
        return toks


# --------------------------
# 최종 파이프라인
# --------------------------
class QueryPipeline:
    """
    normalize → stopwords → phrases → tokenize → rules → slots.route → embed
    """

    def __init__(
        self,
        *,
        synonyms: Dict[str, str],
        stopwords: Set[str],
        garments_phrases: Set[str],
        materials: Set[str],
        slots_def: Slots,
        max_tokens_hint: int = 64,
    ):
        # 단계별 모듈
        self._normalizer = Normalizer(synonyms)
        self._stopwords = StopwordFilter(stopwords)
        self._phrases = PhraseExtractor(garments_phrases)
        self._tokenizer = Tokenizer()
        self._rules = RulesEngine(materials)

        # 슬롯 정의(룰/별칭/허용어)는 공유, 값은 매 요청마다 복제(empty_like)
        self._slots_def = slots_def
        self._max_tokens_hint = max_tokens_hint

    # YAML에서 한 번에 빌드하는 빌더 메서드
    @classmethod
    def from_yaml(
        cls, path: str | None = None, *, max_tokens_hint: int = 64
    ) -> "QueryPipeline":
        if path is None:
            slots, pp = load_default_all()
        else:
            slots, pp = load_all(path)
        return cls(
            synonyms=pp.synonyms,
            stopwords=pp.stopwords,
            garments_phrases=pp.garments_phrases,
            materials=pp.materials,
            slots_def=slots,
            max_tokens_hint=max_tokens_hint,
        )

    def _route_to_slots(self, tokens: Set[str], phrase_garments: Set[str]) -> Slots:
        out = self._slots_def.empty_like()
        # phrase(문구)는 먼저 넣고, 이후 토큰 라우팅
        out.garment.update(phrase_garments)

        out.color.update(tokens)
        out.material.update(tokens)
        out.garment.update(tokens)
        out.fit.update(tokens)
        out.length.update(tokens)
        out.season.update(tokens)
        out.style.update(tokens)
        return out

    def process(
        self, text: str, *, include_tags_for_embed: bool = False
    ) -> Dict[str, object]:
        # 1) normalize
        q_norm = self._normalizer(text)
        # 2) stopwords
        q_norm = self._stopwords(q_norm)
        # 3) phrases
        phrase_garments = self._phrases(q_norm)
        # 4) tokenize
        toks = self._tokenizer(q_norm)
        # 5) rules
        toks = self._rules(toks)
        # 6) route → slots
        slots = self._route_to_slots(toks, phrase_garments)

        # 7) embed text
        if include_tags_for_embed:
            tags = " ".join(
                f"[{k}:{','.join(getattr(slots, k).values)}]"
                for k in slots.__dict__
                if getattr(slots, k).values
            )
            query_for_embed = (q_norm + " " + tags).strip()
        else:
            query_for_embed = q_norm

        if len(query_for_embed.split()) > self._max_tokens_hint:
            query_for_embed = " ".join(query_for_embed.split()[: self._max_tokens_hint])

        return {
            "normalized_query": q_norm,
            "slots": {k: getattr(slots, k).values for k in slots.__dict__},
            "query_for_embed": query_for_embed,
        }
