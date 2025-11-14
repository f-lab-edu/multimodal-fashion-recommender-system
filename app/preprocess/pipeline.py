from __future__ import annotations
from typing import Dict, Set
from app.preprocess.slots.schema import Slots
from app.preprocess.slots.registry import load_all

from app.preprocess.stopword_filter import StopwordFilter
from app.preprocess.normalizer import Normalizer
from app.preprocess.phrase_extractor import PhraseExtractor
from app.preprocess.tokenizer import Tokenizer
from app.preprocess.rules import RulesEngine


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
