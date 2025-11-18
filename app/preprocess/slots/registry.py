# slots/registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Tuple
from .schema import Slot, Slots
import os

# 외부 의존성: PyYAML
try:
    import yaml
except ImportError:
    yaml = None


# --------------------------
# 전처리 설정 번들
# --------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    synonyms: Dict[str, str]  # 한→영 동의어 (Normalizer에서 사용)
    stopwords: Set[str]  # 불용어
    garments_phrases: Set[str]  # 문구 기반 아이템
    materials: Set[str]  # 소재(혼방 룰 등 규칙 엔진에서 사용)


# --------------------------
# 내부 유틸
# --------------------------
def _build_alias_from_syn(syn: Dict[str, str], allowed: Set[str]) -> Dict[str, str]:
    """
    SYN(KO->EN) 중 EN이 해당 슬롯의 allowed에 존재할 때만 별칭으로 채택.
    예: '연청' -> 'light-blue' (colors에 light-blue가 있을 때만)
    """
    out: Dict[str, str] = {}
    for ko, en in syn.items():
        std = str(en).lower()
        if std in allowed:
            out[ko.lower()] = std
    return out


# --------------------------
# 공개 API
# --------------------------
def load_from_yaml(path: str) -> Slots:
    """
    (슬롯 전용) YAML을 읽어 Slots만 구성.
    - aliases: 수동 별칭
    - synonyms: KO->EN은 슬롯 alias로 '필요한 것만' 흡수
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    colors = set(cfg["slots"]["colors"])
    materials = set(cfg["slots"]["materials"])
    garments_w = set(cfg["slots"]["garments_words"])
    garments_p = set(cfg["slots"]["garments_phrases"])
    fits = set(cfg["slots"]["fits"])
    lengths = set(cfg["slots"]["lengths"])
    seasons = set(cfg["slots"]["seasons"])
    styles = set(cfg["slots"]["styles"])

    aliases_cfg = cfg.get("aliases", {})
    syn = cfg.get("synonyms", {})

    color = Slot(
        "color",
        allowed=colors,
        aliases={
            "grey": "gray",
            "dk-blue": "dark-blue",
            "lt-blue": "light-blue",
            **_build_alias_from_syn(syn, colors),
            **aliases_cfg.get("color", {}),
        },
    )
    material = Slot(
        "material",
        allowed=materials,
        aliases={
            **_build_alias_from_syn(syn, materials),
            **aliases_cfg.get("material", {}),
        },
    )
    garment = Slot(
        "garment",
        allowed=garments_w | garments_p,
        aliases={
            "남방": "shirt",
            **_build_alias_from_syn(syn, garments_w | garments_p),
            **aliases_cfg.get("garment", {}),
        },
    )
    fit = Slot("fit", allowed=fits, aliases=aliases_cfg.get("fit", {}))
    length = Slot("length", allowed=lengths, aliases=aliases_cfg.get("length", {}))
    season = Slot(
        "season",
        allowed=seasons,
        aliases={
            **_build_alias_from_syn(syn, seasons),
            **aliases_cfg.get("season", {}),
        },
    )
    style = Slot(
        "style",
        allowed=styles,
        aliases={**_build_alias_from_syn(syn, styles), **aliases_cfg.get("style", {})},
    )

    return Slots(color, material, garment, fit, length, season, style)


def load_all(path: str) -> Tuple[Slots, PreprocessConfig]:
    """
    (권장) 슬롯 + 전처리 설정을 한 번에 로드.
    - slots.*: 허용어/별칭
    - synonyms/stopwords/garments_phrases/materials: 전처리 파이프라인 입력
    """
    if path is None:
        config_path = os.path.join(os.path.dirname(__file__), "defaults.yaml")
    else:
        # 2) path가 디렉터리면 defaults.yaml 붙이고, 파일이면 그대로 사용
        config_path = (
            os.path.join(path, "defaults.yaml") if os.path.isdir(path) else path
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    slots = load_from_yaml(config_path)

    syn = {str(k): str(v) for k, v in cfg.get("synonyms", {}).items()}
    stopwords = set(cfg.get("stopwords", []))
    phrases = set(cfg["slots"]["garments_phrases"])
    materials = set(cfg["slots"]["materials"])

    ppconf = PreprocessConfig(
        synonyms=syn, stopwords=stopwords, garments_phrases=phrases, materials=materials
    )
    return slots, ppconf
