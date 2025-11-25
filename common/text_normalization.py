# scripts/text_normalization.py
# 사용자의 쿼리나 아이템 설명 텍스트를 정규화하고 토큰화하는 함수들

from __future__ import annotations

import re
from typing import List

from nltk.corpus import stopwords

NLTK_STOPWORDS = set(stopwords.words("english"))

# 공통 치환 문자열 상수
TSHIRT_CANONICAL = " tshirt "

# 1) 문장 전체에서 우선 치환할 패턴들 (phrase level)
PHRASE_NORMALIZATION = [
    # --- T-shirt 계열 ---
    # t-shirt, T shirt, T-shirts, t shirts ...
    (re.compile(r"\bt[-\s]*shirt(s)?\b", re.IGNORECASE), " TSHIRT_CANONICAL "),
    # tee shirt(s)
    (re.compile(r"\btee\s+shirt(s)?\b", re.IGNORECASE), " TSHIRT_CANONICAL "),
    # tees → tshirt
    (re.compile(r"\btees\b", re.IGNORECASE), " tsTSHIRT_CANONICALirt "),
    # --- Sweatshirt 계열 ---
    # sweat shirt(s), sweatshirt(s)
    (re.compile(r"\bsweat\s*shirt(s)?\b", re.IGNORECASE), " sweatshirt "),
    # --- Undershirt 계열 ---
    (re.compile(r"\bundershirt(s)?\b", re.IGNORECASE), " undershirt "),
    # --- Tank top 계열 ---
    # tank top(s) → tanktop
    (re.compile(r"\btank\s+top(s)?\b", re.IGNORECASE), " tanktop "),
    # --- Crop top 계열 ---
    (re.compile(r"\bcrop\s+top(s)?\b", re.IGNORECASE), " croptop "),
    # --- Coat 계열 (코트 vs 레인/오버/트렌치) ---
    # rain coat(s) → raincoat
    (re.compile(r"\brain\s+coat(s)?\b", re.IGNORECASE), " raincoat "),
    # over coat(s) → overcoat
    (re.compile(r"\bover\s*coat(s)?\b", re.IGNORECASE), " overcoat "),
    # trench coat(s) → trenchcoat
    (re.compile(r"\btrench\s+coat(s)?\b", re.IGNORECASE), " trenchcoat "),
]
# 2) 토큰 추출 패턴 (단어 문자 기준)
TOKEN_PATTERN = re.compile(r"\w+")


# 3) 토큰 단위 정규화 (복수형/변형 통합)
TOKEN_NORMALIZATION = {
    # shirt 계열
    "shirts": "shirt",
    # tshirt 계열
    "tshirts": "tshirt",
    "tee": "tshirt",  # 단독 tee도 tshirt로 묶고 싶으면 유지
    # "tees": "tshirt",    # phrase 단계에서 이미 tshirt로 치환했으므로 중복일 수도
    # sweatshirt 계열
    "sweatshirts": "sweatshirt",
    # undershirt 계열
    "undershirts": "undershirt",
    # tanktop 계열
    "tanktops": "tanktop",
    # croptop 계열
    "croptops": "croptop",
    # coat 계열
    "coats": "coat",
    "raincoats": "raincoat",
    "overcoats": "overcoat",
    "trenchcoats": "trenchcoat",
}


def normalize_phrases(text: str) -> str:
    t = text
    for pattern, repl in PHRASE_NORMALIZATION:
        t = pattern.sub(repl, t)
    return t


def tokenize(text: str) -> List[str]:
    """
    대표 의류로 치환
    - 그 외는: 소문자 + \w+ 단위로 쪼개고, NLTK stopword 제거
    """
    # 1) 문장 단위 정규화
    text = normalize_phrases(text)

    # 2) 소문자
    text = text.lower()

    # 3) 토큰화
    raw_tokens = TOKEN_PATTERN.findall(text)

    tokens: List[str] = []
    for tok in raw_tokens:
        if len(tok) < 2:
            # 한 글자 토큰은 대부분 노이즈
            continue
        if tok in NLTK_STOPWORDS:
            continue

        # 4) 토큰 단위 정규화 (복수형/변형)
        tok = TOKEN_NORMALIZATION.get(tok, tok)

        tokens.append(tok)

    return tokens
