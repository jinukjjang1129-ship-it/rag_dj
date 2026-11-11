"""common/guard.py — 이게 순천향대 관련 질문인가?를 LLM이 아니라 코드로 판단한다."""

from __future__ import annotations

import re
from typing import Iterable

from . import config


def _normalize(text: str) -> str:
    return re.sub(r"[\s·ㆍ]", "", (text or "").lower())


def has_university_keyword(question: str, extra_keywords: Iterable[str] = ()) -> bool:
    """질문에 순천향대/도메인 관련 키워드가 하나라도 있는가."""
    normalized_q = _normalize(question)
    if not normalized_q:
        return False
    all_keywords = list(config.UNIVERSITY_KEYWORDS) + list(extra_keywords)
    return any(_normalize(kw) in normalized_q for kw in all_keywords)


def looks_out_of_scope(question: str, extra_keywords: Iterable[str] = ()) -> bool:
    """검색을 돌리기도 전에, 명백히 범위 밖으로 보이는 질문인지 1차 판별.

    False가 나와도 확정은 아니다 (키워드가 없어도 관련 질문일 수 있음).
    True일 때만 "명백히 범위 밖"으로 취급해 검색을 생략하는 용도로 쓴다.
    아주 짧은 인사말/잡담 패턴 + 키워드 완전 부재를 함께 볼 때만 True.
    """
    normalized_q = _normalize(question)
    if not normalized_q:
        return True

    # "/" 같은 특수문자 한두 글자, 자음/모음 낱글자(ㅋㅋ, ㅇㅇ) 등 의미 있는
    # 한글·영문·숫자가 전혀 없는 입력은 검색을 돌려봐야 어차피 의미 없는
    # 결과만 나온다. 이런 입력이 우연히 관련도 게이트를 넘어서 엉뚱한 자료
    # 여러 개를 뒤섞어 답하는 사고를 막기 위해 여기서 먼저 차단한다.
    has_meaningful_char = bool(re.search(r"[가-힣a-z0-9]", normalized_q))
    if not has_meaningful_char:
        return True
    if len(normalized_q) < 2:
        return True

    chit_chat_patterns = (
        "안녕",
        "고마워",
        "감사",
        "날씨",
        "오늘기분",
        "누구야",
        "몇살",
        "이름이뭐야",
        "테스트",
        "ㅎㅇ",
        "심심",
    )
    is_chit_chat = any(p in normalized_q for p in chit_chat_patterns)
    if is_chit_chat and not has_university_keyword(question, extra_keywords):
        return True
    return False


def passes_relevance_gate(top_score: float) -> bool:
    """검색된 최상위 결과의 관련도(0~100)가 답변에 쓸 만큼 충분한가."""
    return top_score >= config.SCOPE_RELEVANCE_THRESHOLD


OUT_OF_SCOPE_MESSAGE = (
    "죄송하지만 해당 질문은 제가 다루는 순천향대학교 자료 범위에서 "
    "확인되지 않습니다. 학과/입시/논문 중 관련된 내용으로 다시 질문해 주세요."
)
