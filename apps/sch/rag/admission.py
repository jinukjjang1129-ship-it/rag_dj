"""rag/admission.py — 입시정보 도메인. 2027 수시모집 요강 PDF 기반."""

from __future__ import annotations

import re

from common import config
from ingest.ingest_admission import ADMISSION_SEARCH_TERMS, ADMISSION_SECTION_RANGES

from . import engine

DOMAIN = "admission"

ADMISSION_KEYWORDS = (
    "전형",
    "모집단위",
    "모집인원",
    "원서접수",
    "합격자",
    "등록금",
    "제출서류",
    "학교생활기록부",
    "면접",
    "실기",
    "장학금",
    "전형료",
    "지역인재",
    "정시",
    "수시",
)


def _normalize(text: str) -> str:
    return re.sub(r"[\s·ㆍ]", "", (text or "").lower())


# ── 1단계: 특정 전형 하나를 콕 집는 별칭 ──
# (별칭, 실제 track 메타데이터 안에서 찾을 부분 문자열) 쌍.
# 부분 문자열로 매칭하는 이유는, 실제 저장된 track 값이 "학생부교과(지역
# 의사선발전형-천안권, 공주권, 서산권, 논산권, 홍성권, 광역권)"처럼 문서
# 원문 그대로라 완전히 똑같은 문자열을 미리 알 수 없기 때문이다.
_TRACK_NICKNAME_TO_SUBSTRING = [
    ("교과우수자전형", "교과우수자전형"),
    ("교과우수전형", "교과우수자전형"),
    ("충남형지역인재전형", "충남형지역인재전형"),
    ("충남지역인재전형", "충남형지역인재전형"),
    ("충청형지역인재전형", "충청형지역인재전형"),
    ("충청지역인재전형", "충청형지역인재전형"),
    ("지역의사선발전형", "지역의사선발전형"),
    ("지역의사전형", "지역의사선발전형"),
    ("기회균형전형", "기회균형전형"),
    ("일반학생전형", "일반학생전형"),
    ("SW융합전형", "SW융합전형"),
    ("SW전형", "SW융합전형"),
    ("소프트웨어융합전형", "SW융합전형"),
    ("사회통합전형", "사회통합전형"),
    ("다문화전형", "사회통합전형"),
    ("다자녀전형", "사회통합전형"),
    ("실기우수자전형", "실기우수자전형"),
    ("특기자전형", "특기자전형"),
    ("체육특기생", "특기자전형"),
    ("기초생활수급자및차상위계층전형", "기초생활수급자및차상위계층전형"),
    ("기초수급자전형", "기초생활수급자및차상위계층전형"),
    ("차상위전형", "기초생활수급자및차상위계층전형"),
    ("농어촌학생전형", "농어촌학생전형"),
    ("농어촌전형", "농어촌학생전형"),
    ("특성화고교졸업자전형", "특성화고교졸업자전형"),
    ("특성화고전형", "특성화고교졸업자전형"),
    ("조기취업형계약학과전형", "조기취업형계약학과전형"),
    ("조기취업형전형", "조기취업형계약학과전형"),
    ("계약학과전형", "조기취업형계약학과전형"),
    ("특성화고졸재직자전형", "특성화고졸재직자전형"),
    ("재직자전형", "특성화고졸재직자전형"),
    ("평생학습자전형", "평생학습자전형"),
    ("만학도전형", "평생학습자전형"),
    ("성인학습자전형", "평생학습자전형"),
]
_TRACK_NICKNAME_LOOKUP = sorted(
    ((_normalize(nick), substring) for nick, substring in _TRACK_NICKNAME_TO_SUBSTRING),
    key=lambda x: len(x[0]),
    reverse=True,
)

# ── 2단계: 큰 갈래만 말하는 별칭 ──
_TRACK_CATEGORY_NICKNAMES = [
    ("학생부교과전형", "학생부교과"),
    ("교과전형", "학생부교과"),
    ("학생부종합전형", "학생부종합"),
    ("학종", "학생부종합"),
    ("실기실적전형", "실기/실적"),
    ("실기전형", "실기/실적"),
]
_TRACK_CATEGORY_LOOKUP = sorted(
    ((_normalize(nick), category) for nick, category in _TRACK_CATEGORY_NICKNAMES),
    key=lambda x: len(x[0]),
    reverse=True,
)

# ── 3단계: 기존 섹션별 연관검색어 재활용 ──
_SECTION_TERM_LOOKUP = sorted(
    (
        (_normalize(word), section)
        for section, terms in ADMISSION_SEARCH_TERMS.items()
        for word in terms.split()
    ),
    key=lambda x: len(x[0]),
    reverse=True,
)


def _load_tracks() -> list[str]:
    """적재된 청크들의 track 메타데이터 값(중복 제거)을 모은다."""
    vectordb = engine.get_vectordb(DOMAIN)
    try:
        got = vectordb._collection.get(include=["metadatas"])
        metadatas = got.get("metadatas", []) or []
    except Exception:
        return []
    tracks = {(m or {}).get("track", "").strip() for m in metadatas}
    tracks.discard("")
    return list(tracks)


def detect_track_filter(question: str) -> tuple[dict | None, str]:
    """1단계(특정 전형) → 2단계(전형 갈래) 순서로 검사해 필터를 반환.

    반환: (Chroma where 필터 또는 None, 사람이 읽을 설명 문자열)
    """
    normalized_q = _normalize(question)
    if not normalized_q:
        return None, ""

    for nick_norm, substring in _TRACK_NICKNAME_LOOKUP:
        if nick_norm and nick_norm in normalized_q:
            matching_tracks = [t for t in _load_tracks() if substring in t]
            if matching_tracks:
                return {"track": {"$in": matching_tracks}}, f"전형 '{substring}'"

    for nick_norm, category in _TRACK_CATEGORY_LOOKUP:
        if nick_norm and nick_norm in normalized_q:
            return {"track_category": category}, f"'{category}' 계열 전형"

    return None, ""


def detect_matching_sections(question: str) -> list[str]:
    """질문에 걸리는 연관검색어가 있는 섹션을 전부 모아서 반환."""
    normalized_q = _normalize(question)
    if not normalized_q:
        return []
    matched = []
    for word_norm, section in _SECTION_TERM_LOOKUP:
        if word_norm and word_norm in normalized_q and section not in matched:
            matched.append(section)
    return matched


def answer(
    question: str,
    top_k: int = config.DOMAINS["admission"].get("top_k", config.DEFAULT_TOP_K),
) -> dict:
    if "전형료" in question:
        # Fees live in a common table, not in a track's eligibility pages.
        result = engine.answer_question(
            DOMAIN,
            question,
            top_k,
            where={"section": "전형료"},
            scope_note="\n전형료 공통 표에서 질문에 해당하는 전형의 금액을 확인하세요.",
            extra_scope_keywords=ADMISSION_KEYWORDS,
        )
        result["detected_track_filter"] = None
        result["detected_sections"] = ["전형료"]
        return result
    where, track_desc = detect_track_filter(question)
    sections: list[str] = []

    if where:
        scope_note = f"\n이 질문은 {track_desc}에 대한 질문입니다."
    else:
        sections = detect_matching_sections(question)
        where = {"section": {"$in": sections}} if sections else None
        scope_note = (
            f"\n이 질문은 {', '.join(repr(s) for s in sections)} 섹션과 관련이 있습니다."
            if sections
            else ""
        )

    result = engine.answer_question(
        DOMAIN,
        question,
        top_k,
        where=where,
        scope_note=scope_note,
        extra_scope_keywords=ADMISSION_KEYWORDS,
    )
    result["detected_track_filter"] = track_desc or None
    result["detected_sections"] = sections
    return result


def get_status() -> dict:
    """목차 섹션 전체 대비 실제로 적재된 섹션이 몇 개인지 집계."""
    all_sections = [name for _, _, name in ADMISSION_SECTION_RANGES]
    vectordb = engine.get_vectordb(DOMAIN)
    try:
        got = vectordb._collection.get(include=["metadatas"])
        metadatas = got.get("metadatas", []) or []
    except Exception:
        metadatas = []

    indexed_sections: dict[str, int] = {}
    for m in metadatas:
        section = (m or {}).get("section", "").strip()
        if not section:
            continue
        indexed_sections[section] = indexed_sections.get(section, 0) + 1

    items = [
        {
            "section": name,
            "trained": name in indexed_sections,
            "chunk_count": indexed_sections.get(name, 0),
        }
        for name in all_sections
    ]
    return {
        "items": items,
        "trained_count": sum(1 for i in items if i["trained"]),
        "total_count": len(items),
        "total_chunks": sum(i["chunk_count"] for i in items),
    }
