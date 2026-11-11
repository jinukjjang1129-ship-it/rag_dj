"""rag/paper.py — 논문정보 도메인. PDF 논문(집) 기반 검색."""

from __future__ import annotations

import re
from pathlib import Path

from common import config

from . import engine

DOMAIN = "paper"

PAPER_KEYWORDS = (
    "논문",
    "저자",
    "초록",
    "학술제",
    "연구",
    "게재",
    "발표",
    "인용",
    "키워드",
)

NO_MATCHING_PAPER_MESSAGE = (
    "죄송하지만 지금 적재된 논문 중에는 이 질문과 관련된 논문이 없는 것 "
    "같습니다. 논문 목록에 있는 주제로 다시 질문해 주세요."
)


def _normalize(text: str) -> str:
    return re.sub(r"[\s;,·ㆍ]+", "", (text or "").lower())


def _load_paper_keyword_map() -> dict[str, dict]:
    """적재된 모든 논문의 (파일명 -> {keywords, title})를 모은다.

    청크마다 같은 논문이면 같은 메타데이터가 반복 저장돼 있어서,
    파일명(source) 기준으로 한 번씩만 모으면 된다.
    """
    vectordb = engine.get_vectordb(DOMAIN)
    try:
        got = vectordb._collection.get(include=["metadatas"])
        metadatas = got.get("metadatas", []) or []
    except Exception:
        return {}

    info_map: dict[str, dict] = {}
    for m in metadatas:
        source = (m or {}).get("source", "").strip()
        if not source or source in info_map:
            continue
        raw_keywords = (m or {}).get("paper_keywords", "").strip()
        info_map[source] = {
            "keywords": [k.strip() for k in raw_keywords.split(";") if k.strip()],
            "title": (m or {}).get("paper_title", "").strip(),
        }
    return info_map


def detect_matching_papers(question: str, info_map: dict[str, dict]) -> list[str]:
    """질문과 키워드/제목이 겹치는 논문의 파일명 목록을 반환 (없으면 빈 리스트).

    키워드만 보면 "웹 취약점 점검"을 물었는데 논문 키워드가 영어
    ("Vulnerability")로만 뽑혀서 못 찾는 경우가 실제로 있었다(5번 논문).
    그래서 키워드뿐 아니라 논문 제목(항상 한국어)도 같이 매칭 재료로
    쓴다 — "웹 취약점 점검 방법에 관한 연구"라는 제목 안에 "웹", "취약점"이
    이미 들어있어서, 키워드가 영어여도 제목에서 걸려준다.

    키워드/제목 구를 통째로 검사하는 것과, 단어 단위로 쪼개서 검사하는
    것을 같이 한다. 완전히 똑같은 표현을 써야만 걸리면 너무 빡빡해서
    실사용에서 자꾸 놓치게 되기 때문이다.
    """
    normalized_q = _normalize(question)
    if not normalized_q:
        return []

    matches = []
    for source, info in info_map.items():
        phrases = list(info.get("keywords", []))
        # Curated filenames retain the topic when PDF title extraction fails.
        phrases.append(Path(source).stem)
        if info.get("title"):
            phrases.append(info["title"])

        for phrase in phrases:
            phrase_norm = _normalize(phrase)
            generic = {
                "연구",
                "관한",
                "활용한",
                "기반",
                "이용한",
                "방법",
                "논문",
                "학술",
            }
            words = [
                w
                for w in re.split(r"\s+", phrase)
                if len(w) >= 2 and w not in generic and not re.fullmatch(r"\d+\.?", w)
            ]
            hit = bool(phrase_norm) and phrase_norm in normalized_q
            hit = hit or any(_normalize(w) in normalized_q for w in words)
            if hit:
                matches.append(source)
                break
    return matches


def answer(question: str, top_k: int = config.DEFAULT_TOP_K) -> dict:
    info_map = _load_paper_keyword_map()

    # 논문이 아예 없거나(적재 전) 전부 키워드 추출에 실패한 경우엔 키워드
    # 매칭 자체가 무의미하므로, 이 경우에만 기존처럼 관련도 점수에 맡긴다.
    has_any_signal = any(
        info.get("keywords") or info.get("title") for info in info_map.values()
    )

    matched_sources: list[str] = []
    if has_any_signal:
        matched_sources = detect_matching_papers(question, info_map)
        if not matched_sources:
            return {
                "question": question,
                "answer": NO_MATCHING_PAPER_MESSAGE,
                "sources": [],
                "used_fallback": False,
                "in_scope": False,
                "matched_papers": [],
                "timing": {"search_sec": 0.0, "generate_sec": 0.0, "total_sec": 0.0},
            }

    where = {"source": {"$in": matched_sources}} if matched_sources else None

    result = engine.answer_question(
        DOMAIN,
        question,
        top_k,
        where=where,
        scope_note="",
        extra_scope_keywords=PAPER_KEYWORDS,
    )
    result["matched_papers"] = matched_sources
    return result


def get_status() -> dict:
    """적재된 논문 파일(source) 목록과 파일별 청크 수, 키워드 추출 여부."""
    vectordb = engine.get_vectordb(DOMAIN)
    try:
        got = vectordb._collection.get(include=["metadatas"])
        metadatas = got.get("metadatas", []) or []
    except Exception:
        metadatas = []

    files: dict[str, dict] = {}
    for m in metadatas:
        source = (m or {}).get("source", "").strip()
        if not source:
            continue
        info = files.setdefault(
            source,
            {
                "source": source,
                "trained": True,
                "chunk_count": 0,
                "title": (m or {}).get("paper_title", ""),
                "keywords": (m or {}).get("paper_keywords", ""),
            },
        )
        info["chunk_count"] += 1

    items = sorted(files.values(), key=lambda x: -x["chunk_count"])
    return {
        "items": items,
        "trained_count": len(items),
        "total_count": len(items),
        "total_chunks": sum(i["chunk_count"] for i in items),
    }
