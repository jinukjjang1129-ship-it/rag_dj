"""rag/department.py — 학과정보 도메인. 질문에서 학과를 감지해 그 학과 안에서만 검색."""

from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple

from common import config

from . import engine

DOMAIN = "department"


def _normalize(text: str) -> str:
    return re.sub(r"[\s·ㆍ]", "", (text or "").lower())


def detect_department_from_question(question: str) -> Optional[str]:
    """긴 별칭부터 검사해 가장 구체적인 학과를 찾는다."""
    normalized_q = _normalize(question)
    if not normalized_q:
        return None

    candidates: List[Tuple[str, str]] = []
    for dept, aliases in config.DEPARTMENT_ALIASES.items():
        for alias in aliases:
            candidates.append((_normalize(alias), dept))
    candidates.sort(key=lambda x: len(x[0]), reverse=True)

    for alias_norm, dept in candidates:
        if alias_norm and alias_norm in normalized_q:
            return dept
    return None


def _boost_department_head(results: list[dict]) -> list[dict]:
    """검색 결과 중 학과장 교수 청크가 있으면 맨 앞으로 옮긴다.

    실제 데이터(data/chunks.json)를 보면 학과장인 교수의 청크만 본문 끝에
    "학과장"이라는 표시가 붙어있다 (예: "...이메일: sdkim.mie@sch.ac.kr\n
    의료IT공학과 학과장"). 이 표시 유무로 학과장 청크를 가려낸다.

    "OO학과 교수님 알려줘" 같은 질문에서 학과장이 검색 결과 2~3번째로
    밀려나는 경우가 있어서(벡터 유사도 순서라 보장이 안 됨), 사람이 자연스럽게
    기대하는 순서(학과장부터)로 앞당긴다. 학과장 청크가 없는 검색 결과(교육과정,
    진로정보 등)에는 아무 영향이 없다 — "학과장"이라는 문자열 자체가 그런
    청크 본문엔 안 나오기 때문이다.
    """
    head = [r for r in results if "학과장" in (r.get("text") or "")]
    rest = [r for r in results if "학과장" not in (r.get("text") or "")]
    return head + rest


# 학과가 특정된 검색은 이 값을 최소 기준으로 씀 (기본 top_k=5보다 교수
# 인원이 많은 학과가 있어서 — 예: 컴퓨터소프트웨어공학과 교수 6명).
# 학과 필터로 이미 검색 범위가 좁혀진 상태라, 넉넉하게 가져와도 노이즈
# 걱정 없이 안전하다.
MIN_TOP_K_WHEN_DEPARTMENT_DETECTED = 10


def answer(question: str, top_k: int = config.DEFAULT_TOP_K) -> dict:
    started = time.perf_counter()
    department = detect_department_from_question(question)
    if (
        department
        and any(word in question for word in ("교수", "교원"))
        and any(
            word in question
            for word in ("누가", "누구", "명단", "목록", "모두", "전체")
        )
        and "학과장" not in question
    ):
        # Enumeration comes from metadata, not an arbitrary vector top-k or LLM.
        where = {"$and": [{"department": department}, {"content_type": "professor"}]}
        got = engine.get_vectordb(DOMAIN).get(
            where=where, include=["metadatas", "documents"]
        )
        items = {}
        for meta, document in zip(got.get("metadatas", []), got.get("documents", [])):
            meta = meta or {}
            key = meta.get("title", "").strip()
            if re.fullmatch(r".{2,30}\s교수(?:님)?", key):
                items.setdefault(
                    key,
                    {
                        "title": key,
                        "url": meta.get("url", ""),
                        "text": meta.get("raw_text", document),
                        "metadata": meta,
                    },
                )
        sources = [
            {
                **item,
                "rank": i + 1,
                "score": None,
                "score_type": "exact_metadata_filter",
            }
            for i, item in enumerate(sorted(items.values(), key=lambda x: x["title"]))
        ]
        answer_text = (
            (
                f"{department}의 적재 자료에서 확인한 교수 목록입니다.\n"
                + "\n".join(f"- {r['title']}" for r in sources)
            )
            if sources
            else "해당 학과의 교수 자료가 적재되어 있지 않습니다."
        )
        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "used_fallback": False,
            "in_scope": bool(sources),
            "detected_department": department,
            "route": "metadata_list",
            "generation_mode": "deterministic_metadata",
            "timing": {
                "search_sec": time.perf_counter() - started,
                "generate_sec": 0.0,
                "total_sec": time.perf_counter() - started,
            },
        }
    where = {"department": department} if department else None
    scope_note = f"\n이 질문은 '{department}'에 대한 질문입니다." if department else ""

    # 학과가 특정됐는데 top_k가 그보다 작으면 올려준다. 학과 필터 없이
    # 전체 검색할 때는 그대로 둔다 (여러 학과가 섞여 있어서 넉넉하게
    # 가져오면 오히려 관련 없는 결과가 늘어날 수 있음).
    effective_top_k = (
        max(top_k, MIN_TOP_K_WHEN_DEPARTMENT_DETECTED) if department else top_k
    )

    result = engine.answer_question(
        DOMAIN,
        question,
        effective_top_k,
        where=where,
        scope_note=scope_note,
        extra_scope_keywords=tuple(
            a for aliases in config.DEPARTMENT_ALIASES.values() for a in aliases
        ),
        rerank_fn=_boost_department_head,
    )
    result["detected_department"] = department
    return result


def get_status() -> dict:
    """학과별 적재 현황 (적재/미적재, 청크 수, 정보 종류)."""
    vectordb = engine.get_vectordb(DOMAIN)
    try:
        got = vectordb._collection.get(include=["metadatas"])
        metadatas = got.get("metadatas", []) or []
    except Exception:
        metadatas = []

    depts: dict[str, dict] = {}
    for m in metadatas:
        dept = (m or {}).get("department", "").strip()
        if not dept:
            continue
        info = depts.setdefault(
            dept,
            {
                "department": dept,
                "trained": True,
                "chunk_count": 0,
                "has_professor": False,
                "has_curriculum": False,
                "has_career": False,
            },
        )
        info["chunk_count"] += 1
        if (m or {}).get("content_type") == "professor":
            info["has_professor"] = True
        if (m or {}).get("section") == "교육과정":
            info["has_curriculum"] = True
        if (m or {}).get("content_type") == "career":
            info["has_career"] = True

    for item in config.UNTRAINED_DEPARTMENTS:
        depts.setdefault(
            item["department"],
            {
                "department": item["department"],
                "trained": False,
                "chunk_count": 0,
                "has_professor": False,
                "has_curriculum": False,
                "has_career": False,
            },
        )

    items = sorted(depts.values(), key=lambda x: (-x["chunk_count"], x["department"]))
    return {
        "items": items,
        "trained_count": sum(1 for i in items if i["trained"]),
        "total_count": len(items),
        "total_chunks": sum(i["chunk_count"] for i in items),
    }
