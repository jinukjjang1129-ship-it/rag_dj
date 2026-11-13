"""Company question routing with explicit scopes and evidence provenance."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config import (
    COL_DOCS,
    LOCAL_LLM_MODEL,
    PERSIST_DOCS,
    embedding_signature,
    get_embeddings,
    get_llm,
)
from document_rag.retrieval import answer as grounded_answer
from document_rag.retrieval import open_index

BUSINESS = [
    "반도체 (사업분야)",
    "디스플레이 (사업분야)",
    "신재생에너지 (사업분야)",
    "발포제 (사업분야)",
]
HINTS = {
    "반도체": "반도체 (사업분야)",
    "디스플레이": "디스플레이 (사업분야)",
    "신재생": "신재생에너지 (사업분야)",
    "이차전지": "신재생에너지 (사업분야)",
    "연료전지": "신재생에너지 (사업분야)",
    "발포제": "발포제 (사업분야)",
    "회사개요": "회사개요",
    "소개": "회사개요",
    "CEO": "CEO인사말 (회사소개)",
    "인사말": "CEO인사말 (회사소개)",
    "사업장": "사업장 소개 (회사소개)",
    "찾아오시는": "찾아오시는길 (회사소개)",
    "위치": "찾아오시는길 (회사소개)",
    "윤리경영": "윤리경영 (지속가능경영)",
    "혁신경영": "혁신경영 (지속가능경영)",
    "환경안전": "환경안전 (지속가능경영)",
    "ESG": "ESG 정보공개 (지속가능경영)",
    "IR공고": "IR공고 (IR/PR)",
    "공시": "공시정보 (IR/PR)",
    "공지사항": "공지사항 (IR/PR)",
    "회사소식": "회사소식 (IR/PR)",
    "홍보영상": "홍보영상 (IR/PR)",
    "Q&A": "Q&A (IR/PR)",
    "채용": "동진HR (채용정보)",
    "인재상": "동진HR (채용정보)",
}
BUSINESS_GROUP_HINTS = [
    "사업분야",
    "사업 분야",
    "무슨 사업",
    "어떤 사업",
    "사업영역",
    "사업 영역",
]


@lru_cache(maxsize=1)
def load_vectordb():
    if not Path(PERSIST_DOCS).exists():
        raise FileNotFoundError("Company index is missing; run python ingest.py")
    return open_index(
        Path(PERSIST_DOCS),
        COL_DOCS,
        model=embedding_signature(),
        embedding_function=get_embeddings(),
    )


def answer(query: str, k: int = 4, generate: bool = True) -> dict:
    scope_terms = tuple(HINTS) + (
        "동진",
        "회사",
        "기업",
        "제품",
        "사업",
        "경영",
        "설립",
        "연혁",
        "대표",
        "연구개발",
        "문의",
        "재무",
        "매출",
        "투자",
        "품질",
        "복리후생",
    )
    if not any(term.casefold() in query.casefold() for term in scope_terms):
        return {
            "answer": "기업정보 자료 범위에서 확인할 질문을 입력해 주세요.",
            "sources": [],
            "source_details": [],
            "contexts": [],
            "generation_mode": "abstain",
        }
    database = load_vectordb()
    broad = any(hint in query for hint in BUSINESS_GROUP_HINTS) and not any(
        kw in query
        for kw in ("반도체", "디스플레이", "신재생", "이차전지", "연료전지", "발포제")
    )
    if broad and any(word in query for word in ("전체", "목록", "종류", "어떤")):
        got = database.get(where={"category": {"$in": BUSINESS}}, include=["metadatas"])
        sources = {(meta["category"], meta["source"]) for meta in got["metadatas"]}
        return {
            "answer": "적재 자료에서 확인한 사업분야입니다.\n"
            + "\n".join(f"- {c}" for c in sorted({c for c, _ in sources})),
            "sources": [f"{c}: {u}" for c, u in sorted(sources)],
            "contexts": [],
            "generation_mode": "deterministic_metadata",
        }
    categories = [
        category
        for keyword, category in HINTS.items()
        if keyword.casefold() in query.casefold()
    ]
    where = {"category": {"$in": sorted(set(categories))}} if categories else None
    evidence = None
    if broad:
        query_embedding = database._embedding_function.embed_query(query)
        evidence = [
            (doc, None)
            for category in BUSINESS
            for doc in database.similarity_search_by_vector(
                query_embedding,
                k=max(1, k // len(BUSINESS)),
                filter={"category": category},
            )
        ]
    result = grounded_answer(
        database,
        query,
        where=where,
        generate=generate,
        k=k,
        evidence=evidence,
        llm_model=LOCAL_LLM_MODEL,
        llm=get_llm(temperature=0) if generate else None,
    )
    result["source_details"] = result["sources"]
    result["sources"] = [
        f"{s.get('category', '')}: {s['source']}" for s in result["source_details"]
    ]
    result["contexts"] = result["source_details"]
    return result
