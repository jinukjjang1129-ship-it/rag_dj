"""
rag.py — 검색 + 생성. 동진쎄미켐 페이지 요약/질의응답용.

저번(규정 조항 찾기)과 다른 점:
- 질문이 "사업분야 10줄 요약" 같은 '요약형'이 많다.
- 그래서 ① 카테고리 우선 검색 ② 요약 지시를 잘 따르는 프롬프트를 쓴다.
- 'N줄로' 같은 길이 지정도 프롬프트로 전달한다.

환각 억제:
- 페이지 본문에 있는 내용만 사용. 없으면 "찾을 수 없습니다".
"""
import os
import re
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (PERSIST_DOCS, COL_DOCS, get_embeddings, get_llm)

DEFAULT_K = 4

SYSTEM_PROMPT = """너는 동진쎄미켐 회사 정보를 안내하는 도우미다.

규칙:
- 아래 [참고 자료]에 있는 내용만으로 답한다. 자료 밖 지식·추측은 금지.
- 자료에 근거가 없으면: "제공된 자료에서 해당 내용을 찾을 수 없습니다." 라고만 답한다.
- 한국어 존댓말로, 정확하고 담백하게 정리한다.
- 사용자가 'N줄로' 또는 '몇 줄로' 요약을 요청하면 그 줄 수에 맞춰 답한다.
- 나열이 필요하면 간결한 문장으로 정리한다."""


def load_vectordb() -> Chroma:
    if not os.path.isdir(PERSIST_DOCS):
        raise FileNotFoundError(
            f"벡터DB가 없습니다: {PERSIST_DOCS}\n"
            f"먼저 'python ingest.py' 로 pages.json 을 적재하세요."
        )
    return Chroma(
        collection_name=COL_DOCS,
        persist_directory=PERSIST_DOCS,
        embedding_function=get_embeddings(),
    )


def detect_line_request(query: str) -> str:
    """'10줄로' 같은 길이 요청을 추출해 프롬프트에 강조로 넣기."""
    m = re.search(r"(\d+)\s*줄", query)
    if m:
        return f"\n\n[요청 형식] 반드시 약 {m.group(1)}줄로 요약해서 답하세요."
    return ""


def retrieve(vectordb: Chroma, query: str, k: int) -> List[Document]:
    """
    카테고리 힌트가 질문에 있으면 그 카테고리를 우선 검색.
    (예: '반도체' 포함 → 반도체 페이지 우선)
    """
    hints = {
        "반도체": "반도체 (사업분야)",
        "디스플레이": "디스플레이 (사업분야)",
        "신재생": "신재생에너지 (사업분야)",
        "이차전지": "신재생에너지 (사업분야)",
        "연료전지": "신재생에너지 (사업분야)",
        "배터리": "신재생에너지 (사업분야)",
        "발포제": "발포제 (사업분야)",
        "회사": "회사개요",
        "개요": "회사개요",
        "소개": "회사개요",
    }
    matched = None
    for kw, cat in hints.items():
        if kw in query:
            matched = cat
            break

    if matched:
        # 해당 카테고리로 필터 검색
        docs = vectordb.similarity_search(query, k=k, filter={"category": matched})
        if docs:
            return docs
    # 힌트 없거나 결과 없으면 전체 검색
    return vectordb.similarity_search(query, k=k)


def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        cat = d.metadata.get("category", "")
        head = f"[자료{i} | {cat}]"
        blocks.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def sources_of(docs: List[Document]) -> List[str]:
    seen = []
    for d in docs:
        cat = d.metadata.get("category", "?")
        src = d.metadata.get("source", "")
        label = f"{cat}  ({src})"
        if label not in seen:
            seen.append(label)
    return seen


def answer(query: str, k: int = DEFAULT_K) -> Dict[str, Any]:
    vectordb = load_vectordb()
    docs = retrieve(vectordb, query, k=k)

    if not docs:
        return {"answer": "제공된 자료에서 해당 내용을 찾을 수 없습니다.",
                "sources": [], "contexts": []}

    context = format_context(docs)
    line_req = detect_line_request(query)
    llm = get_llm(temperature=0.2)

    prompt = f"""{SYSTEM_PROMPT}

[참고 자료]
{context}

[질문]
{query}{line_req}

[답변]"""

    resp = llm.invoke(prompt)
    text = (getattr(resp, "content", None) or str(resp)).strip()

    return {"answer": text, "sources": sources_of(docs), "contexts": docs}
