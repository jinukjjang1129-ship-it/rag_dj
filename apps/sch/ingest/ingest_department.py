"""ingest/ingest_department.py — data/chunks.json(학과 홈페이지 크롤링 결과)을"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DOMAIN_CFG = config.DOMAINS["department"]

NOISE_MARKERS = ("등록된 글이 없습니다",)
EMPTY_STAT_HEADERS = ("취업현황", "취업분야", "근무지역", "초임월급여")


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.LOCAL_EMBED_MODEL)


def build_embed_text(chunk: dict) -> str:
    parts = [
        chunk.get("department", ""),
        chunk.get("title", ""),
        chunk.get("search_terms", ""),
        chunk.get("text", ""),
    ]
    return "\n".join(p.strip() for p in parts if p and p.strip())


def is_noise_chunk(text: str) -> tuple[bool, str]:
    if not text or len(text.strip()) < 10:
        return True, "내용 없음"
    for marker in NOISE_MARKERS:
        if marker in text:
            return True, f"'{marker}' 안내만 있음"
    if all(h in text for h in EMPTY_STAT_HEADERS) and not re.search(r"\d", text):
        return True, "취업통계 표 제목만 있고 수치 없음(동적 페이지)"
    return False, ""


def build_documents() -> tuple[List[Document], List[str]]:
    with open(config.CHUNKS_JSON, encoding="utf-8") as f:
        chunks = json.load(f)

    docs: List[Document] = []
    doc_ids: List[str] = []
    skipped = 0

    for i, c in enumerate(chunks):
        text = (c.get("text") or "").strip()
        noise, reason = is_noise_chunk(text)
        if noise:
            skipped += 1
            print(f"  [건너뜀] {reason}: [{c.get('department')}] {c.get('title')}")
            continue

        content_type = c.get("content_type", "")
        if content_type == "professor" and not re.fullmatch(
            r".{2,30}\s교수(?:님)?", c.get("title", "").strip()
        ):
            content_type = "general"
        docs.append(
            Document(
                page_content=build_embed_text(c),
                metadata={
                    "department": c.get("department", ""),
                    "content_type": content_type,
                    "section": c.get("section", ""),
                    "title": c.get("title", ""),
                    "url": c.get("url", ""),
                    "chunk_no": c.get("chunk_no", 0),
                    "raw_text": text,
                },
            )
        )
        doc_ids.append(f"chunk-{i}")

    print(f"\n총 {len(chunks)}개 중 {len(docs)}개 적재 대상 ({skipped}개 건너뜀)")
    return docs, doc_ids


def main():
    reset = "--reset" in sys.argv

    if not config.CHUNKS_JSON.exists():
        print(f"[중단] {config.CHUNKS_JSON} 이 없습니다.")
        return

    if reset and Path(DOMAIN_CFG["persist_dir"]).exists():
        shutil.rmtree(DOMAIN_CFG["persist_dir"])
        print(f"기존 DB 삭제: {DOMAIN_CFG['persist_dir']}")

    print(f"임베딩 모델: {config.LOCAL_EMBED_MODEL}")
    docs, doc_ids = build_documents()
    if not docs:
        print("[중단] 적재할 문서가 없습니다.")
        return

    vectordb = Chroma(
        collection_name=DOMAIN_CFG["collection"],
        persist_directory=DOMAIN_CFG["persist_dir"],
        embedding_function=get_embeddings(),
        collection_metadata={
            "hnsw:space": "cosine",
            "embedding_model": config.LOCAL_EMBED_MODEL,
        },
    )
    vectordb.add_documents(docs, ids=doc_ids)

    from collections import Counter

    dept_counts = Counter(d.metadata["department"] for d in docs)
    print(f"\n완료. 총 {len(docs)}개 청크 적재됨 -> {DOMAIN_CFG['persist_dir']}")
    print("\n학과별 적재 현황:")
    for dept, cnt in dept_counts.most_common():
        print(f"  {dept:<25} {cnt:>4}청크")
    print("\n이제 'streamlit run app_streamlit.py' 로 실행하세요.")


if __name__ == "__main__":
    main()
