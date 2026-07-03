"""
ingest.py — data/pages.json (크롤링 결과)을 정제하여 벡터DB에 적재.

이번 프로젝트의 핵심 처리:
1) 페이지마다 반복되는 메뉴·푸터·문의처를 잘라내고 '본문'만 남긴다.
   (안 자르면 "사업분야 요약" 질문에 메뉴/주소가 섞여 나옴)
2) URL에서 카테고리(디스플레이/반도체 등)를 자동 추출해 metadata로 넣는다.
   → "반도체 사업 요약해줘" 하면 그 페이지를 정확히 찾게 함.
3) 페이지가 길면 문단 단위로 분할하되, 카테고리는 유지한다.

사용법:
    python ingest.py            # 적재
    python ingest.py --reset    # 기존 DB 비우고 새로 적재
"""
import sys
import re
import json
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import (PAGES_JSON, PERSIST_DOCS, COL_DOCS,
                    get_embeddings, embedding_signature)

# URL 조각 → 사람이 읽는 카테고리 이름
CATEGORY_MAP = {
    "company/overview": "회사개요",
    "business/display": "디스플레이 (사업분야)",
    "business/semiconductor": "반도체 (사업분야)",
    "business/renewableenergy": "신재생에너지 (사업분야)",
    "business/foamagen": "발포제 (사업분야)",
}

# 본문 시작 지점 후보 (이 앞의 메뉴 덩어리를 잘라냄)
BODY_START = ["주요제품", "사업분야 소개", "국내 정밀화학"]
# 본문 끝 지점 후보 (이 뒤의 문의처/푸터를 잘라냄)
BODY_END = ["제품문의", "적용 제품 분야", "개인정보처리방침", "COPYRIGHT"]


def categorize(url: str) -> str:
    for key, name in CATEGORY_MAP.items():
        if key in url:
            return name
    # 폴백: /business/xxx.php → "xxx (business)"
    m = re.search(r"/([^/]+)/([^/]+)\.php", url)
    if m:
        return f"{m.group(2)} ({m.group(1)})"
    return "기타"


def clean_text(text: str) -> str:
    """반복되는 메뉴·푸터 제거 후 공백 정리."""
    for marker in BODY_START:
        idx = text.find(marker)
        if idx > 0:
            text = text[idx:]
            break
    for marker in BODY_END:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
            break
    return re.sub(r"\s+", " ", text).strip()


def build_documents() -> List[Document]:
    with open(PAGES_JSON, encoding="utf-8") as f:
        pages = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: List[Document] = []
    for p in pages:
        url = p.get("url", "")
        category = categorize(url)
        body = clean_text(p.get("text", ""))
        if not body:
            print(f"  [경고] 본문이 비어있음: {url}")
            continue

        # 짧으면 통째로, 길면 분할 (카테고리 metadata 유지)
        chunks = [body] if len(body) <= 900 else splitter.split_text(body)
        for i, ch in enumerate(chunks):
            docs.append(Document(
                page_content=ch,
                metadata={
                    "source": url,
                    "category": category,
                    "part": i,
                    "embed_sig": embedding_signature(),
                },
            ))
        print(f"  [{category}] {len(body)}자 → {len(chunks)}개 청크")
    return docs


def main():
    reset = "--reset" in sys.argv
    if reset and Path(PERSIST_DOCS).exists():
        shutil.rmtree(PERSIST_DOCS)
        print(f"기존 DB 삭제: {PERSIST_DOCS}")

    if not Path(PAGES_JSON).exists():
        print(f"[중단] {PAGES_JSON} 이 없습니다.")
        return

    print(f"임베딩: {embedding_signature()}\n")
    docs = build_documents()
    if not docs:
        print("[중단] 적재할 문서가 없습니다.")
        return

    vectordb = Chroma(
        collection_name=COL_DOCS,
        persist_directory=PERSIST_DOCS,
        embedding_function=get_embeddings(),
    )
    vectordb.add_documents(docs)

    cats = sorted({d.metadata["category"] for d in docs})
    print(f"\n완료. 총 {len(docs)}개 청크 적재됨 → {PERSIST_DOCS}")
    print(f"카테고리: {', '.join(cats)}")
    print("이제 'streamlit run app.py' 로 앱을 실행하세요.")


if __name__ == "__main__":
    main()
