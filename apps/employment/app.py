"""Private, local-only PDF article retrieval with visible page evidence."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path

import streamlit as st

from document_rag.documents import documents_from_pdf
from document_rag.retrieval import answer, ingest, open_index

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path(
    os.environ.get("EMPLOYMENT_PRIVATE_DIR", ROOT / "results/private/employment")
)
st.set_page_config(page_title="취업규칙 문서 검색", layout="wide")
st.title("취업규칙 문서 검색")
st.caption("PDF를 로컬에서 처리하고, 검색 근거의 파일명·조항·실제 페이지를 표시합니다.")
upload = st.file_uploader("취업규칙 PDF", type=["pdf"])
identity = hashlib.sha256(upload.getvalue()).hexdigest() if upload else None
if upload and st.button("검색 자료 준비", type="primary"):
    safe_name = re.sub(
        r"[^\w.() -]", "_", unicodedata.normalize("NFC", Path(upload.name).name)
    )
    folder = PRIVATE / identity
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / safe_name
    path.write_bytes(upload.getvalue())
    try:
        with st.spinner("PDF 조항을 추출하고 검색 인덱스를 만듭니다."):
            database = open_index(folder / "index", "employment")
            count = ingest(database, documents_from_pdf(path))
        st.session_state["employment_ready"] = identity
        st.success(f"{count}개 청크 준비 완료")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
question = st.chat_input("예: 제8조 내용을 확인해 주세요.")
if question:
    if not identity or st.session_state.get("employment_ready") != identity:
        st.info("현재 PDF의 검색 자료를 먼저 준비하세요.")
    else:
        database = open_index(PRIVATE / identity / "index", "employment")
        with st.spinner("자료를 검색합니다."):
            result = answer(database, question)
        st.write(result["answer"])
        st.caption(
            "생성 모델 응답"
            if result["generation_mode"].startswith("generated")
            else "검색 근거를 그대로 표시한 결과"
        )
        for source in result["sources"]:
            st.caption(
                f"{source['source']} · {source.get('article', '')} · PDF {source.get('pages', '')}쪽"
            )
