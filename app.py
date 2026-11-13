"""Company information UI; ingestion and LLM calls are explicit actions."""

import streamlit as st

import ingest
import rag
from config import LLM_MODE

st.set_page_config(page_title="동진 기업정보 RAG", layout="wide")
st.title("기업정보 자료 검색")
st.caption(f"{LLM_MODE} 모드 · 자료의 출처와 실제 응답 방식을 함께 표시합니다.")
if st.sidebar.button("저장된 수집 자료 적재"):
    try:
        with st.spinner("저장된 자료를 적재합니다."):
            result = ingest.build_database()
        rag.load_vectordb.clear()
        st.sidebar.success(f"{result['index_count']}개 청크")
    except (ValueError, OSError) as exc:
        st.sidebar.error(str(exc))
question = st.chat_input("예: 사업분야 전체 목록은 무엇인가요?")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            result = rag.answer(question)
            st.write(result["answer"])
            st.caption(
                "생성 모델 응답"
                if result["generation_mode"].startswith("generated")
                else "검색 근거 또는 정형 조회 결과"
            )
            for source in result["sources"]:
                st.caption(source)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
