"""
app.py — 동진쎄미켐 정보 RAG 챗봇 (Streamlit)

구성:
  - 사이드바: 현재 모델/DB 상태, 예시 질문
  - 메인: 질문 → 페이지 근거 답변 + 출처(카테고리) 표시
"""
import os
import streamlit as st

st.set_page_config(page_title="동진쎄미켐 RAG", page_icon="🏭", layout="wide")

from config import (PERSIST_DOCS, LLM_MODE, API_LLM_MODEL, LOCAL_LLM_MODEL)
import rag as ragmod

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


def db_exists() -> bool:
    return os.path.isdir(PERSIST_DOCS)


@st.cache_resource(show_spinner="최초 실행: 자료를 벡터DB에 적재하는 중...")
def ensure_db():
    """
    벡터DB가 없으면 자동으로 적재한다.
    Streamlit Cloud는 처음 배포 시 chroma_store가 비어있으므로 이 과정이 필요.
    @st.cache_resource 로 최초 1회만 실행된다.
    """
    if db_exists():
        return "exists"
    import ingest
    ingest.main()          # pages.json → 벡터DB 적재
    return "built"


# 앱 시작 시 DB 보장 (없으면 자동 적재)
ensure_db()


# ── 사이드바 ──
with st.sidebar:
    st.header("동진쎄미켐 RAG")
    cur = LOCAL_LLM_MODEL if LLM_MODE == "local" else API_LLM_MODEL
    st.caption(f"모드: **{LLM_MODE}** · 모델: `{cur}`")
    st.caption(f"DB: {'있음' if db_exists() else '없음 (적재 필요)'}")

    st.divider()
    st.subheader("예시 질문")
    examples = [
        "동진쎄미켐 사업분야를 10줄로 요약해줘",
        "반도체 제품을 10줄로 요약해줘",
        "디스플레이 사업을 5줄로 정리해줘",
        "신재생에너지 분야는 뭘 하나요?",
        "발포제 주요 제품 알려줘",
        "회사 개요를 간단히 요약해줘",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending = ex

    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── 메인 ──
st.title("🏭 동진쎄미켐 정보 도우미")
st.caption("동진쎄미켐 홈페이지 자료에 근거해 답합니다. 근거가 없으면 '찾을 수 없습니다'라고 답합니다.")

if not db_exists():
    st.info("먼저 터미널에서 `python ingest.py` 로 자료를 적재하세요.")
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            with st.expander("출처"):
                for src in m["sources"]:
                    st.caption(f"· {src}")

typed = st.chat_input("예: 동진쎄미켐 사업분야를 10줄로 요약해줘")
q = typed or st.session_state.pending
st.session_state.pending = None

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("자료를 찾아 정리하는 중..."):
            try:
                result = ragmod.answer(q, k=4)
            except Exception as e:
                st.error(f"오류: {e}")
                st.stop()
        st.write(result["answer"])
        if result["sources"]:
            with st.expander("출처"):
                for src in result["sources"]:
                    st.caption(f"· {src}")
    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"], "sources": result["sources"],
    })
