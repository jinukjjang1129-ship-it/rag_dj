"""app_streamlit.py — 순천향대 RAG 챗봇 (학과정보 / 입시정보 / 논문정보 3탭 통합)."""

from __future__ import annotations

import streamlit as st
from common import config, history

from rag import admission, department, engine, paper

DOMAIN_MODULES = {
    "department": department,
    "admission": admission,
    "paper": paper,
}

st.set_page_config(page_title="순천향대 RAG 챗봇", page_icon="🎓", layout="wide")

# ── 세션 상태 초기화 ──
if "active_domain" not in st.session_state:
    st.session_state.active_domain = "department"
for domain in config.DOMAINS:
    if f"{domain}_session_id" not in st.session_state:
        st.session_state[f"{domain}_session_id"] = history.new_session_id()
    if f"{domain}_messages" not in st.session_state:
        st.session_state[f"{domain}_messages"] = []


def start_new_conversation(domain: str) -> None:
    st.session_state[f"{domain}_session_id"] = history.new_session_id()
    st.session_state[f"{domain}_messages"] = []


def load_conversation(domain: str, session_id: str) -> None:
    st.session_state[f"{domain}_session_id"] = session_id
    rows = history.get_messages(session_id)
    st.session_state[f"{domain}_messages"] = [
        {"role": r["role"], "content": r["content"], "total_sec": r.get("total_sec")}
        for r in rows
    ]


# ── 사이드바 ──
with st.sidebar:
    st.markdown("### 🎓 순천향대 RAG")

    domain = st.radio(
        "탭 선택",
        options=list(config.DOMAINS.keys()),
        format_func=lambda d: config.DOMAINS[d]["label"],
        key="active_domain",
        horizontal=True,
    )
    domain_cfg = config.DOMAINS[domain]
    domain_module = DOMAIN_MODULES[domain]

    if st.button("🆕 새 대화", use_container_width=True):
        start_new_conversation(domain)
        st.rerun()

    with st.expander("💬 대화 기록", expanded=False):
        search_query = st.text_input(
            "기록 검색", key=f"search_{domain}", placeholder="키워드로 검색"
        )
        if search_query:
            hits = history.search_messages(domain, search_query)
            if not hits:
                st.caption("검색 결과가 없습니다.")
            for h in hits:
                label = f"{h['title']} · {h['content'][:24]}"
                if st.button(label, key=f"hit_{h['session_id']}_{h['created_at']}"):
                    load_conversation(domain, h["session_id"])
                    st.rerun()
        else:
            sessions = history.list_sessions(domain)
            if not sessions:
                st.caption("아직 대화 기록이 없습니다.")
            for s in sessions:
                if st.button(f"🗂️ {s['title']}", key=f"sess_{s['session_id']}"):
                    load_conversation(domain, s["session_id"])
                    st.rerun()

    with st.expander("ℹ️ 안내", expanded=False):
        st.markdown(
            f"""
**{domain_cfg["label"]} 탭**

- 이 탭은 다른 탭과 별도의 데이터/대화로 동작합니다.
- 답변은 검색된 순천향대 자료에 근거해서만 생성됩니다.
- 관련 자료를 찾지 못하면 추측하지 않고 "확인되지 않습니다"라고 답합니다.
- 왼쪽 위에서 다른 탭으로 전환할 수 있습니다.
            """
        )

    with st.expander("📊 학습현황", expanded=False):
        try:
            status = domain_module.get_status()
            st.metric("적재 청크 수", status["total_chunks"])
            st.metric(
                "적재 항목", f"{status['trained_count']} / {status['total_count']}"
            )
            for item in status["items"]:
                icon = "✅" if item.get("trained") else "⬜"
                name = (
                    item.get("department")
                    or item.get("section")
                    or item.get("source", "?")
                )
                cnt = item.get("chunk_count", 0)
                st.caption(f"{icon} {name} ({cnt}청크)")
        except Exception as exc:
            st.caption(f"학습현황을 불러오지 못했습니다: {exc}")

    st.divider()
    st.caption("💡 추천 질문")
    for q in domain_cfg["suggested_questions"]:
        if st.button(q, key=f"suggest_{domain}_{q}", use_container_width=True):
            st.session_state[f"{domain}_pending_question"] = q


# ── 메인 영역 ──
st.title(f"{domain_cfg['label']}")

messages = st.session_state[f"{domain}_messages"]
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("total_sec") is not None:
            st.caption(f"⏱ {msg['total_sec']}초")

pending_key = f"{domain}_pending_question"
question = st.chat_input(f"{domain_cfg['label']}에 대해 질문해 주세요")
if not question and st.session_state.get(pending_key):
    question = st.session_state.pop(pending_key)

if question:
    session_id = st.session_state[f"{domain}_session_id"]
    history.ensure_session(session_id, domain, title_hint=question)

    st.session_state[f"{domain}_messages"].append({"role": "user", "content": question})
    history.add_message(session_id, domain, "user", question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("검색하고 답변을 생성하는 중..."):
            if not engine.has_data(domain):
                answer_text = domain_cfg["not_ready_message"]
                total_sec = 0.0
                sources = []
            else:
                result = domain_module.answer(question)
                answer_text = result["answer"]
                total_sec = result["timing"]["total_sec"]
                sources = result.get("sources", [])

        st.markdown(answer_text)
        st.caption(f"⏱ {total_sec}초")

        if sources:
            with st.expander("📎 참고한 자료"):
                for s in sources:
                    label = (
                        "메타데이터 일치"
                        if s.get("score") is None
                        else f"검색 유사도 {s['score']} (정확도 아님)"
                    )
                    st.markdown(f"**[{s['rank']}] {s['title']}** ({label})")
                    st.caption(s["text"][:300])

    st.session_state[f"{domain}_messages"].append(
        {"role": "assistant", "content": answer_text, "total_sec": total_sec}
    )
    history.add_message(
        session_id, domain, "assistant", answer_text, total_sec=total_sec
    )
    history.rename_session_if_untitled(session_id, question)
