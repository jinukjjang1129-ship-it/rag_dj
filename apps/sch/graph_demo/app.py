"""graph_demo/app.py — 입시정보 GraphRAG 시각화 데모 백엔드."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager

import neo4j
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import admission as admission_rag

load_dotenv()

driver: Optional[neo4j.Driver] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global driver
    driver = neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=("neo4j", os.environ["NEO4J_PASSWORD"]),
    )
    driver.verify_connectivity()
    print("✅ Neo4j 연결 완료")
    yield
    driver.close()
    print("🔌 Neo4j 드라이버 종료")


app = FastAPI(title="순천향대 입시정보 GraphRAG 데모", lifespan=lifespan)
app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GRAPH_LABELS = ("전형유형", "세부전형", "공통안내", "선발원칙유형")

NUMERIC_KEYWORDS = ("몇 명", "몇명", "정원", "인원", "모집인원", "몇 명이야", "얼마")

# "학생부교과 전형에는 뭐가 있어"처럼, 특정 속성이 아니라 그 아래 세부전형
# 목록 자체를 묻는 질문. 이 키워드가 있으면 전형유형→세부전형 관계를 그대로
# 나열해서 답한다 (지원자격/모집인원 같은 속성 조회와는 다른 종류의 질문).
LIST_KEYWORDS = (
    "뭐가 있어",
    "어떤 전형",
    "어떤 게 있어",
    "무엇이 있",
    "종류",
    "목록",
    "리스트",
    "전형이 있",
)

# 세부전형 노드에 이미 저장돼 있는 속성들. 질문에 이 키워드가 있으면
# 벡터검색 없이 그래프에서 속성값을 바로 꺼내본다 — "지원자격도 그래프로
# 되는지" 테스트해보기 위한 확장.
FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "지원자격": ("지원자격", "지원 자격", "지원할 수 있", "지원 가능", "자격이"),
    "전형방법": (
        "전형방법",
        "전형 방법",
        "반영비율",
        "반영 비율",
        "어떻게 뽑",
        "점수산출",
    ),
    "수능최저학력기준": ("수능최저", "최저학력기준", "수능 최저", "등급컷", "등급 컷"),
    "동점자처리기준": ("동점자", "동점 처리"),
}

# 그래프 조회 답변을 자연스러운 문장으로 포맷팅할 때 쓰는 표현.
PROP_LABELS: dict[str, str] = {
    "전체모집인원": "모집인원",
    "지원자격": "지원자격",
    "전형방법": "전형방법",
    "수능최저학력기준": "수능최저학력기준",
    "동점자처리기준": "동점자처리기준",
}


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    used_nodes: List[str]
    used_edges: List[str]
    mode: str  # "graph" | "vector"
    sources: List[dict] = Field(default_factory=list)


@app.get("/")
async def root():
    return FileResponse(str(Path(__file__).parent / "index.html"))


@app.get("/graph")
def get_graph():
    """입시 그래프 전체 구조 반환 (시각화용).

    properties(n)을 그대로 다 돌려주면 세부전형 노드의 '원문' 필드(최대
    4000자 가까운 PDF 페이지 텍스트)까지 전부 포함돼서 응답이 무거워진다.
    시각화에 실제로 필요한 필드만 골라서 돌려준다.
    """
    t0 = time.time()
    print("[/graph] 요청 시작")
    try:
        with driver.session() as session:
            label_filter = " OR ".join(f"n:{lbl}" for lbl in GRAPH_LABELS)
            nodes_result = session.run(f"""
                MATCH (n)
                WHERE {label_filter}
                RETURN elementId(n) as id, labels(n)[0] as label,
                       n.name as name, n.type as type, n.전체모집인원 as headcount
            """)
            nodes = []
            for record in nodes_result:
                label = record["label"]
                title = record["name"] or record["type"] or "제목없음"
                props = (
                    {"headcount": record["headcount"]} if label == "세부전형" else {}
                )
                nodes.append(
                    {
                        "id": record["id"],
                        "label": label,
                        "title": title,
                        "properties": props,
                    }
                )

            edge_filter_n = " OR ".join(f"n:{lbl}" for lbl in GRAPH_LABELS)
            edge_filter_m = " OR ".join(f"m:{lbl}" for lbl in GRAPH_LABELS)
            edges_result = session.run(f"""
                MATCH (n)-[r]->(m)
                WHERE ({edge_filter_n}) AND ({edge_filter_m})
                RETURN elementId(r) as id, elementId(n) as source, elementId(m) as target, type(r) as relationship
            """)
            edges = [
                {
                    "id": r["id"],
                    "source": r["source"],
                    "target": r["target"],
                    "relationship": r["relationship"],
                }
                for r in edges_result
            ]

            print(
                f"[/graph] 완료: 노드 {len(nodes)}개, 관계 {len(edges)}개, {time.time() - t0:.2f}초 소요"
            )
            return {"nodes": nodes, "edges": edges}
    except Exception as e:
        print(f"[/graph] 에러: {e}")
        raise HTTPException(status_code=500, detail=f"그래프 조회 실패: {e}")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

    where, track_desc = admission_rag.detect_track_filter(question)

    # ── "학생부교과에는 뭐가 있어"처럼 세부전형 목록 자체를 묻는 질문 ──
    # track_category(예: "학생부교과")로 좁혀졌을 때만 의미가 있다.
    # 특정 세부전형 하나로 좁혀진 경우("지역의사전형")는 이미 그 자체가
    # 목록의 원소 하나라서 나열할 게 없다.
    is_list_query = any(kw in question for kw in LIST_KEYWORDS)
    if where and is_list_query and "track_category" in where:
        category = where["track_category"]
        try:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (tt:전형유형 {name: $category})-[:포함]->(t:세부전형)
                    RETURN elementId(t) as id, t.name as name, t.전체모집인원 as count, t.source as source, t.pages as pages
                    """,
                    category=category,
                )
                rows = [dict(r) for r in result]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"그래프 질의 실패: {e}")

        if rows:
            list_text = "\n".join(
                f"- {r['name']}"
                + (f" (모집인원 {r['count']}명)" if r["count"] is not None else "")
                for r in rows
            )
            results_for_llm = [
                {
                    "rank": 1,
                    "score": 100.0,
                    "title": f"{category} 세부전형 목록",
                    "section": "전형유형 → 세부전형",
                    "text": list_text,
                }
            ]
            # Preserve every retrieved item; an LLM summary can omit valid rows.
            answer = f"{category} 세부전형 목록입니다.\n{list_text}"
            used_nodes = [r["id"] for r in rows]
            return QueryResponse(
                answer=answer,
                used_nodes=used_nodes,
                used_edges=[],
                mode="graph",
                sources=[
                    {"source": r.get("source", ""), "pages": r.get("pages", [])}
                    for r in rows
                ],
            )

    # 질문에 어느 속성 키워드가 있는지 확인. "몇 명/정원"이면 전체모집인원,
    # 그 외엔 FIELD_KEYWORDS에 매칭되는 속성. 여러 개 걸리면 전부 다 답한다.
    target_props: List[str] = []
    if any(kw in question for kw in NUMERIC_KEYWORDS):
        target_props.append("전체모집인원")
    for prop, keywords in FIELD_KEYWORDS.items():
        if any(kw in question for kw in keywords):
            target_props.append(prop)

    # ── 전형이 특정되고 + 그래프에 있는 속성을 묻는 질문이면 그래프에서 직접 조회 (LLM 없이) ──
    if where and target_props:
        used_nodes: List[str] = []
        rows = []
        try:
            with driver.session() as session:
                return_clause = ", ".join(f"t.{p} as {p}" for p in target_props)
                if "track" in where:
                    track_names = where["track"]["$in"]
                    result = session.run(
                        f"""
                        MATCH (t:세부전형) WHERE t.name IN $names
                        RETURN elementId(t) as id, t.name as name, t.source as source, t.pages as pages, {return_clause}
                        """,
                        names=track_names,
                    )
                elif "track_category" in where:
                    category = where["track_category"]
                    result = session.run(
                        f"""
                        MATCH (tt:전형유형 {{name: $category}})-[:포함]->(t:세부전형)
                        RETURN elementId(t) as id, t.name as name, t.source as source, t.pages as pages, {return_clause}
                        """,
                        category=category,
                    )
                else:
                    result = []
                rows = [dict(r) for r in result] if result else []
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"그래프 질의 실패: {e}")

        if rows:
            # 그래프에서 뽑은 값을 "근거 자료"로 만들어서 LLM한테 넘긴다.
            # rag/engine.py의 make_llm_answer()가 기대하는 형식
            # (title/section/text)에 맞춰준다 — 벡터검색 결과를 넣던 자리에
            # 그래프 조회 결과를 대신 넣는 것뿐, 프롬프트 규칙(자료에 있는
            # 내용만 답변, 없는 건 추측 금지)은 그대로 적용된다.
            results_for_llm = []
            for i, r in enumerate(rows, start=1):
                text_lines = []
                for p in target_props:
                    value = r.get(p)
                    label = PROP_LABELS.get(p, p)
                    if value is None or value == "":
                        text_lines.append(f"{label}: 정보 없음")
                    elif p == "전체모집인원":
                        text_lines.append(f"{label}: {value}명")
                    else:
                        text_lines.append(f"{label}: {value}")
                results_for_llm.append(
                    {
                        "rank": i,  # engine.make_basic_answer()의 폴백 경로가 기대하는 필드
                        "score": 100.0,  # 그래프에서 정확히 조회한 값이라 관련도는 항상 만점으로 표기
                        "title": r["name"],
                        "section": track_desc,
                        "text": "\n".join(text_lines),
                    }
                )

            answer = "\n\n".join(f"{r['title']}\n{r['text']}" for r in results_for_llm)
            used_nodes = [r["id"] for r in rows]
            return QueryResponse(
                answer=answer,
                used_nodes=used_nodes,
                used_edges=[],
                mode="graph",
                sources=[
                    {"source": r.get("source", ""), "pages": r.get("pages", [])}
                    for r in rows
                ],
            )
        # 그래프에서 못 찾으면 벡터검색으로 폴백

    # ── 그 외(설명형 질문, 혹은 그래프 조회 실패) → 기존 벡터검색+LLM ──
    result = admission_rag.answer(question)
    return QueryResponse(
        answer=result["answer"],
        used_nodes=[],
        used_edges=[],
        mode="vector",
        sources=result.get("sources", []),
    )


@app.get("/health")
def health_check():
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        return {"status": "healthy", "neo4j": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
