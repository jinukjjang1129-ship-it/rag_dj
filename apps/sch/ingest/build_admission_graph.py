"""ingest/build_admission_graph.py — 입시요강 PDF에서 전형 정보를 뽑아"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config

from ingest.ingest_admission import build_documents_for_pdf

# "1. 모집인원" 처럼 번호가 붙은 필드 라벨들. 실제 PDF 문구 그대로.
FIELD_LABELS = (
    r"전형별\s*모집인원\s*및\s*지원자격",
    "모집인원",
    r"(?:공통\s*)?지원자격",
    "전형방법",
    r"\d{4}학년도\s*\|?\s*대학수학능력시험\s*\|?\s*최저학력기준",
    r"동점자\s*\|?\s*처리\s*\|?\s*기준",
    "선발원칙",
    "전형일정",
    "제출서류",
    "유의사항",
)
FIELD_PATTERN = re.compile(r"(\d+)\.\s*\|?\s*(" + "|".join(FIELD_LABELS) + r")")


def parse_track_fields(raw_text: str) -> dict[str, str]:
    """전형 하나의 원문에서 번호 붙은 필드들을 잘라낸다."""
    text = re.sub(r"\s+", " ", raw_text.replace("|", " "))
    matches = list(FIELD_PATTERN.finditer(text))
    fields: dict[str, str] = {}
    for i, m in enumerate(matches):
        field_name = re.sub(r"\s+", "", m.group(2))
        if field_name == "공통지원자격":
            field_name = "지원자격"
        field_name = re.sub(
            r"\d{4}학년도대학수학능력시험최저학력기준", "수능최저학력기준", field_name
        )
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fields[field_name] = text[start:end].strip(" :").strip()
    return fields


def extract_headcount(모집인원_필드: str) -> Optional[int]:
    """'689명(p.5 참조)' 같은 문자열에서 정원 숫자만 뽑는다."""
    m = re.search(r"(\d[\d,]*)\s*명", 모집인원_필드 or "")
    return int(m.group(1).replace(",", "")) if m else None


EXPECTED_FIELDS = {
    "모집인원",
    "지원자격",
    "전형방법",
    "수능최저학력기준",
    "동점자처리기준",
    "선발원칙",
    "전형일정",
    "제출서류",
    "유의사항",
}


def collect_tracks(pdf_path: Path) -> list[dict]:
    """PDF에서 전형별 정보를 전부 뽑아 리스트로 반환 (Neo4j 적재 전 중간 형태)."""
    docs, ids, section_counts = build_documents_for_pdf(pdf_path)

    # 같은 전형의 청크를 (pdf_page, chunk_no) 순서로 모은다.
    track_chunks: dict[str, list] = defaultdict(list)
    track_category: dict[str, str] = {}
    for d in docs:
        track = d.metadata.get("track")
        if not track:
            continue
        track_chunks[track].append(d)
        track_category[track] = d.metadata.get("track_category", "")

    tracks = []
    for track_name, chunk_docs in track_chunks.items():
        chunk_docs.sort(key=lambda d: (d.metadata["pdf_page"], d.metadata["chunk_no"]))
        raw_text = "".join(d.metadata["raw_text"] for d in chunk_docs)

        fields = parse_track_fields(raw_text)
        missing = EXPECTED_FIELDS - set(fields.keys())

        tracks.append(
            {
                "name": track_name,
                "category": track_category[track_name],
                "headcount": extract_headcount(fields.get("모집인원", "")),
                "fields": fields,
                "missing_fields": sorted(missing),
                "raw_text": raw_text,
                "source": pdf_path.name,
                "pages": sorted({d.metadata["pdf_page"] for d in chunk_docs}),
            }
        )
    return tracks


def print_dry_run(tracks: list[dict]) -> None:
    ok = sum(1 for t in tracks if not t["missing_fields"])
    print(f"전형 {len(tracks)}개 중 9개 필드 전부 파싱됨: {ok}개\n")
    for t in tracks:
        status = "✅" if not t["missing_fields"] else f"⚠️ 누락: {t['missing_fields']}"
        print(f"[{t['category']}] {t['name']}  (정원 {t['headcount']}명)  {status}")


def build_graph(tracks: list[dict], reset: bool = False) -> None:
    """세부전형 노드를 만들고, 여러 전형이 값을 공유하는 필드는 별도 노드로 뺀다.

    [왜 4개 필드만 노드로 빼는가]
    실제 데이터를 확인해보니 전형일정/제출서류/유의사항/선발원칙은 전형마다
    내용이 다른 게 아니라 대부분 "다른 페이지 참조하세요" 같은 동일한
    포인터 문구였다 (예: 전형일정 20개 전형 전부 "원서접수 및 전형일정
    p.10 참조"로 동일). 이런 필드를 세부전형 속성으로 20번 중복 저장하는
    대신, 공유 노드 하나로 빼고 여러 세부전형이 그 노드를 같이 가리키게
    하면 MERGE가 자동으로 중복을 없애준다. 반대로 지원자격/전형방법/
    수능최저학력기준/동점자처리기준은 전형마다 실제로 다 다른 내용이라
    노드로 빼봤자 1:1 매달림만 생겨서 그대로 속성으로 둔다.
    """
    from neo4j import GraphDatabase

    uri = os.environ["NEO4J_URI"]
    password = os.environ["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=("neo4j", password))
    driver.verify_connectivity()

    with driver.session() as session:
        if reset:
            session.run("MATCH (t:세부전형) DETACH DELETE t")
            session.run("MATCH (tt:전형유형) DETACH DELETE tt")
            session.run("MATCH (c:공통안내) DETACH DELETE c")
            session.run("MATCH (s:선발원칙유형) DETACH DELETE s")
            print("기존 전형 관련 노드 전부 삭제됨")

        for t in tracks:
            session.run(
                """
                MERGE (tt:전형유형 {name: $category})
                MERGE (track:세부전형 {name: $name})
                SET track.전체모집인원 = $headcount,
                    track.지원자격 = $지원자격,
                    track.전형방법 = $전형방법,
                    track.수능최저학력기준 = $수능최저학력기준,
                    track.동점자처리기준 = $동점자처리기준,
                    track.원문 = $raw_text,
                    track.source = $source,
                    track.pages = $pages
                MERGE (tt)-[:포함]->(track)
                """,
                category=t["category"],
                name=t["name"],
                headcount=t["headcount"],
                지원자격=t["fields"].get("지원자격", ""),
                전형방법=t["fields"].get("전형방법", ""),
                수능최저학력기준=t["fields"].get("수능최저학력기준", ""),
                동점자처리기준=t["fields"].get("동점자처리기준", ""),
                raw_text=t["raw_text"],
                source=t.get("source", ""),
                pages=t.get("pages", []),
            )

            # 공유 가능성이 큰 4개 필드는 별도 노드로 (MERGE로 자동 중복 제거)
            # 관계 이름을 전부 ":참조"로 뭉뚱그리면 그래프에서 "이게 전형일정
            # 참조인지 유의사항 참조인지" 화살표만 보고 구분이 안 된다.
            # 필드명을 그대로 관계 이름으로 써서(:전형일정, :제출서류, :유의사항)
            # 선발원칙(:선발원칙)과 같은 방식으로 통일한다.
            # Cypher는 관계 타입을 파라미터로 못 받아서(f-string으로 직접
            # 넣어야 함) 조심해야 하는데, field_type이 위 튜플의 고정된
            # 3개 값 중 하나만 들어오므로 사용자 입력이 섞일 위험은 없다.
            for field_type in ("전형일정", "제출서류", "유의사항"):
                value = t["fields"].get(field_type, "")
                if not value:
                    continue
                session.run(
                    f"""
                    MATCH (track:세부전형 {{name: $name}})
                    MERGE (c:공통안내 {{type: $field_type, text: $value}})
                    MERGE (track)-[:{field_type}]->(c)
                    """,
                    name=t["name"],
                    field_type=field_type,
                    value=value,
                )

            선발원칙 = t["fields"].get("선발원칙", "")
            if 선발원칙:
                session.run(
                    """
                    MATCH (track:세부전형 {name: $name})
                    MERGE (s:선발원칙유형 {name: $선발원칙})
                    MERGE (track)-[:선발원칙]->(s)
                    """,
                    name=t["name"],
                    선발원칙=선발원칙,
                )

            print(f"  적재: [{t['category']}] {t['name']}")

    driver.close()
    print(f"\n완료. 세부전형 노드 {len(tracks)}개 적재됨.")
    print(
        "공통안내(전형일정/제출서류/유의사항)와 선발원칙유형은 중복 제거되어 공유됩니다."
    )


def main():
    dry_run = "--dry-run" in sys.argv
    reset = "--reset" in sys.argv

    pdf_files = sorted(config.ADMISSION_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"[중단] {config.ADMISSION_PDF_DIR} 에 PDF가 없습니다.")
        return

    all_tracks = []
    for pdf_path in pdf_files:
        print(f"파싱 중: {pdf_path.name}")
        all_tracks.extend(collect_tracks(pdf_path))

    if dry_run:
        print_dry_run(all_tracks)
        print("\n(--dry-run 모드라 Neo4j에는 아무것도 안 씀)")
        return

    build_graph(all_tracks, reset=reset)


if __name__ == "__main__":
    main()
