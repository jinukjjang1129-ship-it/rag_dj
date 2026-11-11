"""Evaluate SCH retrieval on a shared private index, with generation disabled.

Run each revision in a fresh process. Export counts and metrics, never source
texts, professor names, embeddings, dialogue history or credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path


def fingerprint(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def dataset() -> list[dict]:
    cases = []
    departments = [
        "의료IT공학과",
        "전자공학과",
        "AI빅데이터학과",
        "정보통신공학과",
        "컴퓨터소프트웨어공학과",
    ]
    for i, department in enumerate(departments):
        cases.append(
            {
                "id": f"faculty-{i}",
                "domain": "department",
                "question": f"{department} 교수님은 누가 있나요?",
                "filters": {"department": department, "content_type": "professor"},
                "kind": "list",
            }
        )
        cases.append(
            {
                "id": f"curriculum-{i}",
                "domain": "department",
                "question": f"{department} 교육과정을 알려주세요.",
                "filters": {"department": department, "section": "교육과정"},
                "kind": "retrieval",
            }
        )
    for i, (q, section) in enumerate(
        [
            ("수시모집 원서접수 일정이 어떻게 되나요?", "원서접수·전형일정·합격·등록"),
            ("수시 전형료는 얼마인가요?", "전형료"),
            ("신입생 장학금 지급 기준을 알려주세요.", "신입생 장학금"),
            ("입학처 연락처를 알려주세요.", "입학처 연락처"),
            ("학생부 교과 성적 반영 방법을 알려주세요.", "학교생활기록부 반영 방법"),
        ]
    ):
        cases.append(
            {
                "id": f"admission-{i}",
                "domain": "admission",
                "question": q,
                "filters": {"section": section},
                "kind": "retrieval",
            }
        )
    for i, phrase in enumerate(
        ["FPGA", "AI 반도체", "HTTP Dos", "AI 리터러시", "웹 취약점"]
    ):
        cases.append(
            {
                "id": f"paper-{i}",
                "domain": "paper",
                "question": f"{phrase} 관련 논문 내용을 알려주세요.",
                "source_contains": phrase,
                "kind": "retrieval",
            }
        )
    for i, question in enumerate(
        ["오늘 날씨 알려줘", "비트코인 가격 예측해줘", "저녁 메뉴 추천해줘"]
    ):
        cases.append(
            {
                "id": f"negative-{i}",
                "domain": "department",
                "question": question,
                "kind": "negative",
            }
        )
    return cases


def run(project: Path, index: Path, output: Path, label: str) -> None:
    sys.path.insert(0, str(project.resolve()))
    from common import config

    from rag import admission, department, engine, paper

    config.USE_OLLAMA = False
    for domain in config.DOMAINS:
        config.DOMAINS[domain]["persist_dir"] = str((index / domain).resolve())
    modules = {"department": department, "admission": admission, "paper": paper}
    all_docs = {}
    for domain in modules:
        got = engine.get_vectordb(domain).get(include=["metadatas", "documents"])
        all_docs[domain] = list(zip(got["metadatas"], got["documents"], strict=True))
    # Separate first model-load/index-open latency from repeated request latency.
    engine.search("department", "컴퓨터소프트웨어공학과 교육과정", top_k=3)
    rows = []
    for case in dataset():
        gold = []
        for meta, doc in all_docs[case["domain"]]:
            if case["kind"] == "negative":
                continue
            if any(meta.get(k) != v for k, v in case.get("filters", {}).items()):
                continue
            if (
                "source_contains" in case
                and case["source_contains"].lower()
                not in meta.get("source", "").lower()
            ):
                continue
            if case["kind"] == "list" and not re.fullmatch(
                r".{2,30}\s교수(?:님)?", meta.get("title", "")
            ):
                continue
            gold.append((meta, doc))
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            result = modules[case["domain"]].answer(case["question"])
            timings.append(time.perf_counter() - started)
        sources = result.get("sources", [])
        gold_ids = {fingerprint(m.get("raw_text", d)) for m, d in gold}
        found_ids = {fingerprint(s.get("text", "")) for s in sources}
        intersection = gold_ids & found_ids
        row = {
            "id": case["id"],
            "kind": case["kind"],
            "domain": case["domain"],
            "gold_count": len(gold_ids),
            "returned_count": len(sources),
            "duplicate_count": len(sources) - len(found_ids),
            "retrieval_hit": bool(intersection) if gold_ids else None,
            "evidence_recall": len(intersection) / len(gold_ids) if gold_ids else None,
            "evidence_precision": len(intersection) / len(found_ids)
            if found_ids
            else None,
            "latency_median_sec": statistics.median(timings),
            "generation_accuracy": None,
            "generation_mode": result.get("generation_mode", "extractive_fallback"),
            "route": result.get("route", "vector"),
            "correct_abstention": not bool(result.get("in_scope", True))
            if not gold_ids
            else None,
        }
        if case["kind"] == "list" and gold_ids:
            titles = {m["title"] for m, _ in gold}
            row["list_answer_coverage"] = sum(
                t in result["answer"] for t in titles
            ) / len(titles)
        rows.append(row)
    positive = [r for r in rows if r["retrieval_hit"] is not None]
    lists = [r for r in rows if "list_answer_coverage" in r]
    negatives = [r for r in rows if r["correct_abstention"] is not None]
    report = {
        "label": label,
        "protocol": "23 metadata-labelled questions; USE_OLLAMA=0; 3 warm repetitions",
        "cases": len(rows),
        "positive_cases": len(positive),
        "retrieval_hit_rate": statistics.mean(r["retrieval_hit"] for r in positive),
        "mean_evidence_recall": statistics.mean(r["evidence_recall"] for r in positive),
        "mean_list_answer_coverage": statistics.mean(
            r["list_answer_coverage"] for r in lists
        ),
        "correct_abstention_rate": statistics.mean(
            r["correct_abstention"] for r in negatives
        ),
        "request_latency_median_sec": statistics.median(
            r["latency_median_sec"] for r in rows
        ),
        "duplicate_sources": sum(r["duplicate_count"] for r in rows),
        "llm_answer_accuracy": None,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    run(args.project, args.index, args.output, args.label)
