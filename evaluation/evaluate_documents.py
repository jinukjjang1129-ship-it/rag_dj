"""Compare copied legacy code and current retrieval on the same private sources.

No source text or generated legal answer is exported. This is a retrospective
heading/source-level regression evaluation, not LLM factual-accuracy measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import textwrap
import time
import types
from pathlib import Path

from langchain_chroma import Chroma
from pypdf import PdfReader

from document_rag.documents import ARTICLE_HEADER, document_id, documents_from_pdf
from document_rag.retrieval import embeddings, open_index, retrieve

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str, dedent: bool = False):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(
        compile(
            textwrap.dedent(path.read_text()) if dedent else path.read_text(),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    return module


def run(pdf: Path, pages: Path, originals: Path, private: Path, output: Path):
    sys.path.insert(0, str(ROOT))
    import config

    config.PAGES_JSON = pages
    company = load_module(ROOT / "rag.py", "company_current")
    old_company = load_module(originals / "dj_rag_local/rag.py", "company_legacy")
    old_company_ingest = load_module(
        originals / "dj_rag_local/ingest.py", "company_legacy_ingest"
    )
    old_company_ingest.PAGES_JSON = pages
    employment_dir = next(originals.glob("rag_*"))
    old_ingest = load_module(employment_dir / "ingest.py", "employment_legacy_ingest")
    old_rag = load_module(
        employment_dir / "rag.py", "employment_legacy_rag", dedent=True
    )
    model = embeddings()
    legacy_pdf_docs = old_ingest.build_documents(pdf)
    legacy_pdf = Chroma(
        collection_name="legacy_employment",
        persist_directory=str(private / "pdf_legacy"),
        embedding_function=model,
        collection_metadata={"hnsw:space": "l2"},
    )
    legacy_pdf.add_documents(
        legacy_pdf_docs, ids=[document_id(d) for d in legacy_pdf_docs]
    )
    new_pdf = open_index(private / "pdf_current", "employment")
    from document_rag.retrieval import ingest

    ingest(new_pdf, documents_from_pdf(pdf))
    legacy_company_docs = old_company_ingest.build_documents()
    legacy_company = Chroma(
        collection_name="legacy_company",
        persist_directory=str(private / "company_legacy"),
        embedding_function=model,
        collection_metadata={"hnsw:space": "l2"},
    )
    legacy_company.add_documents(
        legacy_company_docs, ids=[document_id(d) for d in legacy_company_docs]
    )
    new_company = company.load_vectordb()
    text = "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
    headings = list(ARTICLE_HEADER.finditer(text))
    cases = []
    # Five explicit article queries and five topic queries with source headings.
    for number in [1, 5, 7, 8, 12]:
        match = next(
            m for m in headings if re.sub(r"\s+", "", m.group(1)) == f"제{number}조"
        )
        cases.append(
            {
                "id": f"article-{number}",
                "question": f"제{number}조 내용을 확인해 주세요.",
                "anchor": re.sub(r"\s+", "", match.group()),
                "kind": "exact",
            }
        )
    for i, term in enumerate(["근로시간", "연차유급휴가", "휴일", "퇴직", "징계"]):
        match = next(
            (m for m in headings if term in re.sub(r"\s+", "", m.group())), None
        )
        if match:
            cases.append(
                {
                    "id": f"topic-{i}",
                    "question": f"{term} 관련 규정을 알려주세요.",
                    "anchor": re.sub(r"\s+", "", match.group()),
                    "kind": "semantic",
                }
            )
    report = {
        "protocol": "same raw PDF/pages and fixed questions; legacy PDF indentation repaired only; no LLM generation",
        "employment": {},
        "company": {},
        "llm_accuracy": None,
        "syntax_repair_required_for_legacy_employment": True,
        "raw_sources_published": False,
    }
    for name, retriever in [
        ("legacy", lambda q: old_rag.retrieve_scored(legacy_pdf, q, k=3)),
        ("current", lambda q: retrieve(new_pdf, q, k=3)),
    ]:
        rows = []
        retriever(cases[0]["question"])
        for case in cases:
            latencies = []
            for _ in range(3):
                started = time.perf_counter()
                pairs = retriever(case["question"])
                latencies.append(time.perf_counter() - started)
            rows.append(
                {
                    "id": case["id"],
                    "kind": case["kind"],
                    "heading_hit": any(
                        case["anchor"] in re.sub(r"\s+", "", d.page_content)
                        for d, _ in pairs
                    ),
                    "has_page_evidence": bool(pairs)
                    and all(bool(d.metadata.get("pages")) for d, _ in pairs),
                    "retrieved": len(pairs),
                    "latency_median_sec": statistics.median(latencies),
                }
            )
        report["employment"][name] = {
            "questions": len(rows),
            "heading_hit_rate": statistics.mean(r["heading_hit"] for r in rows),
            "page_evidence_rate": statistics.mean(r["has_page_evidence"] for r in rows),
            "latency_median_sec": statistics.median(
                r["latency_median_sec"] for r in rows
            ),
            "rows": rows,
        }
    queries = [
        ("반도체 제품을 요약해 주세요.", "business/semiconductor"),
        ("디스플레이 사업을 알려주세요.", "business/display"),
        ("신재생에너지 사업을 알려주세요.", "business/renewableenergy"),
        ("발포제 제품을 알려주세요.", "business/foamagen"),
        ("회사개요를 알려주세요.", "company/overview"),
        ("인재상과 채용을 알려주세요.", "recruit/hr01"),
        ("ESG 정보공개 자료를 알려주세요.", "sustainable/esg"),
        ("윤리경영 방침을 알려주세요.", "company/system"),
        ("혁신경영 방침을 알려주세요.", "company/quality_vision"),
        ("찾아오시는 길을 알려주세요.", "company/location"),
    ]
    for name in ["legacy", "current"]:
        rows = []
        for i, (query, fragment) in enumerate(queries):
            latencies = []
            for _ in range(3):
                started = time.perf_counter()
                if name == "legacy":
                    docs = old_company.retrieve(legacy_company, query, k=4)
                else:
                    categories = [
                        cat for word, cat in company.HINTS.items() if word in query
                    ]
                    where = (
                        {"category": {"$in": sorted(set(categories))}}
                        if categories
                        else None
                    )
                    docs = [
                        d for d, _ in retrieve(new_company, query, k=4, where=where)
                    ]
                latencies.append(time.perf_counter() - started)
            rows.append(
                {
                    "id": f"company-{i}",
                    "source_hit": any(
                        fragment in d.metadata.get("source", "") for d in docs
                    ),
                    "retrieved": len(docs),
                    "latency_median_sec": statistics.median(latencies),
                }
            )
        broad = "사업분야 전체 목록은 무엇인가요?"
        if name == "legacy":
            docs = old_company.retrieve(legacy_company, broad, k=4)
            business_count = len(
                {
                    d.metadata.get("category")
                    for d in docs
                    if d.metadata.get("category") in company.BUSINESS
                }
            )
        else:
            result = company.answer(broad, generate=False)
            business_count = len(result["sources"])
        report["company"][name] = {
            "questions": len(rows),
            "source_hit_rate": statistics.mean(r["source_hit"] for r in rows),
            "business_category_coverage": business_count / 4,
            "latency_median_sec": statistics.median(
                r["latency_median_sec"] for r in rows
            ),
            "rows": rows,
        }
    negatives = [
        "오늘 서울 날씨 알려줘",
        "비트코인 가격 예측해줘",
        "양자 중력 연구를 설명해줘",
    ]
    report["unrelated_evidence_checks"] = {
        "employment_legacy_returned": sum(
            bool(old_rag.retrieve_scored(legacy_pdf, q, k=3)) for q in negatives
        ),
        "employment_current_returned": sum(
            bool(retrieve(new_pdf, q, k=3)) for q in negatives
        ),
        "company_legacy_returned": sum(
            bool(old_company.retrieve(legacy_company, q, k=4)) for q in negatives
        ),
        "company_current_returned": sum(
            bool(company.answer(q, generate=False)["sources"]) for q in negatives
        ),
        "questions": len(negatives),
        "meaning": "Evidence presence for unrelated queries; not generated-answer accuracy",
    }
    report["chunk_counts"] = {
        "employment_legacy": len(legacy_pdf_docs),
        "employment_current": new_pdf._collection.count(),
        "company_legacy": len(legacy_company_docs),
        "company_current": new_company._collection.count(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--private-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.pdf, args.pages, args.originals, args.private_index, args.output)
