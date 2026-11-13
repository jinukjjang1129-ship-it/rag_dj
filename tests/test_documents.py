"""Document-boundary, provenance, retrieval and crawler regression tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from document_rag.documents import article_documents, documents_from_pages
from document_rag.retrieval import answer, ingest, open_index, retrieve

ROOT = Path(__file__).resolve().parents[1]


def test_inline_article_reference_is_not_a_boundary_and_pages_are_preserved():
    docs = article_documents(
        [
            "제1조(목적) 규정입니다. 제2조를 참조합니다.",
            "이전 조항의 계속 문장입니다.\n제2조(적용) 다음 규정입니다.",
        ],
        "fixture.pdf",
    )
    assert len(docs) == 2
    assert docs[0].metadata["article"] == "제1조"
    assert docs[0].metadata["pages"] == "1,2"
    assert "제2조를 참조" in docs[0].page_content


def test_subarticle_header_and_long_article_parts():
    docs = article_documents(
        ["제8조의2(조건) " + "내용 " * 200], "fixture.pdf", size=300
    )
    assert len(docs) > 1
    assert {d.metadata["article"] for d in docs} == {"제8조의2"}
    assert all(d.metadata["pages"] == "1" for d in docs)


def test_company_category_and_useful_contact_text_are_not_lost():
    page = {
        "url": "https://example.org",
        "category": "회사",
        "text": "회사 목적\n제품문의\n연락 안내\nCOPYRIGHT 2020\nfooter",
    }
    docs = documents_from_pages([page, page])
    assert len(docs) == 1
    assert "회사 목적" in docs[0].page_content and "연락 안내" in docs[0].page_content
    assert "COPYRIGHT" not in docs[0].page_content
    assert docs[0].metadata["category"] == "회사"


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        self.calls = getattr(self, "calls", 0) + 1
        return [[float(len(t)), float(sum(map(ord, t)) % 13), 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_actual_chroma_upsert_is_idempotent_and_model_mismatch_fails(tmp_path):
    database = open_index(
        tmp_path, "fixture", model="fixture-model", embedding_function=TinyEmbeddings()
    )
    docs = [
        Document(
            page_content="controlled test",
            metadata={"source": "fixture", "document_sha256": "v1"},
        )
    ]
    assert ingest(database, docs) == 1
    assert ingest(database, docs) == 1
    assert database._embedding_function.calls == 1
    with pytest.raises(ValueError, match="mismatch"):
        open_index(
            tmp_path,
            "fixture",
            model="other-model",
            embedding_function=TinyEmbeddings(),
        )
    changed = [
        Document(
            page_content="changed",
            metadata={"source": "fixture", "document_sha256": "v2"},
        )
    ]
    with pytest.raises(ValueError, match="Source changed"):
        ingest(database, changed)


def test_weak_evidence_is_not_forced_back_in():
    database = SimpleNamespace(
        similarity_search_with_score=lambda *a, **k: [
            (Document(page_content="weak", metadata={"source": "s"}), 1.5)
        ]
    )
    assert retrieve(database, "질문") == []
    assert answer(database, "질문", generate=False)["generation_mode"] == "abstain"


def test_explicit_article_returns_every_part_and_no_similarity_score():
    metadata = [
        {"source": "fixture", "article": "제8조", "part": i, "pages": "1"}
        for i in range(9)
    ]
    database = SimpleNamespace(
        get=lambda **kwargs: {
            "documents": [str(i) for i in range(9)],
            "metadatas": metadata,
        }
    )
    pairs = retrieve(database, "제8조 전체 내용을 확인", k=1)
    assert len(pairs) == 9 and all(distance is None for _, distance in pairs)


def test_generation_failure_and_context_budget_are_explicit():
    doc = Document(page_content="증거", metadata={"source": "fixture"})
    database = SimpleNamespace(
        similarity_search_with_score=lambda *a, **k: [(doc, 0.1)]
    )

    class Broken:
        def invoke(self, prompt):
            raise ConnectionError("private")

    result = answer(database, "질문", llm=Broken())
    assert result["generation_mode"] == "extractive_fallback"
    assert result["generation_error"] == "ConnectionError"
    assert "private" not in result["answer"]


def load_crawler():
    spec = importlib.util.spec_from_file_location(
        "crawler_under_test", ROOT / "crawl/crawl_playwright.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crawler_retries_server_error_and_records_source(monkeypatch):
    crawler = load_crawler()
    monkeypatch.setattr(crawler.time, "sleep", lambda _: None)

    class Page:
        url = "https://example.org/final"

        def __init__(self):
            self.calls = 0

        def goto(self, *a, **kwargs):
            self.calls += 1
            return SimpleNamespace(status=503 if self.calls == 1 else 200)

        def locator(self, selector):
            return SimpleNamespace(inner_text=lambda **kwargs: "source text " * 10)

        def title(self):
            return "Fixture"

    page = Page()
    result = crawler.extract_page(page, "https://example.org", "fixture")
    assert result["status"] == "ok" and result["attempts"] == 2
    assert result["selector"] == "#contents" and not result["needs_review"]
    assert result["sha256"] and result["retrieved_at"]


def test_crawler_does_not_retry_404():
    crawler = load_crawler()

    class Page:
        def goto(self, *args, **kwargs):
            return SimpleNamespace(status=404)

    result = crawler.extract_page(Page(), "https://example.org/missing", "fixture")
    assert result["status"] == "http_error" and result["attempts"] == 1


def test_semantic_seed_expands_whole_article_not_only_last_paragraph():
    metadata = [
        {"source": "fixture", "article": "제8조", "part": i, "pages": "1"}
        for i in range(2)
    ]
    seed = Document(page_content="뒷부분", metadata=metadata[1])
    database = SimpleNamespace(
        similarity_search_with_score=lambda *a, **k: [(seed, 0.2)],
        get=lambda **kwargs: {
            "documents": ["제8조(의무) 앞부분", "뒷부분"],
            "metadatas": metadata,
        },
    )
    results = retrieve(database, "복무 의무를 설명", k=1)
    assert len(results) == 2 and "앞부분" in results[0][0].page_content


def test_company_scope_guard_skips_unrelated_search(monkeypatch):
    spec = importlib.util.spec_from_file_location("company_under_test", ROOT / "rag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "load_vectordb",
        lambda: pytest.fail("Unrelated question must not search"),
    )
    result = module.answer("오늘 서울 날씨 알려줘", generate=False)
    assert result["generation_mode"] == "abstain" and result["sources"] == []


def test_legacy_l2_index_is_not_silently_reused(tmp_path):
    from langchain_chroma import Chroma

    old = Chroma(
        collection_name="old_fixture",
        persist_directory=str(tmp_path),
        embedding_function=TinyEmbeddings(),
    )
    old.add_documents([Document(page_content="legacy", metadata={"source": "fixture"})])
    with pytest.raises(ValueError, match="Legacy index"):
        open_index(
            tmp_path,
            "old_fixture",
            model="fixture-model",
            embedding_function=TinyEmbeddings(),
        )


def test_multiple_explicit_articles_are_all_returned():
    def get(**kwargs):
        assert kwargs["where"]["article"]["$in"] == ["제8조", "제9조"]
        return {
            "documents": ["eight", "nine"],
            "metadatas": [
                {"article": "제8조", "part": 0},
                {"article": "제9조", "part": 0},
            ],
        }

    result = retrieve(SimpleNamespace(get=get), "제8조와 제9조 비교", k=1)
    assert len(result) == 2


def test_truncated_generated_answer_is_explicit():
    response = SimpleNamespace(
        content="unfinished", response_metadata={"done_reason": "length"}
    )
    llm = SimpleNamespace(invoke=lambda _: response)
    evidence = [
        (Document(page_content="evidence", metadata={"source": "fixture"}), None)
    ]
    result = answer(None, "question", llm=llm, evidence=evidence)
    assert result["generation_mode"] == "generated_truncated"
    assert "중단" in result["answer"]


@pytest.mark.parametrize("claim", ["claim [2]", "claim [1, 2]"])
def test_nonexistent_citation_returns_evidence_instead_of_generated_claim(claim):
    llm = SimpleNamespace(
        invoke=lambda _: SimpleNamespace(content=claim, response_metadata={})
    )
    evidence = [
        (Document(page_content="source evidence", metadata={"source": "fixture"}), None)
    ]
    result = answer(None, "question", llm=llm, evidence=evidence)
    assert result["generation_mode"] == "extractive_fallback"
    assert result["generation_error"] == "InvalidCitation"
    assert "source evidence" in result["answer"]
    assert claim not in result["answer"]
