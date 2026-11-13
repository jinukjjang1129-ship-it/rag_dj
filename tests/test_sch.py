"""Small dependency-isolated tests for retrieval and deterministic enumeration."""

from types import SimpleNamespace

import pytest
from common import config
from langchain_core.documents import Document

from rag import department, engine


class VectorStub:
    def __init__(self, pairs=()):
        self.pairs = list(pairs)
        self.calls = []

    def similarity_search_with_score(self, question, **kwargs):
        self.calls.append(kwargs)
        return self.pairs


def test_filtered_search_does_not_escape_scope(monkeypatch):
    db = VectorStub()
    monkeypatch.setattr(engine, "get_vectordb", lambda _: db)
    results, fallback = engine.search(
        "department", "학과", where={"department": "없는학과"}
    )
    assert results == [] and not fallback
    assert len(db.calls) == 1 and db.calls[0]["filter"] == {"department": "없는학과"}


def test_unfiltered_search_runs_once_and_deduplicates(monkeypatch):
    a = Document(page_content="같은  내용", metadata={"title": "문서"})
    b = Document(page_content="같은 내용", metadata={"title": "복사"})
    db = VectorStub([(a, 0.1), (b, 0.2)])
    monkeypatch.setattr(engine, "get_vectordb", lambda _: db)
    results, _ = engine.search("department", "학과")
    assert len(results) == 1 and len(db.calls) == 1
    assert results[0]["score_type"] == "cosine_similarity_heuristic"


@pytest.mark.parametrize(
    "distance, expected", [(-0.01, 100), (0, 100), (0.5, 50), (2, 0)]
)
def test_distance_display_is_bounded(distance, expected):
    assert engine._to_relevance(distance) == expected


def test_metadata_list_has_no_top_k_truncation_or_false_professor(monkeypatch):
    metas = [
        {
            "title": f"테스트{i} 교수",
            "raw_text": "교수 소개",
            "url": "https://example.org",
        }
        for i in range(12)
    ] + [{"title": "오시는길", "raw_text": "주소"}]

    class MetadataStub:
        def get(self, **kwargs):
            assert "where" in kwargs
            return {"metadatas": metas, "documents": [m["raw_text"] for m in metas]}

    monkeypatch.setattr(engine, "get_vectordb", lambda _: MetadataStub())
    monkeypatch.setattr(
        engine,
        "make_llm_answer",
        lambda *a, **k: pytest.fail("List must not invoke LLM"),
    )
    result = department.answer("컴공 교수님은 누가 있나요?", top_k=1)
    assert len(result["sources"]) == 12
    assert "오시는길" not in result["answer"]
    assert result["route"] == "metadata_list"
    assert result["timing"]["generate_sec"] == 0


def test_missing_faculty_abstains(monkeypatch):
    db = SimpleNamespace(get=lambda **kwargs: {"metadatas": [], "documents": []})
    monkeypatch.setattr(engine, "get_vectordb", lambda _: db)
    result = department.answer("AI빅데이터학과 교수 전체 목록")
    assert not result["in_scope"] and result["sources"] == []


def test_generation_fallback_is_explicit(monkeypatch):
    monkeypatch.setattr(config, "USE_OLLAMA", True)

    class Broken:
        def invoke(self, prompt):
            raise ConnectionError("do not expose this private error")

    monkeypatch.setattr(engine, "_get_llm", lambda: Broken())
    answer, mode = engine.generate_answer(
        "안내", "질문", [{"rank": 1, "title": "제목", "score": 50, "text": "근거"}]
    )
    assert mode == "extractive_fallback" and "private" not in answer


def test_graph_listing_does_not_invoke_llm(monkeypatch):

    from graph_demo import app

    rows = [{"id": str(i), "name": f"테스트전형{i}", "count": 10} for i in range(9)]

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, *args, **kwargs):
            return rows

    monkeypatch.setattr(app, "driver", SimpleNamespace(session=Session))
    monkeypatch.setattr(
        app.admission_rag,
        "detect_track_filter",
        lambda _: ({"track_category": "학생부교과"}, "교과"),
    )
    monkeypatch.setattr(
        engine,
        "make_llm_answer",
        lambda *a, **k: pytest.fail("No LLM for lists"),
    )
    result = app.query(
        app.QueryRequest(question="학생부교과에는 어떤 전형 목록이 있나요?")
    )
    assert result.mode == "graph" and len(result.used_nodes) == 9
    assert all(r["name"] in result.answer for r in rows)


def test_pdf_zero_overlap_does_not_repeat_previous_chunk():
    from ingest.ingest_admission import split_pdf_page

    chunks = split_pdf_page("alpha\nbeta\ngamma\ndelta", max_chars=12, overlap_lines=0)
    assert sum(c.count("alpha") for c in chunks) == 1
    assert all(len(c) <= 12 for c in chunks)


def test_headcount_preserves_thousands_separator():
    from ingest.build_admission_graph import extract_headcount

    assert extract_headcount("1,234명") == 1234
    assert extract_headcount("정보 없음") is None


def test_track_continuation_not_reclassified_by_cross_reference():
    from ingest.ingest_admission import detect_section_header

    assert (
        detect_section_header(
            "3. 전형방법 | : 일괄합산\n- 학생부종합전형 서류평가 방법 p.57 참조"
        )
        is None
    )


def test_common_eligibility_and_combined_table_remain_separate():
    from ingest.build_admission_graph import parse_track_fields

    fields = parse_track_fields(
        "1. 공통 지원자격 공통조건 2. 전형별 모집인원 및 지원자격 표 4. 전형방법 단계별"
    )
    assert fields["지원자격"] == "공통조건"
    assert fields["전형별모집인원및지원자격"] == "표"
    assert "모집인원" not in fields


def test_missing_korean_ocr_pack_fails_explicitly(monkeypatch, tmp_path):
    import pytesseract

    from ingest import ingest_paper

    monkeypatch.setattr(pytesseract, "get_languages", lambda: ["eng"])
    monkeypatch.setattr(config, "OCR_LANG", "kor+eng")
    with pytest.raises(RuntimeError, match="kor"):
        ingest_paper.extract_page_texts_ocr(tmp_path / "missing.pdf")


def test_ocr_failure_is_not_reported_as_clean_text(monkeypatch, tmp_path):
    from ingest import ingest_paper

    monkeypatch.setattr(config, "OCR_ENABLED", True)
    monkeypatch.setattr(ingest_paper, "extract_page_texts_normal", lambda _: ["x"])

    def fail(_):
        raise RuntimeError("OCR unavailable")

    monkeypatch.setattr(ingest_paper, "extract_page_texts_ocr", fail)
    texts, method = ingest_paper.get_page_texts(tmp_path / "sample.pdf")
    assert texts == ["x"] and method == "text_ocr_failed"


def test_index_model_mismatch_is_rejected_and_legacy_is_unverified():
    from common.index import validate_embedding_model

    assert not validate_embedding_model(None, "model-a")
    assert validate_embedding_model({"embedding_model": "model-a"}, "model-a")
    with pytest.raises(ValueError, match="differs"):
        validate_embedding_model({"embedding_model": "model-b"}, "model-a")


def test_incompatible_index_is_never_cached(monkeypatch):
    monkeypatch.setattr(engine, "_vectordb_cache", {})
    monkeypatch.setattr(engine, "_get_embeddings", lambda: None)
    monkeypatch.setattr(
        engine,
        "Chroma",
        lambda **kwargs: SimpleNamespace(
            _collection=SimpleNamespace(metadata={"embedding_model": "wrong"})
        ),
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="differs"):
            engine.get_vectordb("department")
    assert engine._vectordb_cache == {}


def test_paper_filename_recovers_topic_when_title_is_bad():
    from rag.paper import detect_matching_papers

    source = "코드 생성 AI를 이용한 웹 취약점 점검 방법에 관한 연구.pdf"
    mapping = {source: {"title": "Journal Volume", "keywords": ["Vulnerability"]}}
    assert detect_matching_papers("웹 취약점 관련 논문", mapping) == [source]
    assert detect_matching_papers("양자 중력 연구", mapping) == []


def test_track_fee_question_uses_common_fee_table(monkeypatch):
    from rag import admission

    captured = {}

    def answer_stub(*args, **kwargs):
        captured.update(kwargs)
        return {"answer": "fixture", "sources": []}

    monkeypatch.setattr(engine, "answer_question", answer_stub)
    result = admission.answer("SW융합전형 전형료는 얼마인가요?")
    assert captured["where"] == {"section": "전형료"}
    assert result["detected_sections"] == ["전형료"]


def test_truncated_model_output_is_not_marked_complete(monkeypatch):
    monkeypatch.setattr(config, "USE_OLLAMA", True)
    response = SimpleNamespace(
        content="unfinished", response_metadata={"done_reason": "length"}
    )
    monkeypatch.setattr(
        engine, "_get_llm", lambda: SimpleNamespace(invoke=lambda _: response)
    )
    text, mode = engine.generate_answer(
        "안내", "질문", [{"rank": 1, "title": "제목", "text": "근거"}]
    )
    assert mode == "ollama_truncated"
    assert "중단" in text


def test_paper_keyword_list_preserves_metadata_without_generation(monkeypatch):
    from rag import paper

    monkeypatch.setattr(
        paper,
        "_load_paper_keyword_map",
        lambda: {
            "HTTP study.pdf": {
                "title": "HTTP study",
                "keywords": ["HTTP", "Cloud", "HTTP"],
            }
        },
    )
    monkeypatch.setattr(
        engine, "answer_question", lambda *a, **kw: pytest.fail("no generation")
    )
    result = paper.answer("HTTP 논문 키워드를 알려주세요")
    assert result["generation_mode"] == "deterministic_metadata"
    assert result["sources"][0]["text"] == "HTTP; Cloud"
    assert result["sources"][0]["metadata"]["source"] == "HTTP study.pdf"
