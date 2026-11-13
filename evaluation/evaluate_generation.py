"""Compare actual local generation on the same questions and source indexes.

Run sequentially, in fresh processes. Outputs include private evidence; keep
them outside Git. Legacy PDF code receives indentation repair only. Inspect
answers manually; successful model invocation is not factual correctness.
"""

import argparse
import json
import sys
import textwrap
import time
import types
from pathlib import Path

from langchain_ollama import ChatOllama

parser = argparse.ArgumentParser(
    description="Private live-generation audit; output contains source text and must not be published"
)
parser.add_argument("version", choices=["legacy", "current"])
parser.add_argument("group", choices=["sch", "documents"])
parser.add_argument("--originals", type=Path, required=True)
parser.add_argument("--sch-baseline", type=Path, required=True)
parser.add_argument("--runtime", type=Path, required=True)
parser.add_argument("--private-output", type=Path, required=True)
parser.add_argument("--case", help="Run a single case after a targeted correction")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = args.private_output.resolve()
if output.is_relative_to(root) and not output.is_relative_to(root / "results/private"):
    parser.error("Private answers must be outside the repository or in results/private")
version = args.version
group = args.group
runtime = args.runtime.resolve()
calls = []


class RecordedOllama(ChatOllama):
    def invoke(self, *args, **kwargs):
        response = super().invoke(*args, **kwargs)
        calls.append(response.response_metadata)
        return response


llm = RecordedOllama(
    model="gemma3:4b",
    temperature=0,
    num_ctx=8192,
    num_predict=192,
    num_thread=2,
    seed=42,
    keep_alive="30m",
    client_kwargs={"timeout": 180},
)


def load(path, name, dedent=False):
    m = types.ModuleType(name)
    m.__file__ = str(path)
    exec(
        compile(
            textwrap.dedent(path.read_text()) if dedent else path.read_text(),
            str(path),
            "exec",
        ),
        m.__dict__,
    )
    return m


if group == "sch":
    sys.path.insert(
        0,
        str(root / "apps/sch")
        if version == "current"
        else str(args.sch_baseline.resolve()),
    )
    from common import config

    from rag import admission, department, engine, paper

    config.USE_OLLAMA = True
    for d in config.DOMAINS:
        config.DOMAINS[d]["persist_dir"] = str(runtime / "sch_rebuilt_verified" / d)
    engine._get_llm = lambda: llm
    cases = [
        ("faculty", department.answer, "컴퓨터소프트웨어공학과 교수님은 누가 있나요?"),
        (
            "application",
            admission.answer,
            "수시모집 원서접수 시작과 마감 날짜 및 시간을 알려주세요.",
        ),
        ("fee", admission.answer, "SW융합전형 전형료는 얼마인가요?"),
        ("paper", paper.answer, "HTTP Dos 관련 논문의 키워드를 알려주세요."),
    ]
else:
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from langchain_chroma import Chroma

    from document_rag.retrieval import answer, embeddings, open_index

    model = embeddings()
    import config

    config.PERSIST_DOCS = str(runtime / "company_validated")
    if version == "current":
        db = open_index(runtime / "employment_validated", "employment")

        def legal(q):
            return answer(db, q, llm=llm, k=3)

        company = load(root / "rag.py", "company")
        company.get_llm = lambda **kw: llm
    else:
        source = args.originals.resolve()
        old = load(next(source.glob("rag_*/rag.py")), "oldlegal", True)
        db = Chroma(
            collection_name="legacy_employment",
            persist_directory=str(runtime / "document_comparison/pdf_legacy"),
            embedding_function=model,
        )
        old.load_vectordb = lambda: db
        old.get_llm = lambda **kw: llm
        legal = old.answer
        company = load(source / "dj_rag_local/rag.py", "oldcompany")
        cdb = Chroma(
            collection_name="legacy_company",
            persist_directory=str(runtime / "document_comparison/company_legacy"),
            embedding_function=model,
        )
        company.load_vectordb = lambda: cdb
        company.get_llm = lambda **kw: llm
    cases = [
        ("probation", legal, "제7조 수습기간은 몇 개월인가요?"),
        ("hours", legal, "제23조 성인 근로자의 주당 및 하루 근로시간은 얼마인가요?"),
        ("holiday", legal, "근로자의 날은 몇 월 며칠인가요?"),
        ("company", company.answer, "동진쎄미켐 회사개요에서 설립 연도는 언제인가요?"),
    ]
rows = []
for cid, fn, q in cases:
    if args.case and cid != args.case:
        continue
    calls.clear()
    start = time.perf_counter()
    try:
        r = fn(q)
    except Exception as e:
        r = {"error": type(e).__name__, "detail": str(e)}
    rows.append(
        {
            "id": cid,
            "question": q,
            "seconds": time.perf_counter() - start,
            "result": r,
            "model_calls": list(calls),
        }
    )
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str)
    )
    print(
        cid,
        round(rows[-1]["seconds"], 2),
        r.get("generation_mode", "legacy"),
        r.get("error", ""),
        flush=True,
    )
