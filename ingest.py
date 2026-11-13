"""Prepare company-page documents and upsert a separately versioned index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import (
    COL_DOCS,
    PAGES_JSON,
    PERSIST_DOCS,
    embedding_signature,
    get_embeddings,
)
from document_rag.documents import documents_from_pages
from document_rag.retrieval import ingest, open_index

ROOT = Path(__file__).resolve().parent


def build_documents(path: Path = PAGES_JSON):
    targets = json.loads((ROOT / "configs/crawl_targets.json").read_text())
    categories = {item["url"]: item["category"] for item in targets}
    pages = json.loads(path.read_text())
    enriched = [
        {
            **page,
            "category": page.get("category") or categories.get(page.get("url"), "기타"),
        }
        for page in pages
        if page.get("status", "ok") == "ok"
    ]
    return documents_from_pages(enriched)


def build_database(path: Path = PAGES_JSON) -> dict:
    docs = build_documents(path)
    if not docs:
        raise ValueError("No nonempty successful pages to ingest")
    database = open_index(
        Path(PERSIST_DOCS),
        COL_DOCS,
        model=embedding_signature(),
        embedding_function=get_embeddings(),
    )
    return {"documents": len(docs), "index_count": ingest(database, docs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PAGES_JSON)
    args = parser.parse_args()
    print(json.dumps(build_database(args.input), indent=2))


if __name__ == "__main__":
    main()
