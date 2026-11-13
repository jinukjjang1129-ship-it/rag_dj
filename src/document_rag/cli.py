"""Ingest private PDF/JSON or query a local-only index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .documents import documents_from_pages, documents_from_pdf
from .retrieval import answer, ingest, open_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--no-generation", action="store_true")
    args = parser.parse_args()
    database = open_index(args.index, args.collection)
    if args.input:
        docs = (
            documents_from_pdf(args.input)
            if args.input.suffix.lower() == ".pdf"
            else documents_from_pages(json.loads(args.input.read_text()))
        )
        print(json.dumps({"chunks": len(docs), "index_count": ingest(database, docs)}))
    if args.question:
        print(
            json.dumps(
                answer(database, args.question, generate=not args.no_generation),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
