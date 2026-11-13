"""Pure cleaning and source-span-preserving document construction."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

ARTICLE_HEADER = re.compile(
    r"(?m)^[ \t]*(제\s*\d+\s*조(?:의\s*\d+)?)[ \t]*(?:\([^\n)]{1,100}\)|(?=\n|$))"
)


def clean_web_text(text: str) -> str:
    """Remove explicit footer lines without guessing where useful content starts."""
    lines = []
    for raw in (text or "").splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if re.match(r"^(?:COPYRIGHT|개인정보처리방침\s*$)", line, flags=re.IGNORECASE):
            break
        if line:
            lines.append(line)
    return "\n".join(lines)


def _parts(text: str, size: int = 1200, overlap: int = 150):
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("Require 0 <= overlap < size")
    for start in range(0, len(text), size - overlap):
        yield start, min(len(text), start + size), text[start : start + size]
        if start + size >= len(text):
            break


def documents_from_pages(pages: list[dict]) -> list[Document]:
    """Preserve crawler-assigned categories and exact source URLs."""
    documents, seen = [], set()
    for page in pages:
        text = clean_web_text(page.get("text", ""))
        source = page.get("url", "")
        if not source or not text:
            continue
        identity = (source, hashlib.sha256(text.encode()).hexdigest())
        if identity in seen:
            continue
        seen.add(identity)
        for part, (_, _, chunk) in enumerate(_parts(text, 900, 120)):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "title": page.get("title", ""),
                        "category": page.get("category", "") or "기타",
                        "part": part,
                        "document_sha256": identity[1],
                        "retrieved_at": page.get("retrieved_at", "unrecorded"),
                    },
                )
            )
    return documents


def article_documents(
    pages: list[str], source: str, size: int = 1200
) -> list[Document]:
    """Split only heading lines, keeping every paragraph and actual PDF page span."""
    text = "\n".join(pages)
    page_spans = []
    offset = 0
    for number, page in enumerate(pages, 1):
        page_spans.append((offset, offset + len(page), number))
        offset += len(page) + 1
    matches = list(ARTICLE_HEADER.finditer(text))
    boundaries = [(0, "기타")] if not matches or matches[0].start() > 0 else []
    boundaries.extend((m.start(), re.sub(r"\s+", "", m.group(1))) for m in matches)
    documents = []
    for i, (start, article) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        content = text[start:end]
        for part, (local_start, local_end, chunk) in enumerate(
            _parts(content, size, min(150, size // 4))
        ):
            if not chunk.strip():
                continue
            actual_pages = [
                number
                for a, b, number in page_spans
                if a < start + local_end and b > start + local_start
            ]
            documents.append(
                Document(
                    page_content=chunk.strip(),
                    metadata={
                        "source": source,
                        "article": article,
                        "part": part,
                        "pages": ",".join(map(str, actual_pages)),
                        "title": article,
                    },
                )
            )
    return documents


def documents_from_pdf(path: Path) -> list[Document]:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    documents = article_documents(pages, path.name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for doc in documents:
        doc.metadata["document_sha256"] = digest
    return documents


def document_id(document: Document) -> str:
    metadata = document.metadata
    key = "|".join(
        str(metadata.get(k, "")) for k in ["source", "article", "part", "pages"]
    )
    return hashlib.sha256((key + "|" + document.page_content).encode()).hexdigest()
