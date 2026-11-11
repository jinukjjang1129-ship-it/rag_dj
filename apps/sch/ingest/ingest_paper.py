"""ingest/ingest_paper.py — data/paper_pdfs/ 안의 논문(집) PDF들을"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DOMAIN_CFG = config.DOMAINS["paper"]


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.LOCAL_EMBED_MODEL)


def clean_space(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


_TITLE_EXCLUDE_PATTERNS = re.compile(
    r"issn|doi|vol\.|저널|journal|https?://|^\d{4}[년.\s]"  # issn->고유번호,doi~
    r"|^(?:abstract|keyword|중심어)",
    re.IGNORECASE,
)


def guess_title(first_page_text: str, fallback: str) -> str:
    """첫 페이지에서 논문 제목처럼 보이는 줄을 찾는다.

    [학술지 서지정보 줄을 걸러내는 이유]
    실제 논문(특히 OCR로 처리한 것)의 첫 페이지 맨 위에는 보통 논문 제목보다
    "Smart Media Journal / Vol.12, No.2 / ISSN:2287-1322", "https://dx.doi.org/..."
    같은 학술지 서지정보 줄이 먼저 나온다. 예전엔 "6자 이상 80자 이하인 첫
    줄"을 그냥 제목으로 썼는데, 그러면 이런 서지정보 줄이 걸려서 실제
    논문 제목("머신러닝 기반 클라우드 웹 애플리케이션 HTTP DoS 공격 탐지")
    대신 "...스마트미디어저널 ... ISSN:2287-1322" 같은 게 제목으로 잘못
    뽑혔다. 그래서 ISSN/DOI/Vol./URL/연도로 시작하는 줄, 초록·키워드
    표시줄은 후보에서 제외하고 그 다음 줄을 제목으로 쓴다.
    """
    lines = [clean_space(ln) for ln in (first_page_text or "").splitlines()]
    candidates = [
        ln
        for ln in lines
        if 6 <= len(ln) <= 80 and not _TITLE_EXCLUDE_PATTERNS.search(ln)
    ]
    return candidates[0] if candidates else fallback


_KEYWORD_LINE_PATTERN = re.compile(
    r"(?:중심어|핵심어|주제어|keywords?)\s*[:：]\s*(.+)", re.IGNORECASE
)


def extract_keywords(first_pages_text: str) -> List[str]:
    """논문이 직접 써둔 '중심어/핵심어/주제어/Keywords' 줄을 찾아 키워드 목록으로 파싱.

    [왜 필요한가]
    department.py는 사람이 미리 적어둔 학과 별칭 사전으로 질문을 분류하는데,
    논문은 학과처럼 5개로 고정된 목록이 아니라 주제가 무한히 다양해서 그런
    사전을 미리 만들어둘 수 없다. 대신 논문 저자가 초록 끝에 직접 써놓은
    키워드 줄("@ 중심어 : AI 반도체 ; 데이터센터 ; 스마트폰 ...")을 ingest
    시점에 자동으로 뽑아서, 이걸 학과 별칭 사전 대신 쓴다.
    """
    keywords: List[str] = []
    for line in (first_pages_text or "").splitlines():
        m = _KEYWORD_LINE_PATTERN.search(line)
        if not m:
            continue
        for part in re.split(r"[;,·ㆍ]", m.group(1)):
            cleaned = clean_space(part)
            cleaned = re.sub(r"^[@#*\-\s]+", "", cleaned)  # 앞에 붙는 "@", "•" 등 제거
            if 1 <= len(cleaned) <= 30:
                keywords.append(cleaned)

    seen = set()
    unique = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def split_page(text: str, max_chars: int = 1200, overlap_lines: int = 2) -> List[str]:
    lines = [ln for ln in text.splitlines() if clean_space(ln)]
    if not lines:
        return []
    chunks, buffer, size = [], [], 0
    for line in lines:
        addition = len(line) + 1
        if buffer and size + addition > max_chars:
            chunks.append("\n".join(buffer))
            buffer = buffer[-overlap_lines:] if overlap_lines else []
            size = sum(len(x) + 1 for x in buffer)
        buffer.append(line)
        size += addition
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


# ── 텍스트 추출 ──


def extract_page_texts_normal(pdf_path: Path) -> List[str]:
    """pypdf로 페이지별 텍스트를 뽑는다. 실패한 페이지는 빈 문자열로 채운다."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        # Narrative papers do not need admission-table layout reconstruction.
        raw = page.extract_text() or ""
        texts.append(raw)
    return texts


def needs_ocr(page_texts: List[str]) -> bool:
    """쪽당 평균 글자 수가 임계값 미만이면 '텍스트가 사실상 없는 문서'로 판단."""
    if not page_texts:
        return False
    avg = sum(len(t) for t in page_texts) / len(page_texts)
    return avg < config.OCR_MIN_AVG_CHARS_PER_PAGE


def extract_page_texts_ocr(pdf_path: Path) -> List[str]:
    """각 페이지를 이미지로 렌더링해서 OCR로 텍스트를 뽑는다.

    쪽당 대략 15~20초 걸리는 느린 작업이라, 진단(needs_ocr)에서 OCR이
    필요하다고 판단된 문서에만 실행한다. 진행 상황을 페이지마다 출력해서
    멈춘 것처럼 보이지 않게 한다.

    [PATH 문제 우회] Windows에서 conda로 poppler/tesseract를 설치해도
    터미널이 새 PATH를 못 읽어서 "poppler installed and in PATH?" 에러가
    자주 난다. config.POPPLER_PATH / config.TESSERACT_CMD가 설정돼 있으면
    PATH 탐색 대신 그 경로를 직접 사용한다.

    [TESSDATA_DIR을 환경변수로 넘기는 이유] 처음엔 `--tessdata-dir "경로"`
    처럼 명령줄 옵션 문자열로 만들어 넘겼는데, Windows 경로(백슬래시 포함)가
    공백 기준으로 인자를 쪼개는 과정에서 따옴표가 안 벗겨지고 파일 경로
    안에 그대로 껴버리는 문제가 실제로 있었다("...\\tessdata"/kor.traineddata
    처럼 따옴표가 파일명에 섞여 들어가 파일을 못 찾음). 명령줄 문자열
    조립 대신 os.environ으로 TESSDATA_PREFIX를 직접 설정하면, 이 값이
    따옴표/공백 처리 없이 그대로 하위 프로세스(tesseract)에 전달되어
    이 문제 자체가 생기지 않는다.
    """
    import pytesseract
    from pdf2image import convert_from_path, pdfinfo_from_path

    if config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    if config.TESSDATA_DIR:
        os.environ["TESSDATA_PREFIX"] = config.TESSDATA_DIR

    convert_kwargs = {"dpi": config.OCR_DPI}
    if config.POPPLER_PATH:
        convert_kwargs["poppler_path"] = config.POPPLER_PATH

    available = set(pytesseract.get_languages())
    missing = set(config.OCR_LANG.split("+")) - available
    if missing:
        raise RuntimeError(f"Missing OCR language packs: {sorted(missing)}")
    info = pdfinfo_from_path(str(pdf_path), poppler_path=config.POPPLER_PATH)
    texts = []
    for i in range(1, info["Pages"] + 1):
        image = convert_from_path(
            str(pdf_path), first_page=i, last_page=i, **convert_kwargs
        )[0]
        print(f"    OCR 중... {i}/{info['Pages']}쪽", end="\r", flush=True)
        text = pytesseract.image_to_string(image, lang=config.OCR_LANG)
        texts.append(text)
        image.close()
    print()  # 진행 표시 줄바꿈
    return texts


def get_page_texts(pdf_path: Path) -> tuple[List[str], str]:
    """이 PDF에 맞는 추출 방식을 자동 판별해서 페이지별 텍스트를 반환.

    반환: (페이지별 텍스트 리스트, 사용한 방식 — "text" 또는 "ocr")
    """
    normal_texts = extract_page_texts_normal(pdf_path)

    if not config.OCR_ENABLED:
        return normal_texts, "text"

    if needs_ocr(normal_texts):
        avg = sum(len(t) for t in normal_texts) / max(len(normal_texts), 1)
        print(f"  [진단] 쪽당 평균 {avg:.0f}자 -> 텍스트가 사실상 없음, OCR로 전환")
        try:
            ocr_texts = extract_page_texts_ocr(pdf_path)  # ocr 함수 실행
            return ocr_texts, "ocr"
        except Exception as exc:
            print(
                f"  [OCR 실패] {exc} -> 원래 텍스트로 계속 진행 (내용이 부실할 수 있음)"
            )
            return normal_texts, "text_ocr_failed"

    return normal_texts, "text"


def build_documents_for_pdf(
    pdf_path: Path,
) -> tuple[List[Document], List[str], str, List[str]]:
    page_texts, method = get_page_texts(pdf_path)

    documents, ids = [], []
    paper_title = pdf_path.stem
    # 키워드는 보통 첫 페이지(국문 초록) 또는 두 번째 페이지(영문 Keywords)에
    # 걸쳐 나오는 경우가 많아서, 앞 2쪽을 합쳐서 찾는다.
    paper_keywords = extract_keywords("\n".join(page_texts[:2]))
    keywords_joined = "; ".join(paper_keywords)

    for pdf_index, raw_text in enumerate(page_texts):
        pdf_page = pdf_index + 1
        page_text = clean_space(raw_text) and raw_text.strip()
        if not page_text:
            continue

        if pdf_index == 0:
            paper_title = guess_title(raw_text, pdf_path.stem)

        for chunk_no, chunk_text in enumerate(split_page(raw_text), start=1):
            embedding_text = "\n".join(
                [
                    f"논문 제목(추정): {paper_title}",
                    f"키워드: {keywords_joined}" if keywords_joined else "",
                    f"파일: {pdf_path.name}",
                    f"내용:\n{chunk_text}",
                ]
            )
            identity = f"paper|{pdf_path.name}|{pdf_page}|{chunk_no}|{chunk_text}"
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
            documents.append(
                Document(
                    page_content=embedding_text,
                    metadata={
                        "document_type": "paper",
                        "paper_title": paper_title,
                        "paper_keywords": keywords_joined,  # rag/paper.py의 스코프 필터링에 씀
                        "title": f"{paper_title} ({pdf_path.name} {pdf_page}쪽)",
                        "source": pdf_path.name,
                        "pdf_page": pdf_page,
                        "chunk_no": chunk_no,
                        "extraction_method": method,  # "text" | "ocr" — 나중에 품질 이슈 추적용
                        "raw_text": chunk_text,
                    },
                )
            )
            ids.append(chunk_id)

    return documents, ids, method, paper_keywords


def main():
    reset = "--reset" in sys.argv
    config.PAPER_PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(config.PAPER_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"[중단] {config.PAPER_PDF_DIR} 에 논문 PDF가 없습니다.")
        return

    if reset and Path(DOMAIN_CFG["persist_dir"]).exists():
        shutil.rmtree(DOMAIN_CFG["persist_dir"])
        print(f"기존 DB 삭제: {DOMAIN_CFG['persist_dir']}")

    vectordb = Chroma(
        collection_name=DOMAIN_CFG["collection"],
        persist_directory=DOMAIN_CFG["persist_dir"],
        embedding_function=get_embeddings(),
        collection_metadata={
            "hnsw:space": "cosine",
            "embedding_model": config.LOCAL_EMBED_MODEL,
        },
    )

    total = 0
    ocr_used_files = []
    no_keyword_files = []
    for pdf_path in pdf_files:
        print(f"\n처리 중: {pdf_path.name}")
        docs, ids, method, paper_keywords = build_documents_for_pdf(pdf_path)
        if docs:
            vectordb.add_documents(docs, ids=ids)
        total += len(docs)
        if method == "ocr":
            ocr_used_files.append(pdf_path.name)
        if paper_keywords:
            print(f"  -> {len(docs)}개 청크 적재 (방식: {method})")
            print(f"     키워드: {', '.join(paper_keywords)}")
        else:
            no_keyword_files.append(pdf_path.name)
            print(
                f"  -> {len(docs)}개 청크 적재 (방식: {method}) — [경고] 키워드를 못 찾음"
            )

    print(f"\n완료. 총 {total}개 청크 적재됨 -> {DOMAIN_CFG['persist_dir']}")
    if no_keyword_files:
        print(
            f"\n[안내] 키워드(중심어/핵심어/Keywords)를 못 찾은 파일 ({len(no_keyword_files)}개):"
        )
        print("  이 논문들은 rag/paper.py의 키워드 매칭 대상에서 빠지므로,")
        print("  질문이 이 논문과 관련 있어도 스코프 게이트에서 걸릴 수 있습니다.")
        for name in no_keyword_files:
            print(f"  - {name}")
    if ocr_used_files:
        print(
            f"\n[안내] OCR로 처리된 파일 ({len(ocr_used_files)}개) — 텍스트 품질이 완벽하지 않을 수 있습니다:"
        )
        for name in ocr_used_files:
            print(f"  - {name}")
    print("\n이제 'streamlit run app_streamlit.py' 로 실행하세요.")


if __name__ == "__main__":
    main()
