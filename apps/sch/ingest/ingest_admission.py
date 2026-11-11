"""ingest/ingest_admission.py — 입시요강 PDF를 ChromaDB(도메인: admission)에 적재."""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import config
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DOMAIN_CFG = config.DOMAINS["admission"]

# (섹션명, 이 섹션이 시작됐다고 판단할 헤더 키워드들) — 목차 순서와 동일하게 유지.
# admission.py의 get_status()가 이 이름들로 "적재/미적재"를 보여준다.
ADMISSION_SECTION_RANGES = [
    (None, None, "전형요약 및 주요 변경사항"),
    (None, None, "모집단위 및 입학정원"),
    (None, None, "전형별 모집인원"),
    (None, None, "원서접수·전형일정·합격·등록"),
    (None, None, "수시모집 세부 전형별 안내"),
    (None, None, "전형기간 자율화 전형"),
    (None, None, "제출서류"),
    (None, None, "학교생활기록부 반영 방법"),
    (None, None, "서류 및 면접평가"),
    (None, None, "지원자 유의사항·등록 포기·환불"),
    (None, None, "전형료"),
    (None, None, "학사안내·ROTC"),
    (None, None, "신입생 장학금"),
    (None, None, "조기취업형계약학과 모집기업"),
    (None, None, "입학처 연락처"),
]

SECTION_HEADER_KEYWORDS: list[tuple[str, list[str]]] = [
    ("전형요약 및 주요 변경사항", ["전형요약", "주요 변경사항", "주요변경사항"]),
    ("모집단위 및 입학정원", ["모집단위(입학정원)", "모집단위 및 입학정원"]),
    ("전형별 모집인원", ["전형별 모집인원", "정원 내 모집인원", "정원 외 모집인원"]),
    (
        "원서접수·전형일정·합격·등록",
        ["원서접수 및 전형일정", "최초합격자 발표", "충원합격자 발표", "등록금 납부"],
    ),
    (
        "수시모집 세부 전형별 안내",
        ["수시모집 세부 전형별 안내", "학생부교과(", "학생부종합(", "실기/실적("],
    ),
    (
        "전형기간 자율화 전형",
        ["전형기간 자율화 전형 세부", "특성화고졸재직자전형", "평생학습자전형"],
    ),
    ("제출서류", ["제출서류 안내사항", "전형별 제출서류", "지원자 공통제출서류"]),
    (
        "학교생활기록부 반영 방법",
        ["학교생활기록부 반영 방법", "교과 성적 반영 방법", "대학지정교과분류표"],
    ),
    (
        "서류 및 면접평가",
        ["서류 및 면접평가 반영 방법", "학생부종합전형 서류평가", "면접평가 방법"],
    ),
    (
        "지원자 유의사항·등록 포기·환불",
        ["지원자 유의사항", "대학입학전형 기본사항", "등록 포기 및 환불"],
    ),
    ("전형료", ["전형료", "전형료 반환"]),
    ("학사안내·ROTC", ["학사안내", "학생군사교육단", "rotc"]),
    ("신입생 장학금", ["신입생 장학금", "향설장학금", "입학 성적우수 장학금"]),
    ("조기취업형계약학과 모집기업", ["조기취업형계약학과", "참여기업", "모집기업"]),
    ("입학처 연락처", ["입학처 연락처", "입학팀 전화번호"]),
]

ADMISSION_SEARCH_TERMS = {
    "전형요약 및 주요 변경사항": "수시 전형요약 변경사항 전형방법 수능최저 모집인원",
    "모집단위 및 입학정원": "모집단위 학과 단과대학 입학정원 모집인원",
    "전형별 모집인원": "전형별 모집인원 학과별 인원 학생부교과 학생부종합 실기",
    "원서접수·전형일정·합격·등록": "원서접수 접수기간 면접 실기 합격자 발표 등록 충원합격 일정",
    "수시모집 세부 전형별 안내": (
        "지원자격 전형방법 전형요소 수능최저 모집인원 학생부교과 "
        "학생부종합 지역인재 SW융합 실기 특기자 농어촌"
    ),
    "전형기간 자율화 전형": "특성화고졸재직자 평생학습자 지원자격 전형방법",
    "제출서류": "제출서류 증명서 신청서 학교생활기록부 지원자격 서류제출",
    "학교생활기록부 반영 방법": "학생부 반영교과 내신 성적 산출식 등급 점수 검정고시 학교폭력",
    "서류 및 면접평가": "서류평가 면접평가 평가방법 평가요소 배점",
    "지원자 유의사항·등록 포기·환불": "지원 유의사항 복수지원 면접 실기 등록포기 환불",
    "전형료": "전형료 접수비 반환 환불 금액",
    "학사안내·ROTC": "학사안내 전공 학과 ROTC 학생군사교육단",
    "신입생 장학금": "신입생 장학금 향설장학금 성적우수 등록금 지급기준",
    "조기취업형계약학과 모집기업": "조기취업형계약학과 참여기업 모집기업 기업명 모집인원",
    "입학처 연락처": "입학처 입학팀 전화번호 홈페이지 문의",
}

_HEADER_LOOKUP = [
    (name, kw.lower()) for name, keywords in SECTION_HEADER_KEYWORDS for kw in keywords
]
# 긴 키워드부터 검사 (짧은 키워드의 오탐 방지, 예: "학사안내" vs "학사안내·ROTC")
_HEADER_LOOKUP.sort(key=lambda x: len(x[1]), reverse=True)


def _header_zone(page_text: str) -> str:
    """헤더 감지용 판별 사본: 첫 2줄만 뽑고 표 구분자(|)를 공백으로 되돌린다.

    detect_section_header()와 detect_track_header() 둘 다 이 판별 사본을
    쓴다 — "첫 2줄만 본다"와 "|를 다시 공백으로 되돌린다"는 규칙이 두
    함수에 동일하게 적용돼야 하기 때문이다.
    """
    first_lines = "\n".join(page_text.split("\n", 2)[:2])
    return re.sub(r"\s+", " ", first_lines.replace("|", " "))


def detect_section_header(page_text: str) -> Optional[str]:
    """이 페이지 앞부분에서 새 섹션이 시작되는지 감지.

    [주의: 첫 줄만 본다]
    처음엔 앞 200자를 통째로 봤는데, 실제 PDF에서 표지 페이지의 안내문구
    ("...전형기간 자율화 전형[학생부종합(특성화고졸재직자전형)과...")에
    본문 중간의 우연한 키워드 언급 때문에 표지가 엉뚱한 섹션으로 오분류되는
    문제가 있었다. 실제 섹션 헤더는 항상 그 페이지의 "첫 줄"에 온다는 점을
    이용해, 첫 줄만 검사하도록 좁혀서 본문 중간 언급으로 인한 오탐지를 막는다.

    [주의: 표 구분자(|)는 매칭 전에 공백으로 되돌린다]
    clean_pdf_page_text()가 표의 열 경계를 보존하려고 연속 공백을 " | "로
    바꿔두는데, 제목 줄도 글자 사이 간격이 넓게 조판된 경우가 많아 "전형별
    모집인원"이 "전형별 | 모집인원"처럼 쪼개질 수 있다. 그대로 두면
    SECTION_HEADER_KEYWORDS의 "전형별 모집인원"(공백 기준)과 안 맞아서
    헤더 감지가 실패한다. 그래서 헤더 매칭에 한해서만 "|"를 공백으로
    되돌리고 공백을 다시 정리한다 (표 본문 자체의 "|"는 그대로 유지됨 —
    이 함수는 판별용 사본만 만들 뿐 원본 page_text는 건드리지 않는다).
    """
    first_line = (
        re.sub(r"\s+", " ", page_text.splitlines()[0].replace("|", " ")).strip()
        if page_text.strip()
        else ""
    )
    # A continued track page can cite another section in its first two lines.
    if re.match(
        r"\d+\.\s*(모집인원|지원자격|전형방법|동점자|20\d{2}학년도)", first_line
    ):
        return None
    zone = _header_zone(page_text).lower()
    for name, keyword in _HEADER_LOOKUP:
        if keyword in zone:
            return name
    return None


# "▮ 학생부교과(교과우수자전형)" 처럼 세부 전형 하나가 시작되는 줄의 패턴.
# "수시모집 세부 전형별 안내"/"전형기간 자율화 전형" 섹션 안의 페이지들이
# 전형 하나당 한 쪽씩 이렇게 시작된다.
_TRACK_HEADER_PATTERN = re.compile(
    r"[▮■]\s*(학생부교과|학생부종합|실기/실적)\(([^)]+)\)"
)


def detect_track_header(page_text: str) -> tuple[Optional[str], Optional[str]]:
    """이 페이지가 특정 세부 전형(예: 학생부교과(교과우수자전형))으로
    시작하는지 감지.

    반환: (전형 전체명, 전형 갈래) — 둘 다 못 찾으면 (None, None).
    예: ("학생부교과(교과우수자전형)", "학생부교과")

    section보다 한 단계 더 세밀한 분류다. section은 "수시모집 세부
    전형별 안내"처럼 20여 개 전형을 통째로 묶은 큰 범주라서, "학생부교과
    전형 지원자격"처럼 특정 전형을 콕 집은 질문에도 그 범주 전체가
    검색 대상이 돼버려 엉뚱한 전형 내용이 섞여 나오는 문제가 있었다.
    """
    zone = _header_zone(
        page_text
    )  # 전형명은 원래 표기(대소문자) 그대로 써야 하므로 소문자 변환 안 함
    m = _TRACK_HEADER_PATTERN.search(zone)
    if not m:
        return None, None
    category, name = m.group(1), m.group(2)
    return f"{category}({name})", category


_TOC_PAGE_PATTERN = re.compile(r"Ⅰ\s*\d")


def is_toc_or_cover_page(page_text: str) -> bool:
    """목차 페이지 판별.

    이 문서의 목차는 "가. 전형요약 Ⅰ 1" 처럼 로마숫자Ⅰ+쪽번호로 표시된다.
    이런 참조가 한 페이지에 여러 번(3회 이상) 나오면 실제 내용이 아니라
    목차(챕터 제목 나열)로 보고, 검색 노이즈가 되지 않도록 청크 생성에서
    제외한다.
    """
    return len(_TOC_PAGE_PATTERN.findall(page_text)) >= 3


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.LOCAL_EMBED_MODEL)


def clean_space(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


_REPEATED_LINE_PATTERNS = [
    re.compile(r"^더\s*큰.*세상$"),  # 학교 슬로건 ("더 큰 순천향, 더 나은 세상")
    re.compile(r"순천향대학교.*모집요강$"),  # 매 쪽 반복되는 문서 제목
]


def clean_pdf_page_text(text: str) -> str:
    """표 행 구분을 위해 줄바꿈은 유지하고, 반복되는 머리말·꼬리말·쪽번호만 제거.

    [표 열 경계 보존]
    pypdf의 layout 추출은 표의 열 간격을 넓은 공백으로 표현해준다. 그런데
    단순히 "공백 여러 개 → 1개"로 뭉개버리면, 빈 셀이 있는 행에서 값들이
    앞으로 밀려버려 어느 값이 어느 열인지 알 수 없게 된다 (예: "의예과 12 9
    20 10 9 20 3" — 원래 12개 열 중 몇 개는 비어있는데 그 정보가 사라짐).
    그래서 2칸 이상 이어지는 공백은 " | "(열 구분자)로 바꿔서 "여기서 칸이
    나뉜다"는 신호를 남겨두고, 그 다음에 남은 잡다한 공백(탭 등)만 1칸으로
    정리한다. 문장으로 된 일반 텍스트는 보통 연속 공백이 없어 영향이 없다.
    """
    cleaned_lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r" {2,}", " | ", line)  # 열 경계 보존 (먼저 처리)
        line = re.sub(r"\s+", " ", line).strip()  # 남은 공백류(탭 등) 정리
        if re.fullmatch(r"-?\s*\d+\s*-?", line):
            continue
        if any(p.search(line) for p in _REPEATED_LINE_PATTERNS):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def split_pdf_page(
    text: str, max_chars: int = 4000, overlap_lines: int = 3
) -> List[str]:
    """표 줄 단위를 최대한 유지하며 분할.

    [왜 4000자냐면]
    실제 2027 수시모집 요강 70개 유효 페이지를 재보니 평균 1488자, 최댓값
    3494자였다 (표지·목차 제외). 이 문서의 페이지들은 "전형 하나 = 페이지
    하나"로 완결되게 편집돼 있어서, 예전처럼 1400자에서 잘라버리면 절반이
    넘는 페이지(70개 중 40개)가 중간에서 끊겨 정보가 두 청크로 쪼개졌다.
    4000자로 올리면 이번 PDF의 모든 페이지가 청크 1개로 유지되면서도,
    혹시 다음 학년도 PDF에 이보다 훨씬 긴 페이지가 나오면 여전히 분할
    안전장치로 작동한다 (완전히 없애지 않고 여유 있는 상한선만 둔 것).
    """
    lines = [line for line in text.splitlines() if clean_space(line)]
    if not lines:
        return []
    chunks, buffer, current_size = [], [], 0
    for line in lines:
        addition = len(line) + 1
        if buffer and current_size + addition > max_chars:
            chunks.append("\n".join(buffer))
            buffer = buffer[-overlap_lines:] if overlap_lines else []
            current_size = sum(len(item) + 1 for item in buffer)
        buffer.append(line)
        current_size += addition
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


def build_documents_for_pdf(
    pdf_path: Path,
) -> tuple[List[Document], List[str], Counter]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    documents, ids = [], []
    section_counts: Counter = Counter()
    current_section = "입시요강 기타"
    current_track: str = ""
    current_track_category: str = ""

    for pdf_index, page in enumerate(reader.pages):
        pdf_page = pdf_index + 1
        try:
            raw_text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:
            raw_text = page.extract_text() or ""
        page_text = clean_pdf_page_text(raw_text)
        if not page_text:
            continue
        if is_toc_or_cover_page(page_text):
            continue

        detected = detect_section_header(page_text)
        if detected:
            if detected != current_section:
                # 섹션이 바뀌면 이전 전형 정보를 들고 오면 안 된다 (예: "학생부
                # 교과(교과우수자전형)" 다음 페이지가 "제출서류" 섹션으로
                # 넘어갔는데 track만 이전 값 그대로 남아있으면 안 됨).
                current_track, current_track_category = "", ""
            current_section = detected

        track_name, track_category = detect_track_header(page_text)
        if track_name:
            current_track, current_track_category = track_name, track_category

        search_terms = ADMISSION_SEARCH_TERMS.get(
            current_section, "순천향대학교 수시모집 입시 모집요강"
        )
        page_chunks = split_pdf_page(page_text)
        chunk_total_on_page = len(page_chunks)

        for chunk_no, chunk_text in enumerate(page_chunks, start=1):
            embedding_text = "\n".join(
                [
                    "문서: 순천향대학교 신입학 수시모집 모집요강",
                    f"구분: {current_section}",
                    f"전형: {current_track}" if current_track else "",
                    f"연관 검색어: {search_terms}",
                    f"내용:\n{chunk_text}",
                ]
            )
            identity = f"admission|{pdf_path.name}|{pdf_page}|{chunk_no}|{chunk_text}"
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]

            # max_chars(4000자)를 넘는 드문 페이지만 청크가 2개 이상 생긴다.
            # 이 경우 제목에 "(2/3)"처럼 위치를 남겨서, 같은 페이지에서 쪼개진
            # 조각들이라는 걸 검색 결과 화면에서 바로 알아볼 수 있게 한다.
            page_label = f"{pdf_path.name} {pdf_page}쪽"
            if chunk_total_on_page > 1:
                page_label += f", {chunk_no}/{chunk_total_on_page}"

            documents.append(
                Document(
                    page_content=embedding_text,
                    metadata={
                        "document_type": "admission",
                        "section": current_section,
                        "track": current_track,  # 예: "학생부교과(교과우수자전형)"
                        "track_category": current_track_category,  # 예: "학생부교과"
                        "title": (
                            f"{current_track} ({page_label})"
                            if current_track
                            else f"{current_section} ({page_label})"
                        ),
                        "source": pdf_path.name,
                        "pdf_page": pdf_page,
                        "chunk_no": chunk_no,
                        "chunk_total_on_page": chunk_total_on_page,
                        "raw_text": chunk_text,
                    },
                )
            )
            ids.append(chunk_id)
            section_counts[current_section] += 1

    return documents, ids, section_counts


def main():
    reset = "--reset" in sys.argv
    config.ADMISSION_PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(config.ADMISSION_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(
            f"[중단] {config.ADMISSION_PDF_DIR} 에 PDF가 없습니다. "
            f"입시요강 PDF를 이 폴더에 넣어주세요 (여러 파일로 나눠도 됩니다)."
        )
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

    total_docs = 0
    total_sections: Counter = Counter()
    for pdf_path in pdf_files:
        print(f"\n처리 중: {pdf_path.name}")
        docs, ids, section_counts = build_documents_for_pdf(pdf_path)
        if docs:
            vectordb.add_documents(docs, ids=ids)
        total_docs += len(docs)
        total_sections.update(section_counts)
        print(f"  -> {len(docs)}개 청크 적재")
        for section, cnt in section_counts.most_common():
            print(f"     {section:<20} {cnt:>4}청크")

    print(f"\n완료. 총 {total_docs}개 청크 적재됨 -> {DOMAIN_CFG['persist_dir']}")
    known_sections = {name for _, _, name in ADMISSION_SECTION_RANGES}
    missing = known_sections - set(total_sections.keys())
    if missing:
        print(
            "\n[안내] 아직 감지되지 않은 섹션 (해당 PDF 파일을 아직 안 올렸거나, "
            "헤더 문구가 목차와 달라 감지를 못 했을 수 있습니다):"
        )
        for name in missing:
            print(f"  - {name}")
    print("\n이제 'streamlit run app_streamlit.py' 로 실행하세요.")


if __name__ == "__main__":
    main()
