"""common/config.py — 경로 / 모델 / 도메인별 설정을 한 곳에서 관리."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=str(ENV_PATH))
DATA_DIR = Path(os.environ.get("SCH_DATA_DIR", PROJECT_ROOT / "data"))
CHUNKS_JSON = DATA_DIR / "chunks.json"
ADMISSION_PDF_DIR = DATA_DIR / "admission_pdfs"
PAPER_PDF_DIR = DATA_DIR / "paper_pdfs"
HISTORY_DB = DATA_DIR / "history" / "history.sqlite3"

CHROMA_ROOT = Path(os.environ.get("SCH_CHROMA_ROOT", PROJECT_ROOT / "chroma_store"))


def get_secret(key: str, default=None):
    return os.environ.get(key, default)


# ── 모델 설정 (로컬 전용, dj_rag_local과 동일한 변수명) ──
LOCAL_EMBED_MODEL = get_secret("LOCAL_EMBED_MODEL", "jhgan/ko-sroberta-multitask")
LOCAL_LLM_MODEL = get_secret("LOCAL_LLM_MODEL", "gemma3:4b")
OLLAMA_TEMPERATURE = float(get_secret("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_NUM_CTX = int(get_secret("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(get_secret("OLLAMA_NUM_PREDICT", "1024"))
USE_OLLAMA = get_secret("USE_OLLAMA", "1") not in ("0", "false", "False")

# ── 검색 공통 설정 ──
DEFAULT_TOP_K = 5
MIN_RESULTS_BEFORE_FALLBACK = 1

# 검색 결과 하나를 LLM 프롬프트에 넣을 때 자르는 길이.
# 입시요강처럼 청크 자체가 큰 도메인이 있어서, engine.build_prompt()가
# 이 값을 공통으로 참조한다. top_k와 곱한 값이 OLLAMA_NUM_CTX(8192)를
# 문자 수와 토큰 수는 다르므로 실제 LLM 토크나이저로 별도 검증해야 한다.
PROMPT_CHUNK_CHAR_LIMIT = 2000

# 검색 관련도(0~100)가 이 값보다 낮으면 "자료에 없음"으로 즉시 응답하고
# LLM 프롬프트에 넣지 않는다. 로컬 소형 모델이 스스로 "모른다"고 판단하는 데
# 약하기 때문에, 이 게이트는 LLM 판단이 아니라 코드(점수 임계값)로 강제한다.
SCOPE_RELEVANCE_THRESHOLD = 32.0

# ── 학과 별칭 ──
DEPARTMENT_ALIASES = {
    "의료IT공학과": [
        "의료it공학과",
        "의료it공학",
        "의료it",
        "의료 it",
        "메디컬it",
        "의료정보공학",
        "의료아이티",
    ],
    "정보통신공학과": [
        "정보통신공학과",
        "정보통신공학",
        "정보통신",
        "정통과",
    ],
    "전자공학과": [
        "전자공학과",
        "전자공학",
        "전자과",
    ],
    "컴퓨터소프트웨어공학과": [
        "컴퓨터소프트웨어공학과",
        "컴퓨터소프트웨어공학",
        "컴퓨터소프트웨어",
        "컴소공",
        "컴공",
        "소프트웨어공학과",
        "소프트웨어공학",
    ],
    "AI빅데이터학과": [
        "ai빅데이터학과",
        "ai빅데이터",
        "ai·빅데이터",
        "ai 빅데이터",
        "빅데이터학과",
        "인공지능학과",
        "ai학과",
        "에이아이빅데이터",
    ],
}

UNTRAINED_DEPARTMENTS = [
    {"department": "의예과", "homepage": "https://home.sch.ac.kr/premedicine/"},
]

# 순천향대 전체를 가리키는 일반 키워드 (스코프 게이트 1차 판별용)
UNIVERSITY_KEYWORDS = [
    "순천향",
    "sch",
    "학과",
    "교수",
    "전형",
    "입시",
    "수시",
    "정시",
    "모집요강",
    "학점",
    "교육과정",
    "졸업",
    "진로",
    "취업",
    "캠퍼스",
    "논문",
    "학술제",
    "장학금",
    "등록금",
    "입학처",
]

# ── 도메인(탭)별 설정 ──
DOMAINS = {
    "department": {
        "label": "학과정보",
        "collection": "sch_departments",
        "persist_dir": str(CHROMA_ROOT / "department"),
        "role": "순천향대학교 학과 정보를 안내하는 도우미",
        "not_ready_message": (
            "학과정보가 아직 적재되지 않았습니다. "
            "`python ingest/ingest_department.py` 를 먼저 실행해 주세요."
        ),
        "suggested_questions": [
            "컴퓨터소프트웨어공학과 교수님은 누가 있나요?",
            "의료IT공학과 졸업 후 진로는 어떻게 되나요?",
            "전자공학과 교육과정을 알려주세요.",
            "정보통신공학과 졸업자격요건이 궁금합니다.",
            "AI빅데이터학과는 무엇을 배우나요?",
        ],
    },
    "admission": {
        "label": "입시정보",
        "collection": "sch_admission",
        "persist_dir": str(CHROMA_ROOT / "admission"),
        "role": "순천향대학교 2027학년도 수시모집 입시요강을 안내하는 도우미",
        # 청크가 페이지 단위(최대 4000자)로 커졌기 때문에, 학과정보(top_k=5,
        # 청크가 작음)와 달리 여기는 top_k를 낮춰서 프롬프트가
        # OLLAMA_NUM_CTX(8192)를 넘지 않게 관리한다.
        "top_k": 3,
        "not_ready_message": (
            "입시요강 PDF가 아직 적재되지 않았습니다. "
            "data/admission_pdfs/ 에 PDF를 넣고 "
            "`python ingest/ingest_admission.py` 를 실행해 주세요."
        ),
        "suggested_questions": [
            "학생부교과전형 지원자격이 궁금합니다.",
            "수시모집 원서접수 일정이 어떻게 되나요?",
            "AI빅데이터학과 수시 모집인원은 몇 명인가요?",
            "농어촌학생전형 지원자격을 알려주세요.",
            "전형료는 얼마인가요?",
        ],
    },
    "paper": {
        "label": "논문정보",
        "collection": "sch_papers",
        "persist_dir": str(CHROMA_ROOT / "paper"),
        "role": "순천향대학교 학술제 및 소속 연구자의 논문 정보를 안내하는 도우미",
        "not_ready_message": (
            "논문 자료가 아직 적재되지 않았습니다. "
            "data/paper_pdfs/ 에 논문집 PDF를 넣고 "
            "`python ingest/ingest_paper.py` 를 실행해 주세요."
        ),
        "suggested_questions": [
            "2024 학술제에서 발표된 AI 관련 논문이 있나요?",
            "의료IT 분야 논문 초록을 요약해 주세요.",
            "이 논문의 저자와 소속을 알려주세요.",
        ],
    },
}

# ── 논문 OCR 설정 ──
# 일부 논문 PDF(특히 특정 학술지)는 본문 글자가 폰트가 아니라 아주 작은
# 이미지 조각으로 박혀 있어서, pypdf로 텍스트가 거의 안 뽑힌다(실측: 쪽당
# 평균 70~80자). 이런 문서는 페이지를 이미지로 렌더링한 뒤 OCR을 돌려서
# 텍스트를 얻는다. 정상 문서(쪽당 평균 1500~3000자)는 그대로 pypdf만 씀.
OCR_ENABLED = get_secret("OCR_ENABLED", "1") not in ("0", "false", "False")
OCR_LANG = get_secret("OCR_LANG", "kor+eng")
OCR_DPI = int(
    get_secret("OCR_DPI", "300")
)  # 200 DPI는 실제로 글자가 심하게 깨짐, 300에서 확 나아짐
OCR_MIN_AVG_CHARS_PER_PAGE = int(get_secret("OCR_MIN_AVG_CHARS_PER_PAGE", "200"))

# poppler/tesseract를 conda·brew·apt로 설치해도, Windows에서는 설치 직후
# 터미널이 PATH 갱신을 못 읽어서 "poppler installed and in PATH?" 에러가
# 자주 난다. PATH 문제와 씨름하는 대신, 필요하면 여기서 폴더를 직접
# 지정할 수 있게 해뒀다 (지정 안 하면 기존처럼 PATH에서 자동으로 찾음).
#   예: POPPLER_PATH=C:\Users\me\anaconda3\envs\rag_env\Library\bin
#       TESSERACT_CMD=C:\Users\me\anaconda3\envs\rag_env\Library\bin\tesseract.exe
POPPLER_PATH = get_secret("POPPLER_PATH", None)
TESSERACT_CMD = get_secret("TESSERACT_CMD", None)

# tesseract가 언어팩(kor.traineddata 등)을 찾는 기준도 TESSDATA_PREFIX라는
# 별도 환경변수인데, 이것도 Windows에서 conda 설치 스크립트가 제대로 못
# 잡아주는 경우가 흔하다 (tesseract --list-langs가 "./"에서 찾는 등).
# 이것도 시스템 환경변수에 기대지 않고 직접 지정할 수 있게 해둔다.
#   예: TESSDATA_DIR=C:\Users\me\anaconda3\envs\rag_env\share\tessdata
TESSDATA_DIR = get_secret("TESSDATA_DIR", None)

TARGET_NAME = "순천향대학교"
