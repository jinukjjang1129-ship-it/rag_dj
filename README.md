# 동진쎄미켐 정보 RAG 챗봇

동진쎄미켐 홈페이지 자료(공개 정보)에 근거해 회사 정보를 요약·안내하는 RAG 챗봇입니다.

- 메뉴·푸터 자동 제거 후 본문만 저장
- URL에서 카테고리(디스플레이/반도체/신재생에너지/발포제/회사개요) 자동 분류
- "N줄로 요약" 요청 인식
- 자료에 없으면 "찾을 수 없습니다"로 답 (환각 억제)

---

## 두 가지 사용 방법

### A. 링크로 바로 쓰기 (OpenAI API 버전)
배포된 주소를 열면 바로 질문할 수 있습니다.
→ (배포 후 여기에 Streamlit 링크 붙여넣기: https://xxxxx.streamlit.app)

### B. 내 PC에서 로컬로 쓰기 (gemma, 오프라인)
이 코드를 받아 자기 PC에서 실행합니다. OpenAI 키가 필요 없습니다.
(로컬 LLM은 클라우드에서 못 돌아가므로 링크 공유가 아닌 '파일 공유'용입니다.)

---

## A. OpenAI API 버전 — GitHub + Streamlit Cloud 배포

### 1) GitHub에 올릴 파일
```
app.py  ingest.py  rag.py  config.py
requirements.txt  .python-version  .gitignore
.env.example  README.md
data/pages.json
```
※ **.env 와 chroma_store/ 는 올리지 않습니다** (.gitignore가 자동 차단)

### 2) Streamlit Cloud 배포
1. https://share.streamlit.io 접속 → GitHub 로그인
2. New app → 저장소/브랜치 선택, Main file = `app.py`
3. **Advanced settings → Secrets** 에 아래 입력 (키는 여기에만!):
   ```
   OPENAI_API_KEY = "sk-실제키"
   LLM_MODE = "api"
   ```
4. Deploy → 생성된 링크를 공유

### 3) 키 노출 방지 (중요)
- 키는 코드나 .env가 아니라 **Streamlit Secrets 에만** 넣습니다.
- .env 는 .gitignore 로 차단되어 GitHub에 올라가지 않습니다.
- 앱은 배포 시 Secrets를, 로컬 실행 시 .env를 자동으로 읽습니다.

---

## B. 로컬(gemma) 버전 — 파일로 공유, 각자 PC에서 실행

### 1) 설치
```bash
conda activate rag_env
pip install -r requirements.txt
```

### 2) Ollama 준비 (최초 1회)
```bash
ollama pull gemma3:4b
```

### 3) .env 설정
`.env.example` → `.env` 복사 후:
```
LLM_MODE=local
LOCAL_LLM_MODEL=gemma3:4b
```

### 4) 적재 후 실행
```bash
python ingest.py
streamlit run app.py
```

---

## 공통 참고
- **모드 바꾸면 재적재**: api ↔ local 전환 시 임베딩이 달라지므로
  `python ingest.py --reset` 을 다시 실행하세요.
- **데이터 갱신**: 홈페이지가 바뀌면 크롤러로 pages.json 을 새로 만든 뒤
  재적재하세요. 배포 버전은 GitHub에 pages.json 을 갱신하면 자동 반영됩니다.
- **자동 적재**: 배포 시 벡터DB가 비어있으면 앱이 처음 켜질 때 자동으로 적재합니다.

## 예시 질문
- 동진쎄미켐 사업분야를 10줄로 요약해줘
- 반도체 제품을 10줄로 요약해줘
- 디스플레이 사업을 5줄로 정리해줘
- 신재생에너지 분야는 뭘 하나요?
- 발포제 주요 제품 알려줘
