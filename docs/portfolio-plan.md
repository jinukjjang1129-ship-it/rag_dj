# 전체 조사 및 순차 작업 계획 — 2026-11-11

## 보존
ZIP 6개는 work/originals에 복사하고 SHA-256 및 파일 목록을 archive_inventory.json에 기록했다. Git 저장소 trump-tweets-topic-modeling, ocv__, rag_dj, ml_streamlit은 mirror로 이력을 보존했다. 원본 코드/노트북/문서는 work/source에 별도 추출했다. 이 디렉터리는 공개 업로드하지 않는다.

## 조사 결과와 저장소 배치
- Trump: 기존 PR #1(faa5120)의 src/notebooks 분리와 문서화는 재작성하지 않는다. 후속 브랜치를 PR head에서 분기한다. 원본 트윗, 2016~2019 SPXUSD OHLC, 가공 topic0/백테스트 CSV가 있다. price_returns CSV 및 중간 Excel, 학습 모델은 없다. 노트북은 탐색 코드이며 빈 노트북도 있다. Gensim/NLTK LDA, Huber 회귀, 이벤트 누적합을 구현했다. 금융자료 시각을 UTC와 NY로 다르게 처리한 코드가 공존하고 행 shift를 경과 분으로 취급한다. 토픽 수 20의 coherence 평가 코드는 있으나 후보 수 비교 근거는 아직 확인되지 않았다. 원본 데이터의 재배포 허가는 확인 필요.
- SCH v2/v3: Streamlit 3도메인, Chroma, HuggingFace 한국어 임베딩, Ollama gemma3:4b, SQLite 이력. v3는 기존 14개 Python 중 5개 변경(engine/admission/department 및 admission/paper ingest), 그래프 적재와 FastAPI 데모 추가. UI/config/guard/history는 동일하다. v3 README는 v2로 남아 있다. Neo4j가 검색 엔진과 통합됐는지 별도 검증한다. 논문 PDF, 원문 chunks, 벡터DB, 대화이력은 공개 제외. GUIDE는 강의 실습 자료이며 사용자는 해당 강의를 직접 진행했다고 확인했다. 교육용 자료와 제품 코드를 용도로 구분하며, 외부 제공 예제로 단정하지 않는다. multi-domain-rag-system은 신규 생성보다 공개 가능한 코드 묶음을 먼저 준비한다.
- OCV: 기존 ocv__가 동일 문제의 Streamlit 앱이다. ml_streamlit도 중복 후보이므로 별도 OCV 저장소를 즉시 만들지 않는다. ZIP에는 두 강의 노트북, pokemon/tips CSV, OCV_강의용_샘플데이터.xlsx 및 회원/학습 관련 parquet 4개가 있다. 초기 조사에서 xlsx/parquet를 놓친 오류를 정정했다. 강의용 OCV 샘플과 원래 제조 데이터 성과의 동일성은 확인 필요. 기존 저장소에는 모델 pickle만 있고 README가 참조하는 train_model.py가 없다. pickle을 임의 실행하지 않는다. OCV4로 정답을 만들면서 OCV4를 입력하면 목적 누수다. Recall .98/Precision .94는 미검증. 강의 실습과 독립 확장 코드를 문서로 구분한다.
- DJ: 기존 rag_dj와 ZIP의 API/local 버전 비교 후 기존 저장소를 우선 사용한다. Playwright 수집 → pages.json → 정제/청킹 → Chroma → 응답 흐름. 로컬 버전은 API 코드 제거 및 Selenium 대안이 있다. 크롤링 허가를 README가 주장하지만 독립 확인 자료는 없다. 원본 페이지 내용의 재배포 권한은 별도 확인 필요.
- 취업규칙: PDF 조항 기반 RAG 강의 실습. rag.py에 IndentationError가 있어 현재 실행 불가. document-rag-experiments에 DJ와 관련 예제로 통합 가능한지 검토하며 단독 저장소를 강제 생성하지 않는다. PDF는 비공개 자료로 취급한다.

## 공통 공개 제외
.env, 비밀번호/키, __pycache__, .pyc, chroma_store, history.sqlite3, 모델 pickle, 사내 문서, 라이선스 불명 PDF/원문 데이터. 값은 보고서나 로그에 출력하지 않는다. 사용자는 강의 진행자다. 이력서에 직접 자료 준비 및 8회 내부 교육을 명시했다. 개별 코드 저작·기여 범위는 자료 근거에 따라 연결한다.

## 실행 순서 및 완료 기준
1. Trump: 기존 스크립트 실제 실행과 오류 기록 → 데이터 품질/시간대/원본 결과 감사 → 명시적 시간대 가정 하에 경과시간 수익률, 1분 실행 지연, 세션 제한, 비중복 이벤트, 비용 포함 동일 노출 롱/숏 비교 → 기간 분할 회귀와 BH 보정 → LDA 원형 재실행 및 train-only 평가 → 테스트/리뷰/보고서/Draft PR. 가격 타임존 출처가 미확인인 한 성과는 조건부 결과다.
2. SCH: v2/v3 코드 diff와 메타데이터 감사 → 목록형 완전성/필터 fallback/중복 문제 재현 → 동일 평가셋에서 검색과 답변 지표 분리 → 단위/통합 테스트. Ollama/Neo4j가 없으면 실제 생성 성능을 미측정으로 기록한다.
3. OCV: 강의용 OCV Excel을 실제 실행하되 원래 제조 성과와 구분하고 train/validation/test 분리, 학습 내 결측 처리, 4모델, 검증셋 임곗값 선택을 재사용 CLI로 구현. 제공된 강의용 샘플에서 측정한 성능은 그 데이터에 한정한다. fixture 테스트는 성과와 구분한다.
4. 취업규칙/DJ: 문법 오류, 수집/정제/검색 분리, 재시도 및 출처 검증, 공개 제외를 적용. Playwright 실행을 시도하고 네트워크/브라우저 실패는 기록한다.

각 단계 테스트/코드 리뷰 후 다음 단계 구현에 진입한다. 모든 수치는 실행 로그와 연결하고 통과하지 못한 검증은 남은 과제로 남긴다. main 수정, merge, 신규 공개 저장소 생성은 하지 않는다.
