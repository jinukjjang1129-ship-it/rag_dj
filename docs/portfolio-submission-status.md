# GitHub 포트폴리오 제출 준비 현황

작성일: 2026-11-13

## 대표 프로젝트 구성

| 프로젝트 | 구성 목적 | 현재 검토 위치 |
|---|---|---|
| 트럼프 트윗과 시장 반응 | LDA, 이벤트 분석, 거래비용·시간 정합성 검토 | [README 정리 PR](https://github.com/jinukjjang1129-ship-it/trump-tweets-topic-modeling/pull/3) |
| 다중 도메인 RAG | 문서 검색, 정형 질문 라우팅, 로컬 Neo4j, 근거 확인 | [구현 검토 PR](https://github.com/jinukjjang1129-ship-it/rag_dj/pull/2) |
| 배터리 OCV 분류 | 시간 분할, 누수 점검, 임곗값별 분류 평가 | [구현 검토 PR](https://github.com/jinukjjang1129-ship-it/ocv__/pull/1) |
| 부산 전·월세 공간 분석 | 실거래 집계, 지표 정의, 지도 시각화 | [README·집계 검토 PR](https://github.com/jinukjjang1129-ship-it/busan-rent-choropleth/pull/1) |

## 추가 저장소 정리

- [프로필 기반 챗봇](https://github.com/jinukjjang1129-ship-it/attachment-profile-chatbot/pull/1): 실제 기능과 실행 전제, 누락된 인덱스 준비 과정 명시
- [설문 프로필 시각화](https://github.com/jinukjjang1129-ship-it/survey/pull/1): 설문 계산과 시각화, 실행 방법 정리
- [OCV 교육용 앱](https://github.com/jinukjjang1129-ship-it/ml_streamlit/pull/1): 실습 목적, 정답 생성 변수에 따른 누수 주의사항 명시
- [지하철 분석 기획 기록](https://github.com/jinukjjang1129-ship-it/old_subway_analysis/pull/1): 현재 README만 존재하는 범위 명시

## 성과와 기여도 표기 기준

업무 원자료를 공개할 수 없고 평가 데이터셋이 다를 가능성이 있으므로, 기존 +2.1%p 및 Recall 0.98·Precision 0.94의 직접 재현 검증은 보류한다. 제공 샘플의 결과를 기존 업무 성과의 반증으로 해석하지 않는다.

교육 관련 코드는 사용자가 강의를 진행한 맥락에서 구분한다. 파일별 구현 주체가 확인되지 않은 경우 외부 강의 제공 코드나 사용자 단독 구현으로 단정하지 않는다.

Neo4j는 로컬 실행 범위를 기준으로 설명한다. 검색 성공률과 생성 답변 정확도를 구분하고, 측정하지 않은 성능을 성과로 표시하지 않는다.

## 준비된 변경과 미완료 작업

원본 ZIP과 저장소 미러를 작업 환경에 보존했다. 위 변경은 별도 브랜치와 Draft PR이며 main에 병합하지 않았다.

새 이름으로 분리할 코드와 README는 작업 환경에 준비했다.

- multi-domain-rag-system
- battery-ocv-classification
- jinukjjang1129-ship-it: 프로필 README용 저장소

GitHub 연결의 저장소 생성·설정 변경 요청이 HTTP 403으로 거절되어 세 저장소는 아직 생성하지 못했다. 사용자에게 승인받은 코드잇 위클리 페이퍼 9개 저장소의 비공개 전환도 완료되지 않았다. 대상은 codeit_weekly_paper_1, _3, _4, _5, _6, _7, _8, _9, _10이다.

저장소 소유자의 Settings → General → Danger Zone → Change repository visibility에서 비공개로 전환할 수 있다. 기존 자료를 삭제하는 작업은 필요하지 않다.

신규 저장소의 공개 전환과 최종 PR 병합은 사용자 검토·승인 후 진행한다. 프로필에 표시할 대표 저장소 고정은 네 대표 프로젝트가 준비된 뒤 수행한다.

## 검증 범위

기존 핵심 프로젝트 테스트는 트럼프 9개, RAG 38개, OCV 8개다. 신규 저장소용 복사본에서도 RAG 38개와 OCV 8개 테스트를 다시 통과했다. 복사본 테스트를 별도의 신규 테스트 수로 합산하지 않는다.

부산 프로젝트는 원본 Excel로 전세 26,679건·월세 22,793건의 16개 자치구 집계를 재계산하여 저장 CSV와 일치함을 확인했다. 모든 추가 앱의 전체 UI·배포를 재검증한 것은 아니다.

이번 안내에는 원자료, 회사·취업규칙 문서, 데이터 파생 화면, 배포용 ZIP을 첨부하지 않는다. 해당 자료의 공개 가능 여부는 별도 검토 대상으로 남겨 둔다.
