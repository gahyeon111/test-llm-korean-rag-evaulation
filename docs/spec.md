# allganize-rag-eval 개발 스펙

> LLM 도입 2차 평가: 한국어 문서 QA 생성능력 벤치마크
> 데이터셋: [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)

---

## 1. 목적 및 평가 전제

- **목적**: 사내 도입 후보 모델의 한국어 문서 기반 QA **생성능력** 비교 평가
- **방식**: Oracle context 방식 — retrieval은 평가 범위에서 제외. `target_page_no` 페이지의 파싱 결과를 context로 고정 주입
- **비교 공정성**: `contexts.jsonl`을 1회 생성 후 **동결**. 모든 모델·설정은 동일 context로 평가
- **주의**: 자체 judge 결과이므로 Allganize 공식 리더보드 수치와 직접 비교 불가. 후보 모델 간 **상대 비교 전용**

## 2. 확정 사항

| 항목 | 값 |
|---|---|
| API | OpenRouter 단일 키 (`OPENROUTER_API_KEY`) |
| 평가 대상 모델 | `deepseek/deepseek-v4-flash-0731` |
| Provider 고정 | fp8 양자화 provider, `allow_fallbacks: false` |
| Reasoning | **high 단일 run** (nested `reasoning.effort` 형식만 사용 — flat 파라미터와 혼용 시 400 에러) |
| 파서 (VLM) | `google/gemini-3.1-pro` — **target 페이지만 파싱** (전량 파싱 금지) |
| Judge | `google/gemini-3.7-flash`, temperature 0 |
| Context 구성 | target 페이지 단독 (±0) |
| 생성 설정 | temperature 0 고정 |

## 3. 디렉토리 구조

```
allganize-rag-eval/
├── .env                      # OPENROUTER_API_KEY
├── requirements.txt
├── data/
│   ├── documents.csv         # HF 데이터셋의 문서 메타 (URL 포함)
│   ├── dataset.csv           # 300문항 (question, target_answer, target_file_name, target_page_no, context_type, domain)
│   └── pdfs/                 # 파일명 = target_file_name 그대로
├── cache/
│   └── parsed/
│       └── <문서명>__p<페이지>.json   # (문서, 페이지) 단위 파서 원응답
├── contexts/
│   └── contexts.jsonl        # ★ 300문항 × oracle context — 동결 대상
├── results/
│   ├── raw/                  # <endpoint>__<model>__<reasoning>.csv
│   └── scored/               # raw + judge 판정 컬럼
├── reports/
└── src/
    ├── download_pdfs.py
    ├── parse_vlm.py
    ├── build_contexts.py
    ├── run_model.py
    ├── judge.py
    └── report.py
```

## 4. 모듈별 스펙

### 4.1 download_pdfs.py
- **입력**: `data/documents.csv`
- **출력**: `data/pdfs/`, `reports/download_log.csv`
- 원문 URL 만료 케이스 존재 → 실패 문서와 그에 걸린 문항 수를 로그에 기록
- 다운로드 실패 문서에 걸린 문항은 평가 제외 처리하고 제외 목록을 리포트에 명시

### 4.2 parse_vlm.py
- **입력**: `data/pdfs/` + `data/dataset.csv`의 (target_file_name, target_page_no) 유니크 페어
- **출력**: `cache/parsed/<문서명>__p<페이지>.json`
- 처리: 해당 페이지만 이미지 렌더링(≥150 DPI) → Gemini 3.1 Pro → 마크다운 변환
- 파싱 프롬프트: 표는 마크다운 표로, 차트·이미지는 수치 포함 서술로 변환하도록 지시
- 캐시 존재 시 스킵 (재실행 안전)
- 원응답(raw response) 그대로 보존 — 후처리는 build_contexts에서

### 4.3 build_contexts.py
- **입력**: `cache/parsed/` + `data/dataset.csv`
- **출력**: `contexts/contexts.jsonl`
- 레코드 스키마: `{qid, domain, context_type, question, target_answer, target_file, target_page, context}`
- 생성 후 **동결** — 이후 수정 금지. 수정 필요 시 버전 접미사(v2) 부여
- 생성 직후 도메인×context_type별 샘플 10건 수동 검수 (특히 image/table 문항의 파싱 품질)

### 4.4 run_model.py
- **입력**: `contexts.jsonl`
- **출력**: `results/raw/openrouter__deepseek-v4-flash-0731__high.csv`
- 요청 옵션: temperature 0, provider 고정(fp8), reasoning effort는 high 고정 (CLI 인자로 변경 가능하게 유지)
- 응답 메타 기록: 실제 라우팅된 provider, latency, input/output/reasoning 토큰 수
- 실패 시 지수 백오프 재시도 3회, 최종 실패 문항은 별도 표기

### 4.5 judge.py
- **입력**: `results/raw/*.csv` (+ contexts.jsonl의 target_answer)
- **출력**: `results/scored/*.csv`
- 판정: target_answer 기준 O/X 이진 + 판정 근거 1줄
- Judge 프롬프트는 파일로 분리·버전 관리 (`src/prompts/judge_v1.txt`)
- 보조 검증: 수치형 정답(금액·비율·날짜)은 정규식 일치 여부 컬럼 병행 기록 → judge 오판 크로스체크
- Judge 불일치·오답 케이스 30~50건 수동 샘플 검증 후 본채점 확정

### 4.6 report.py
- **입력**: `results/scored/`
- **출력**: `reports/`
- 집계 축: 전체 / domain(5개) / context_type(paragraph·table·image)
- 제외 문항(다운로드 실패 등) 수 명시
- image·table 오답은 "모델 실패 vs 파서 실패" 구분을 위한 샘플 목록 첨부

## 5. 비용 추정 (개략)

| 단계 | 규모 | 예상 비용 |
|---|---|---|
| 파싱 (Gemini 3.1 Pro) | ~300 페이지 | ~$3–5 |
| 대상 모델 (V4 Flash, high 1 run) | 300 호출 | ~$1–2 |
| Judge (3.7 Flash, 1 run) | 300 호출 | ~$0.5 |
| **합계** | | **~$7 이내** |

- 후보 모델 추가 시 모델당 ~$3–5 증가 (contexts 동결로 파싱 비용 재발생 없음)

## 6. 리스크 및 주의사항

1. **PDF URL 만료**: 일부 문서 다운로드 실패 가능 → 제외 기준 사전 정의 (4.1)
2. **파서 품질 = 성능 상한**: image/table 문항은 파싱 실패가 모델 오답으로 전가됨 → context_type별 집계 + 샘플 검수로 분리
3. **Provider 오염**: fp8 고정 미적용 시 run 간 양자화가 섞임 → provider 필드 기록으로 사후 검증
4. **Judge 편향**: 향후 Gemini 계열을 도입 후보로 평가할 경우 judge 교체 필요 (self-preference)
5. **재현성**: 모든 run의 모델 스냅샷·프롬프트 버전·provider를 결과 파일에 기록

## 7. 개발 순서 및 게이트

| 순서 | 모듈 | 완료 기준 (검증 포인트) |
|---|---|---|
| 1 | `download_pdfs.py` | 63개 중 성공/실패 집계, 실패 문서에 걸린 문항 수 확인 → **제외 문항 확정** |
| 2 | `parse_vlm.py` | 유니크 (문서, 페이지) 페어 수 확인 → 5페이지 선행 파싱으로 표·이미지 품질 육안 검수 → 전체 실행 |
| 3 | `build_contexts.py` | 300건 스키마 검증 + 도메인×context_type별 샘플 검수 → **contexts.jsonl 동결** |
| 4 | `run_model.py` | 10문항 드라이런으로 provider 고정·reasoning 파라미터 정상 동작 확인 → high 본 run 1회 |
| 5 | `judge.py` | 판정 30~50건 수동 대조로 judge 신뢰도 확인 → 본채점 |
| 6 | `report.py` | 집계표 + 제외 문항·파서 실패 의심 케이스 명시 |

### 게이트 (되돌릴 수 없는 지점)

- **게이트 A (2→3)**: 파싱 품질 검수. image/table 문항 샘플을 충분히 확인 — 동결 이후 발견되는 파싱 문제는 contexts v2 재생성 필요
- **게이트 B (3)**: contexts.jsonl 동결. 이후 모든 모델·설정 비교의 기준선

### 착수 전 결정 사항

- 데이터 수급 방식: HF `datasets` 라이브러리 vs CSV 직접 다운로드 (1단계 시작 시 확정)
