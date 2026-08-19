# allganize-rag-eval

LLM 도입 2차 평가 — 한국어 문서 QA **생성능력** 벤치마크 파이프라인.
데이터셋: [allganize/RAG-Evaluation-Dataset-KO](https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO)

개발 스펙 원본은 [`docs/spec.md`](docs/spec.md).

## 평가 전제

- **Oracle context 방식**: retrieval 은 평가 범위 밖. `target_page_no` 페이지의 파싱 결과만 context 로 고정 주입한다.
- **비교 공정성**: `contexts/contexts.jsonl` 을 1회 생성 후 **동결**. 모든 모델·설정이 동일 context 로 평가된다.
- **주의**: 자체 judge 결과이므로 Allganize 공식 리더보드 수치와 직접 비교 불가. 후보 모델 간 **상대 비교 전용**.

## 확정 사항

| 항목 | 값 | 바꾸는 법 |
|---|---|---|
| API | OpenRouter 단일 키 (`OPENROUTER_API_KEY`) | `.env` |
| 평가 대상 모델 | `deepseek/deepseek-v4-flash-0731` | `run_model.py --model` |
| Provider 고정 | fp8 양자화, `allow_fallbacks: false` | `--quantization` / `--provider-order` |
| Reasoning | high 단일 run (nested `reasoning.effort` 형식만 사용) | `run_model.py --reasoning` |
| 파서 (VLM) | `google/gemini-3.1-pro`, target 페이지만 | `parse_vlm.py --model` |
| Judge | `google/gemini-3.7-flash`, temperature 0 | `judge.py --judge-model` |
| Context 구성 | target 페이지 단독 (±0) | — |
| 생성 설정 | temperature 0 | `run_model.py --temperature` |

> reasoning 은 `{"reasoning": {"effort": "high"}}` nested 형식으로만 전송한다.
> flat 파라미터와 혼용하면 400 이 난다 (`src/common.py:chat_completion`).

## 설치

```bash
pip install -r requirements.txt
cp .env.example .env    # OPENROUTER_API_KEY 입력
```

## 실행 순서와 게이트

```bash
# 0. 데이터 수급 — HF datasets 기본, 실패 시 repo CSV 직접 다운로드로 폴백
python src/fetch_dataset.py

# 1. PDF 다운로드 (URL 만료 문서 = 평가 제외 문항 확정)
python src/download_pdfs.py
#    → reports/download_log.csv, reports/excluded_questions.csv 확인

# 2. target 페이지만 파싱 — 먼저 5페이지로 품질 육안 검수
python src/parse_vlm.py --limit 5 --save-image
#    → cache/parsed/*.json 의 markdown 과 cache/pages/*.png 대조
#    ★ 게이트 A: 표·차트 변환 품질 확인 후 전체 실행
python src/parse_vlm.py

# 3. context 생성 (동결 대상)
python src/build_contexts.py
#    → reports/context_review_samples.md 로 도메인×타입 샘플 검수
#    ★ 게이트 B: 이후 수정 금지. 재생성이 필요하면 --version v2

# 4. 모델 실행 — 10문항 드라이런으로 provider·reasoning 동작 확인 후 본 run
python src/run_model.py --limit 10
python src/run_model.py

# 5. 채점
python src/judge.py
#    → reports/judge_review_*.csv 30~50건 수동 대조 후 본채점 확정

# 6. 리포트
python src/report.py
```

각 단계는 **재실행 안전**하다. 파싱은 캐시를 스킵하고, `run_model`·`judge` 는 기존 성공분을 재사용한다
(전량 재실행은 `--force`).

### 게이트 (되돌릴 수 없는 지점)

- **게이트 A (2→3)**: 파싱 품질 검수. image/table 샘플을 충분히 확인한다. 동결 이후 발견된 파싱 문제는 contexts v2 재생성이 필요하다.
- **게이트 B (3)**: `contexts.jsonl` 동결. 이후 모든 모델·설정 비교의 기준선.
  `contexts.jsonl.meta.json` 에 sha256 이 기록되고, `build_contexts.py --verify` 로 훼손 여부를 확인한다.
  run 결과 CSV 에도 `contexts_sha256` 이 함께 기록된다.

## 산출물

```
data/documents.csv, data/dataset.csv   원본 메타 (qid 는 여기서 1회 부여)
data/pdfs/                             원본 PDF (git 미추적)
cache/parsed/<문서명>__p<페이지>.json   파서 원응답 + 마크다운
contexts/contexts.jsonl                ★ 동결된 oracle context
contexts/contexts.jsonl.meta.json      동결 해시·건수
results/raw/<endpoint>__<model>__<reasoning>.csv    모델 응답 + provider/latency/토큰
results/scored/…                       + verdict, judge_reason, 수치 크로스체크
reports/download_log.csv               문서별 성공/실패 + 걸린 문항 수
reports/excluded_questions.csv         다운로드 실패로 제외된 문항
reports/context_review_samples.md      게이트 A/B 검수용 샘플
reports/judge_review_*.csv             judge 수동 대조 샘플
reports/parser_suspect_*.csv           image·table 오답의 모델 실패 vs 파서 실패 후보
reports/summary.md, summary_by_axis.csv  집계 리포트
```

## 리포트 집계 축

전체 / domain(5개) / context_type(paragraph·table·image). 제외 문항 수를 함께 명시하고,
`results/scored/` 에 여러 run 이 있으면 모델 간 비교표가 자동으로 붙는다.

**"모델 실패 vs 파서 실패" 구분**: image·table 오답에 대해 정답의 수치·날짜 토큰이 context 안에 있는지 검사한다.
context 에 없으면 파서 실패 의심, 있으면 모델 실패 의심으로 분류하고 목록을 `reports/parser_suspect_*.csv` 로 뽑는다
(1차 신호일 뿐이므로 최종 판단은 육안 확인).

## judge 신뢰도 보조 장치

- 판정은 O/X 이진 + 근거 1줄. 프롬프트는 `src/prompts/judge_v1.txt` 로 분리·버전 관리.
- 수치형 정답(금액·비율·날짜)은 정규식 일치 여부를 `numeric_match` 컬럼에 병행 기록한다.
- judge 판정과 정규식 결과가 엇갈리는 케이스를 우선 뽑아 `reports/judge_review_*.csv` 로 내보낸다.

## 재현성

모든 run 결과에 model, reasoning_effort, temperature, prompt_version, 라우팅된 provider,
`contexts_sha256`, run 시각이 기록된다. provider 가 두 종류 이상으로 섞이면 실행 로그와 리포트에 경고가 뜬다.

## 비용 (개략)

| 단계 | 규모 | 예상 |
|---|---|---|
| 파싱 (Gemini 3.1 Pro) | ~300 페이지 | ~$3–5 |
| 대상 모델 (V4 Flash, high 1 run) | 300 호출 | ~$1–2 |
| Judge (3.7 Flash) | 300 호출 | ~$0.5 |
| 합계 | | ~$7 이내 |

후보 모델 추가 시 모델당 ~$3–5. contexts 가 동결돼 있어 파싱 비용은 재발생하지 않는다.

## 개발용

```bash
python tools/smoke_test.py    # 네트워크 없이 build_contexts / report / 수치 크로스체크 검증
```

작업 루트는 `ALLGANIZE_EVAL_ROOT` 환경변수로 바꿀 수 있다 (기본: 레포 루트).
