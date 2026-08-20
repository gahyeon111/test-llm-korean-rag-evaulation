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
| 평가 대상 모델 | `deepseek/deepseek-v4-flash-0731` * | `run_model.py --model` |
| Provider 고정 | fp8 양자화, `allow_fallbacks: false` | `--quantization` / `--provider-order` |
| Reasoning | high 단일 run (nested `reasoning.effort` 형식만 사용) | `run_model.py --reasoning` |
| 파서 (VLM) | `google/gemini-3.7-flash` *, target 페이지만 | `parse_vlm.py --model` |
| Judge | `google/gemini-3.7-flash`, temperature 0 | `judge.py --judge-model` |
| Context 구성 | target 페이지 단독 (±0) | — |
| 생성 설정 | temperature 0 | `run_model.py --temperature` |

> \* **스펙 대비 변경**: 스펙의 `google/gemini-3.1-pro` 는 OpenRouter 카탈로그에 없다
> (실제 ID 는 `google/gemini-3.1-pro-preview`, $2/$12 per 1M).
> 파싱 100페이지 기준 pro 계열 ~$2.2 vs `gemini-3.7-flash` ~$0.8 이고,
> 세대가 더 높은 flash 라 문서 파싱 품질도 크게 뒤지지 않아 기본값으로 잡았다.
> 게이트 A 검수에서 표·차트 재현이 부족하면
> `--model google/gemini-3.1-pro-preview` 로 올린다 (contexts 동결 전에 결정할 것).
>
> 모델 ID 는 스펙 기준값이다. OpenRouter 카탈로그에 없는 ID 면 400 이 나므로
> `python tools/check_models.py` 로 먼저 확인하고 실제 ID 로 바꿔 쓴다.
> 파싱/실행/채점 스크립트는 연속 3회 실패하면 설정 문제로 보고 중단한다(`--abort-after`).
>
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
#    → 300문항 / 64문서. context_type 라벨을 paragraph·table·image 로 통일하고
#      원본 값은 context_type_raw 로 남긴다 (--no-normalize-context-type 로 해제)

# 1. PDF 확보 — url 은 게시판 상세페이지이므로 첨부를 찾아 받는다
python src/download_pdfs.py
#    → reports/needs_review.csv  ★ 파일명이 안 맞아 눈으로 봐야 하는 문서
#    → reports/download_log.csv  어떤 첨부를 골랐는지(picked_name/name_score)
#    → reports/excluded_questions.csv  실패 문서에 걸린 제외 문항
#    첨부를 못 찾은 사이트가 있으면: python tools/inspect_html.py

# 1-1. 받은 PDF 가 그 문서가 맞는지 검증하고, 의심 건은 제외 목록으로 분리
python tools/review_downloads.py --render --write-exclusions
#    → data/excluded_files.txt (파이프라인이 이 목록을 건너뛴다)
#    → reports/download_review.md / review_pages/*.png 로 확인 후 줄을 지우면 복귀
#    ✅ 만으로 돌리려면: --write-exclusions --keep-codes match

# 1-2. 모델 ID 확인 — 스펙의 ID 가 OpenRouter 카탈로그에 없으면 400 이 난다
python tools/check_models.py                     # 파서/평가대상/judge 3개 존재 확인
python tools/check_models.py --search gemini     # 후보 찾기
python tools/check_models.py --endpoints <모델>   # provider·양자화(fp8) 확인

# 2. target 페이지만 파싱 — 먼저 문서를 흩어 샘플로 품질 육안 검수
python src/parse_vlm.py --sample 10 --save-image
#    --limit 은 앞에서부터 잘라 한 문서에 몰리므로, 검수에는 --sample 을 쓴다
#    표·차트 품질만 집중해서 보려면: --sample 10 --types image,table
#    빈 응답이 캐시에 남았으면: --retry-empty
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
reports/download_review.md             받은 PDF 내용 검증 (페이지 수·정답 대조)
data/excluded_files.txt                평가에서 뺄 문서 목록 (직접 편집 가능)
reports/parser_suspect_*.csv           image·table 오답의 모델 실패 vs 파서 실패 후보
reports/summary.md, summary_by_axis.csv  집계 리포트
```

## 원본 데이터 특이사항

- **`documents.csv` 의 url 은 PDF 직링크가 아니다.** 세 종류가 섞여 있다 —
  게시글 **상세페이지**(대부분), 게시판 **목록페이지**(예: `fsc.go.kr/po010101?...`),
  드물게 PDF 직링크. 목록페이지는 제목이 target 파일명과 가장 비슷한 게시글로
  한 단계 더 들어가서 첨부를 찾는다(`--follow-links`, 기본 3건).
  상세페이지의 PDF 는 그 안의 첨부파일이다. 게다가 한 게시글에 첨부가 여러 개라 서로 다른 문서가
  **같은 URL** 을 갖는 경우가 있다 (예: 한국은행 `view.do?nttId=10082951` → `2024년 3월_2…`, `2024년 3월_3…`).
  그래서 `download_pdfs.py` 는 **url 단위로 묶어서** 페이지의 PDF 첨부를 한 번만 전부 받아
  `cache/attachments/` 에 저장한 뒤, 그 url 에 걸린 문서들에 **1:1 로 배정**한다
  (유사도가 높은 쌍부터 확정 — 두 문서가 같은 첨부를 집는 사고를 막는다).
  선택 결과는 `download_log.csv` 의 `picked_name`·`name_score` 에 남고,
  유사도 0.6 미만은 `status=review` + `reports/needs_review.csv` 로 분리된다
  — **그 문서는 반드시 PDF 를 열어서 내용이 맞는지 확인할 것**.
- HF repo 에는 PDF 가 없다 (csv·md 12개뿐). `--from-hf` 는 향후 repo 에 추가될 경우를 위한 옵션이다.

- `context_type` 라벨이 도메인마다 흔들린다 — medical 은 paragraph 대신 `text`(45건)를 쓴다.
  `fetch_dataset.py` 가 이를 paragraph 로 통일하고 원본은 `context_type_raw` 에 보존한다.
  통일하지 않으면 집계 축이 4개로 쪼개져 도메인 간 비교가 깨진다.
- `target_page_no` 가 비어 있는 문항이 1건 있다. context 를 만들 수 없어 자동 제외되며,
  `fetch_dataset.py` 실행 로그와 `reports/context_missing.csv` 에 qid 가 남는다.
  → 실제 평가 모수는 300 − (페이지 결측) − (PDF 다운로드 실패) 문항이다.

## 모델을 여러 개 비교할 때

세 단계 모두 모델/출력이 파일 단위로 분리된다. `contexts.jsonl` 이 동결돼 있으므로
파싱 비용은 재발생하지 않고, 추가 모델은 실행·채점 비용만 든다.

```bash
# 모델 A
python src/run_model.py --model deepseek/deepseek-v4-flash-0731 --reasoning high
# 모델 B (같은 contexts, 같은 프롬프트)
python src/run_model.py --model google/gemini-3.7-flash --reasoning high

python src/judge.py            # results/raw/*.csv 를 모두 채점 (이미 채점된 건 건너뜀)
python src/report.py           # 모든 run 집계 + 모델 간 비교표
```

| 단계 | 인자 | 출력 |
|---|---|---|
| `run_model.py` | `--model` `--reasoning` `--quantization` `--extra-body` `--out` | `results/raw/openrouter__<벤더-모델>__<reasoning>.csv` |
| `judge.py` | `--input` `--judge-model` `--tag` | `results/scored/<raw 파일명>.csv` |
| `report.py` | `--runs` `--out` | `reports/<out>.md`, `reports/<out>_by_axis.csv` |

- run 파일명에 **벤더까지** 넣는다 (`meta-llama/llama-3.3-70b` 와 `nvidia/llama-3.3-70b`
  처럼 뒷부분이 겹칠 수 있다). 같은 모델을 reasoning 만 바꿔 돌리면 파일이 자동으로 갈린다.
- `judge.py` 는 `results/raw/*.csv` 를 전부 훑고 이미 채점된 문항은 건너뛴다.
  특정 run 만 채점하려면 `--input results/raw/<파일>.csv`.
  judge 를 바꿔 재채점할 때는 `--tag` 로 결과를 분리한다.
- `report.py --runs '*deepseek*'` 처럼 일부만 골라 따로 리포트를 낼 수 있다.
  채점 결과가 2개 이상이면 마지막에 **모델 간 비교표**가 자동으로 붙는다.
- 모델 답변이 바뀌면(재실행 등) judge 는 예전 판정을 재사용하지 않는다
  (qid 가 아니라 qid+답변으로 캐시를 맞춘다).

## reasoning 설정 (하이브리드 모델 주의)

| `--reasoning` | 전송 내용 | 쓰는 경우 |
|---|---|---|
| `none` | 파라미터 미전송 | 모델 기본값을 그대로 쓸 때. **하이브리드 모델은 기본이 thinking on 이라 추론한다** |
| `off` | `{"reasoning": {"enabled": false}}` | 추론을 명시적으로 끌 때 (non-thinking 모델과 조건을 맞출 때) |
| `low`/`medium`/`high` | `{"reasoning": {"effort": …}}` | 추론 강도를 지정할 때 |

`run_model.py` 는 실행 후 실제 발생한 reasoning 토큰을 집계해, 껐다고 했는데 추론이
일어났거나 켰는데 0 이면 경고한다. provider 가 `reasoning.enabled` 를 안 받으면
`--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'` 로 우회한다.

## 모델 간 차이가 유의미한지

`report.py` 는 run 이 2개 이상이면 **짝지은 비교**(같은 문항을 같은 context 로 풀었으므로
McNemar 정확검정)를 함께 낸다. 전체 정확도 차이만 보면 표본이 작을 때 우연을 실력으로
오독하기 쉽다 — 예를 들어 130문항에서 5%p 차이는 p ≈ 0.4 로 판단 불가다.

- `A만 정답` / `B만 정답` 이 엇갈린 문항 수이고, 검정은 이 둘만 본다.
- p ≥ 0.05 면 그 표본으로는 우열을 확정할 수 없다는 뜻이다. 문항을 늘리거나
  (제외 문항 회복), 다른 축(비용·지연·도메인별 강점)으로 결정해야 한다.

## 리포트 집계 축

전체 / domain(5개) / context_type(paragraph·table·image). 제외 문항 수를 함께 명시하고,
`results/scored/` 에 여러 run 이 있으면 모델 간 비교표가 자동으로 붙는다.

**"모델 실패 vs 파서 실패" 구분**: image·table 오답에 대해 정답의 수치·날짜 토큰이 context 안에 있는지 검사한다.
context 에 없으면 파서 실패 의심, 있으면 모델 실패 의심으로 분류하고 목록을 `reports/parser_suspect_*.csv` 로 뽑는다
(1차 신호일 뿐이므로 최종 판단은 육안 확인).

## 받은 PDF 검증 (`tools/review_downloads.py`)

파일명만으로는 엉뚱한 첨부를 걸러낼 수 없어서 내용으로 확인한다.

| 코드 | 뜻 | 조치 |
|---|---|---|
| `match` | 정답 수치가 target 페이지 텍스트에 있음 | 그대로 사용 |
| `page_mismatch` | 정답이 다른 페이지에 있음 | 문서는 맞음. 페이지 지정을 확인 |
| `missing_answer` | 정답이 문서 어디에도 없음 | 다른 파일 의심 |
| `page_out_of_range` | target 페이지가 PDF 페이지 수를 넘음 | 확정적으로 다른 파일 |
| `unverifiable` | 스캔 PDF 등 텍스트 대조 불가 | 첫 페이지를 눈으로 확인 (`--render`) |

`--write-exclusions` 로 만든 `data/excluded_files.txt` 를 `parse_vlm` 과
`build_contexts` 가 읽어 해당 문서 문항을 건너뛴다. 줄을 지우면 다시 포함된다.

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
| 파싱 (Gemini 3.7 Flash) | ~100 페이지 | ~$0.8 |
| 대상 모델 (V4 Flash, high 1 run) | 300 호출 | ~$1–2 |
| Judge (3.7 Flash) | 300 호출 | ~$0.5 |
| 합계 | | ~$7 이내 |

후보 모델 추가 시 모델당 ~$3–5. contexts 가 동결돼 있어 파싱 비용은 재발생하지 않는다.

## 개발용

```bash
python tools/smoke_test.py    # 네트워크 없이 build_contexts / report / 수치 크로스체크 검증
```

작업 루트는 `ALLGANIZE_EVAL_ROOT` 환경변수로 바꿀 수 있다 (기본: 레포 루트).
