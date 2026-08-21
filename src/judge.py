"""5단계: LLM judge 로 O/X 이진 채점 + 수치 정규식 크로스체크.

judge 프롬프트는 src/prompts/judge_v*.txt 로 분리·버전 관리한다.

출력
  results/scored/<원본파일명>            raw + 판정 컬럼
  reports/judge_review_<원본명>.csv      수동 대조용 샘플(30~50건)
"""
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import (OpenRouterError, PROMPT_DIR, RESULT_RAW_DIR, RESULT_SCORED_DIR,
                    REPORT_DIR, chat_completion, parse_json_block, require_api_key)

DEFAULT_JUDGE = "google/gemini-3.7-flash"
DEFAULT_PROMPT_VERSION = "judge_v1"

# 금액·비율·수량 (1,234.5 / 12.3% / 3천억 등의 숫자부) 와 날짜(2024년 3월 / 2024-03-01)
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
DATE_RE = re.compile(r"(\d{4})[-.\s년]+\s*(\d{1,2})(?:[-.\s월]+\s*(\d{1,2}))?")


def extract_numbers(text: str) -> list[str]:
    """비교 가능한 정규형 숫자 토큰 목록. 콤마 제거 + 소수점 뒤 0 제거."""
    tokens = []
    for raw in NUM_RE.findall(str(text or "")):
        value = raw.replace(",", "")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        if value:
            tokens.append(value)
    return tokens


def extract_dates(text: str) -> list[str]:
    out = []
    for year, month, day in DATE_RE.findall(str(text or "")):
        parts = [str(int(year)), str(int(month))]
        if day:
            parts.append(str(int(day)))
        out.append("-".join(parts))
    return out


def numeric_cross_check(target: str, answer: str) -> tuple[str, int, int]:
    """정답의 수치·날짜 토큰이 모델 답변에 모두 등장하는지. (판정, 총개수, 일치개수)"""
    targets = set(extract_numbers(target)) | set(extract_dates(target))
    if not targets:
        return "NA", 0, 0
    found = set(extract_numbers(answer)) | set(extract_dates(answer))
    hit = len(targets & found)
    return ("Y" if hit == len(targets) else "N"), len(targets), hit


def judge_one(template: str, row: dict, model: str, api_key: str, retries: int) -> dict:
    prompt = template.format(
        question=row.get("question", ""),
        target_answer=row.get("target_answer", ""),
        model_answer=row.get("model_answer", "") or "(빈 응답)",
    )
    result = chat_completion(model, [{"role": "user", "content": prompt}],
                             temperature=0.0, retries=retries, api_key=api_key,
                             response_format={"type": "json_object"})
    parsed = parse_json_block(result["text"]) or {}
    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in {"O", "X"}:
        verdict = "O" if verdict in {"TRUE", "CORRECT", "정답"} else "PARSE_ERROR"
    return {
        "verdict": verdict,
        "judge_reason": str(parsed.get("reason", ""))[:500] or result["text"][:200],
        "judge_latency_s": result["latency_s"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="", help="raw CSV 경로 (미지정 시 results/raw/*.csv 전부)")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--review-samples", type=int, default=40, help="수동 대조 샘플 수")
    ap.add_argument("--tag", default="",
                    help="출력 파일명 접미사. judge 를 바꿔 재채점할 때 구분용 "
                         "(예: --judge-model X --tag judgeX)")
    ap.add_argument("--force", action="store_true", help="기존 채점 무시하고 재판정")
    ap.add_argument("--save-every", type=int, default=10,
                    help="N건마다 중간 저장 (중단돼도 이어서 채점 가능)")
    ap.add_argument("--abort-after", type=int, default=3,
                    help="연속 N회 실패하면 중단 (judge 모델 ID 오류 등)")
    args = ap.parse_args()

    api_key = require_api_key()
    template = (PROMPT_DIR / f"{args.prompt_version}.txt").read_text(encoding="utf-8")
    inputs = [Path(args.input)] if args.input else sorted(RESULT_RAW_DIR.glob("*.csv"))
    if not inputs:
        raise SystemExit("채점할 raw 결과가 없습니다.")

    for raw_path in inputs:
        df = pd.read_csv(raw_path).fillna("")
        if args.limit:
            df = df.head(args.limit)
        out_path = RESULT_SCORED_DIR / (
            f"{raw_path.stem}__{args.tag}.csv" if args.tag else raw_path.name)

        prev: dict[str, dict] = {}
        if out_path.exists() and not args.force:
            # 모델 답변이 바뀌었으면(재실행 등) 예전 판정을 재사용하면 안 된다.
            prev = {(str(r["qid"]), str(r.get("model_answer", ""))): r
                    for r in pd.read_csv(out_path).fillna("").to_dict("records")
                    if str(r.get("verdict")) in {"O", "X"}}
            print(f"이어하기: 기존 판정 {len(prev)}건 재사용 ({out_path.name})")

        print(f"\n=== {raw_path.name} ({len(df)}건) ===")
        scored: list[dict] = []
        consecutive_failures = 0

        def save(rows_so_far: list[dict], path=out_path) -> pd.DataFrame:
            frame = pd.DataFrame(rows_so_far)
            frame.to_csv(path, index=False)
            return frame
        try:
            for i, row in enumerate(df.to_dict("records"), start=1):
                qid = str(row["qid"])
                nm, n_total, n_hit = numeric_cross_check(row.get("target_answer", ""),
                                                         row.get("model_answer", ""))
                base = row | {
                    "numeric_match": nm, "numeric_targets": n_total, "numeric_hits": n_hit,
                    "judge_model": args.judge_model, "judge_prompt_version": args.prompt_version,
                    "judged_at": datetime.now(timezone.utc).isoformat(),
                }

                if str(row.get("status")) != "ok" or not str(row.get("model_answer")).strip():
                    scored.append(base | {"verdict": "X", "judge_reason": "모델 응답 실패/빈 응답",
                                          "judge_latency_s": 0})
                    continue
                cached = prev.get((qid, str(row.get("model_answer", ""))))
                if cached:
                    scored.append(base | {k: cached[k] for k in
                                          ("verdict", "judge_reason", "judge_latency_s")
                                          if k in cached})
                    continue

                try:
                    verdict = judge_one(template, row, args.judge_model, api_key, args.retries)
                except OpenRouterError as exc:
                    verdict = {"verdict": "JUDGE_FAILED", "judge_reason": str(exc)[:300],
                               "judge_latency_s": 0}
                    consecutive_failures += 1
                    if consecutive_failures >= args.abort_after:
                        raise SystemExit(
                            f"\n연속 {consecutive_failures}회 실패 — 설정 문제로 보고 중단합니다.\n"
                            f"마지막 오류: {str(exc)[:400]}")
                else:
                    consecutive_failures = 0
                scored.append(base | verdict)
                print(f"[{i}/{len(df)}] {qid} … {verdict['verdict']} (numeric={nm})")
                if args.save_every and i % args.save_every == 0:
                    save(scored)
                if args.sleep:
                    time.sleep(args.sleep)

        except KeyboardInterrupt:
            out = save(scored)
            print(f"\n중단됨 — 여기까지 {len(out)}건을 {out_path} 에 저장했습니다. "
                  "같은 명령을 다시 실행하면 이어서 채점합니다.")
            return 130

        out = save(scored)

        n_o = int((out["verdict"] == "O").sum())
        n_valid = int(out["verdict"].isin(["O", "X"]).sum())
        acc = n_o / n_valid if n_valid else 0.0
        print(f"저장: {out_path} — O {n_o}/{n_valid} ({acc:.1%})")

        # judge 신뢰도 확인용 샘플: 크로스체크 불일치 우선, 부족분은 오답에서 채움
        mismatch = out[((out["verdict"] == "O") & (out["numeric_match"] == "N")) |
                       ((out["verdict"] == "X") & (out["numeric_match"] == "Y")) |
                       (~out["verdict"].isin(["O", "X"]))]
        wrong = out[(out["verdict"] == "X") & (~out["qid"].isin(mismatch["qid"]))]
        take = max(0, args.review_samples - len(mismatch))
        sample = pd.concat([mismatch, wrong.head(take)])
        cols = ["qid", "domain", "context_type", "question", "target_answer",
                "model_answer", "verdict", "judge_reason", "numeric_match",
                "numeric_targets", "numeric_hits"]
        review_path = REPORT_DIR / f"judge_review_{raw_path.stem}.csv"
        sample[[c for c in cols if c in sample.columns]].to_csv(review_path, index=False)
        print(f"수동 대조 샘플 {len(sample)}건 → {review_path}")
        print("★ 이 샘플을 눈으로 대조해 judge 신뢰도를 확인한 뒤 본채점을 확정하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
