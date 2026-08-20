"""4단계: 동결된 contexts 로 평가 대상 모델을 호출한다.

요청 옵션은 스펙 §2 고정값: temperature 0, provider fp8 고정(allow_fallbacks=false),
reasoning 은 nested 형식(reasoning.effort)만 사용.

출력
  results/raw/<endpoint>__<model>__<reasoning>.csv
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import (CONTEXTS_JSONL, OpenRouterError, PROMPT_DIR, RESULT_RAW_DIR,
                    build_provider, chat_completion, read_jsonl, require_api_key)

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
PROMPT_VERSION = "answer_v1"
COLUMNS = ["qid", "domain", "context_type", "question", "target_answer",
           "target_file", "target_page", "model_answer", "status", "error",
           "provider", "latency_s", "attempts", "prompt_tokens", "completion_tokens",
           "reasoning_tokens", "total_tokens", "model", "reasoning_effort",
           "temperature", "prompt_version", "contexts_sha256", "run_at"]


def model_slug(model: str) -> str:
    """모델 ID → 파일명 조각. 벤더까지 남긴다
    (meta-llama/llama-3.3-70b 와 nvidia/llama-3.3-70b 처럼 뒷부분이 겹칠 수 있다)."""
    return model.replace("/", "-").replace(":", "-")


def output_path(model: str, reasoning: str, endpoint: str = "openrouter") -> Path:
    return RESULT_RAW_DIR / f"{endpoint}__{model_slug(model)}__{reasoning}.csv"


def contexts_hash(path: Path) -> str:
    meta = path.parent / (path.name + ".meta.json")
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("sha256", "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reasoning", default="high",
                    choices=["none", "off", "low", "medium", "high"],
                    help="none=파라미터 미전송(모델 기본값 그대로), "
                         "off=추론 끄기(reasoning.enabled=false), "
                         "low/medium/high=reasoning.effort")
    ap.add_argument("--contexts", default=str(CONTEXTS_JSONL))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--quantization", default="fp8", help="빈 문자열이면 provider 고정 해제")
    ap.add_argument("--provider-order", default="", help="쉼표 구분 provider 우선순위")
    ap.add_argument("--allow-fallbacks", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="드라이런용 N문항")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--extra-body", default="",
                    help='요청 본문에 합칠 JSON. provider 별 옵션용 '
                         '(예: \'{"chat_template_kwargs":{"enable_thinking":false}}\')')
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true", help="기존 결과 무시하고 전부 재호출")
    ap.add_argument("--abort-after", type=int, default=3,
                    help="연속 N회 실패하면 중단 (모델 ID·provider 설정 오류 등)")
    args = ap.parse_args()

    api_key = require_api_key()
    contexts_path = Path(args.contexts)
    if not contexts_path.exists():
        raise SystemExit(f"{contexts_path} 가 없습니다. build_contexts.py 를 먼저 실행하세요.")

    records = read_jsonl(contexts_path)
    if args.limit:
        records = records[:args.limit]

    template = (PROMPT_DIR / f"{PROMPT_VERSION}.txt").read_text(encoding="utf-8")
    provider = build_provider(
        quantizations=[args.quantization] if args.quantization else [],
        order=[p.strip() for p in args.provider_order.split(",")] if args.provider_order else [],
        allow_fallbacks=args.allow_fallbacks,
    )
    reasoning_effort = None if args.reasoning == "none" else args.reasoning
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    out_path = Path(args.out) if args.out else output_path(args.model, args.reasoning)

    done: dict[str, dict] = {}
    if out_path.exists() and not args.force:
        prev = pd.read_csv(out_path).to_dict("records")
        done = {str(r["qid"]): r for r in prev if str(r.get("status")) == "ok"}
        print(f"이어하기: 기존 성공 {len(done)}건 재사용 ({out_path.name})")

    ctx_sha = contexts_hash(contexts_path)
    run_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    n_failed = 0
    consecutive_failures = 0

    for i, rec in enumerate(records, start=1):
        qid = rec["qid"]
        if qid in done:
            rows.append(done[qid])
            continue

        prompt = template.format(context=rec["context"], question=rec["question"])
        base = {k: rec.get(k, "") for k in
                ["qid", "domain", "context_type", "question", "target_answer",
                 "target_file", "target_page"]}
        base |= {
            "model": args.model, "reasoning_effort": args.reasoning,
            "temperature": args.temperature, "prompt_version": PROMPT_VERSION,
            "contexts_sha256": ctx_sha, "run_at": run_at,
        }

        try:
            result = chat_completion(
                args.model,
                [{"role": "user", "content": prompt}],
                temperature=args.temperature,
                reasoning_effort=reasoning_effort,
                provider=provider or None,
                extra_body=extra_body,
                retries=args.retries,
                api_key=api_key,
            )
        except OpenRouterError as exc:
            n_failed += 1
            rows.append(base | {"model_answer": "", "status": "failed",
                                "error": str(exc)[:500], "provider": "",
                                "latency_s": 0, "attempts": args.retries,
                                "prompt_tokens": 0, "completion_tokens": 0,
                                "reasoning_tokens": 0, "total_tokens": 0})
            print(f"[{i}/{len(records)}] {qid} … 실패: {str(exc)[:300]}")
            consecutive_failures += 1
            if consecutive_failures >= args.abort_after:
                raise SystemExit(
                    f"\n연속 {consecutive_failures}회 실패 — 설정 문제로 보고 중단합니다.\n"
                    f"마지막 오류: {str(exc)[:400]}")
            continue

        rows.append(base | {
            "model_answer": result["text"], "status": "ok", "error": "",
            "provider": result["provider"], "latency_s": result["latency_s"],
            "attempts": result["attempts"], **result["usage"],
        })
        consecutive_failures = 0
        print(f"[{i}/{len(records)}] {qid} … ok "
              f"({result['provider']}, {result['latency_s']}s, "
              f"reasoning {result['usage']['reasoning_tokens']}tok)")
        if args.sleep:
            time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[COLUMNS].to_csv(out_path, index=False)

    print(f"\n저장: {out_path} ({len(df)}건, 실패 {n_failed}건)")
    routed = df.loc[df["status"] == "ok", "provider"].value_counts()
    print("[라우팅된 provider]")
    print(routed.to_string() if len(routed) else "  (없음)")
    if len(routed) > 1:
        print("경고: provider 가 여러 개로 섞였습니다 — 양자화 오염 여부를 확인하세요.")

    # 하이브리드 모델은 기본이 thinking on 이라, 끈 줄 알았는데 계속 추론하는 경우가 있다.
    ok_rows = df[df["status"] == "ok"]
    reasoning_tokens = pd.to_numeric(ok_rows.get("reasoning_tokens", 0),
                                     errors="coerce").fillna(0)
    n_reasoned = int((reasoning_tokens > 0).sum())
    if args.reasoning in ("none", "off") and n_reasoned:
        print(f"\n⚠️ reasoning={args.reasoning} 인데 {n_reasoned}/{len(ok_rows)}건에서 "
              f"reasoning 토큰이 발생했습니다 (평균 {reasoning_tokens.mean():.0f}).")
        print("   하이브리드 모델의 기본 thinking 이 그대로 적용됐을 수 있습니다 — "
              "`--reasoning off` 또는")
        print("   `--extra-body '{\"chat_template_kwargs\":{\"enable_thinking\":false}}'` "
              "로 끄고 --force 로 다시 돌리세요.")
    elif args.reasoning not in ("none", "off") and not n_reasoned:
        print(f"\n⚠️ reasoning={args.reasoning} 인데 reasoning 토큰이 0입니다 — "
              "이 모델/provider 가 reasoning 을 지원하는지 확인하세요.")
    else:
        print(f"reasoning 토큰 발생: {n_reasoned}/{len(ok_rows)}건 "
              f"(평균 {reasoning_tokens.mean():.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
