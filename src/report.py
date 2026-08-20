"""6단계: 채점 결과 집계 리포트.

집계 축: 전체 / domain / context_type (+ 모델이 여러 개면 나란히 비교)
추가: 제외 문항 수, image·table 오답의 "모델 실패 vs 파서 실패" 후보 목록.

출력
  reports/summary.md
  reports/summary_by_axis.csv
  reports/parser_suspect_<run>.csv
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import CONTEXTS_JSONL, REPORT_DIR, RESULT_SCORED_DIR, read_jsonl
from judge import extract_dates, extract_numbers


def accuracy_table(df: pd.DataFrame, axis: str | None) -> pd.DataFrame:
    valid = df[df["verdict"].isin(["O", "X"])].assign(
        _is_correct=lambda d: (d["verdict"] == "O").astype(int))
    if axis:
        grouped = valid.groupby(axis)
        table = pd.DataFrame({"n": grouped.size(), "correct": grouped["_is_correct"].sum()})
    else:
        table = pd.DataFrame({"n": [len(valid)],
                              "correct": [int(valid["_is_correct"].sum())]},
                             index=["전체"])
    table["accuracy"] = (table["correct"] / table["n"]).round(4)
    return table.reset_index().rename(columns={"index": axis or "구분"})


def answer_support(contexts: dict[str, str], row: pd.Series) -> str:
    """정답의 수치 토큰이 context 안에 있는지 — 파서 실패 판별의 1차 신호."""
    context = contexts.get(str(row["qid"]), "")
    if not context:
        return "NO_CONTEXT"
    targets = set(extract_numbers(row.get("target_answer", ""))) | \
        set(extract_dates(row.get("target_answer", "")))
    if not targets:
        return "NA"
    found = set(extract_numbers(context)) | set(extract_dates(context))
    missing = targets - found
    if not missing:
        return "IN_CONTEXT"
    return "MISSING" if len(missing) == len(targets) else "PARTIAL"


def mcnemar_p(b: int, c: int) -> float:
    """McNemar 정확검정(양측). b, c 는 두 모델의 판정이 엇갈린 문항 수.

    같은 문항을 같은 context 로 풀었으므로 짝지은 비교가 맞다.
    전체 정확도 차이만 보면 표본이 작을 때 우연을 실력으로 오독하기 쉽다.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def pairwise_comparison(runs: dict[str, pd.DataFrame]) -> list[str]:
    """run 두 개씩 짝지어 문항 단위로 비교한다."""
    names = list(runs)
    if len(names) < 2:
        return []
    lines = ["### 짝지은 비교 (같은 문항 기준)", "",
             "| A | B | A만 정답 | B만 정답 | 둘 다 정답 | 둘 다 오답 | 차이 | p (McNemar) |",
             "|---|---|---|---|---|---|---|---|"]
    for i, a in enumerate(names):
        for b_name in names[i + 1:]:
            da = {str(r["qid"]): r["verdict"] for _, r in runs[a].iterrows()}
            db = {str(r["qid"]): r["verdict"] for _, r in runs[b_name].iterrows()}
            shared = [q for q in da if q in db and da[q] in "OX" and db[q] in "OX"]
            only_a = sum(1 for q in shared if da[q] == "O" and db[q] == "X")
            only_b = sum(1 for q in shared if da[q] == "X" and db[q] == "O")
            both = sum(1 for q in shared if da[q] == db[q] == "O")
            neither = len(shared) - only_a - only_b - both
            diff = (only_a - only_b) / len(shared) if shared else 0
            p = mcnemar_p(only_a, only_b)
            verdict = "유의미" if p < 0.05 else "표본으로는 판단 불가"
            lines.append(f"| {a} | {b_name} | {only_a} | {only_b} | {both} | {neither} | "
                         f"{diff:+.1%} | {p:.3f} ({verdict}) |")
    lines += ["", "p ≥ 0.05 면 두 모델의 차이를 이 표본(문항 수)으로는 확정할 수 없다는 뜻이다. "
              "그 경우 정확도 숫자만으로 우열을 결론짓지 말 것.", ""]
    return lines


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored-dir", default=str(RESULT_SCORED_DIR))
    ap.add_argument("--contexts", default=str(CONTEXTS_JSONL))
    ap.add_argument("--suspect-samples", type=int, default=20)
    ap.add_argument("--runs", default="*.csv",
                    help="집계할 채점 결과 파일 패턴 (예: '*deepseek*'). 기본: 전부")
    ap.add_argument("--out", default="summary",
                    help="리포트 파일명 접두사 → reports/<out>.md, <out>_by_axis.csv")
    args = ap.parse_args()

    pattern = args.runs if args.runs.endswith(".csv") else f"{args.runs}.csv"
    files = sorted(Path(args.scored_dir).glob(pattern))
    if not files:
        raise SystemExit(f"채점 결과가 없습니다 (패턴: {pattern}). judge.py 를 먼저 실행하세요.")
    print(f"집계 대상 {len(files)}개 run: " + ", ".join(f.stem for f in files))

    contexts_path = Path(args.contexts)
    contexts = ({r["qid"]: r["context"] for r in read_jsonl(contexts_path)}
                if contexts_path.exists() else {})
    n_contexts = len(contexts)

    excluded_path = REPORT_DIR / "excluded_questions.csv"
    n_excluded_dl = 0
    if excluded_path.exists() and excluded_path.stat().st_size > 0:
        try:
            n_excluded_dl = len(pd.read_csv(excluded_path))
        except pd.errors.EmptyDataError:
            n_excluded_dl = 0
    missing_ctx_path = REPORT_DIR / "context_missing.csv"
    n_missing_ctx = len(pd.read_csv(missing_ctx_path)) if missing_ctx_path.exists() else 0

    lines = ["# allganize-rag-eval 결과 리포트", "",
             f"생성: {datetime.now(timezone.utc).isoformat()}", "",
             "## 0. 평가 범위", "",
             f"- 평가 대상 문항(contexts.jsonl): **{n_contexts}건**",
             f"- 다운로드 실패로 제외: {n_excluded_dl}건",
             f"- 파싱 실패/빈 결과로 제외: {n_missing_ctx}건",
             "- Oracle context 방식(retrieval 미평가), judge 는 자체 판정이므로 "
             "Allganize 공식 리더보드와 직접 비교 불가 — 후보 모델 간 상대 비교 전용", ""]

    axis_rows = []
    run_frames: dict[str, pd.DataFrame] = {}
    for section_no, path in enumerate(files, start=1):
        df = pd.read_csv(path).fillna("")
        run_frames[path.stem] = df
        run = path.stem
        model = df["model"].iloc[0] if "model" in df.columns and len(df) else run
        effort = df["reasoning_effort"].iloc[0] if "reasoning_effort" in df.columns and len(df) else ""
        providers = (df.loc[df.get("status", "") == "ok", "provider"].value_counts()
                     if "provider" in df.columns else pd.Series(dtype=int))
        n_failed = int((df.get("status", pd.Series(dtype=str)) == "failed").sum())
        n_judge_failed = int((~df["verdict"].isin(["O", "X"])).sum())

        lines += [f"## {section_no}. {run}", "",
                  f"- 모델: `{model}` / reasoning: `{effort}` / temperature: "
                  f"{df['temperature'].iloc[0] if 'temperature' in df.columns and len(df) else '-'}",
                  f"- 라우팅된 provider: {', '.join(f'{k}({v})' for k, v in providers.items()) or '-'}"
                  + ("  ⚠️ provider 혼합 — 양자화 오염 확인 필요" if len(providers) > 1 else ""),
                  f"- 모델 호출 실패: {n_failed}건 / judge 판정 실패: {n_judge_failed}건",
                  f"- judge: `{df['judge_model'].iloc[0] if 'judge_model' in df.columns and len(df) else '-'}`"
                  f" (prompt `{df['judge_prompt_version'].iloc[0] if 'judge_prompt_version' in df.columns and len(df) else '-'}`)",
                  ""]

        for axis, title in [(None, "전체"), ("domain", "도메인별"), ("context_type", "context_type별")]:
            if axis and axis not in df.columns:
                continue
            table = accuracy_table(df, axis)
            table["accuracy"] = (table["accuracy"] * 100).round(1).astype(str) + "%"
            lines += [f"### {title}", "", md_table(table), ""]
            for row in accuracy_table(df, axis).to_dict("records"):
                axis_rows.append({"run": run, "model": model, "reasoning": effort,
                                  "axis": axis or "overall",
                                  "group": row[axis] if axis else "전체",
                                  "n": row["n"], "correct": row["correct"],
                                  "accuracy": row["accuracy"]})

        # 수치 크로스체크 불일치 (judge 오판 후보)
        if "numeric_match" in df.columns:
            disagree = df[((df["verdict"] == "O") & (df["numeric_match"] == "N")) |
                          ((df["verdict"] == "X") & (df["numeric_match"] == "Y"))]
            lines += [f"### judge × 수치 크로스체크 불일치: {len(disagree)}건",
                      "", "judge 오판 후보. `reports/judge_review_*.csv` 에서 수동 대조.", ""]

        # image/table 오답: 모델 실패 vs 파서 실패
        wrong = (df[(df["verdict"] == "X") & df["context_type"].isin(["image", "table"])].copy()
                 if "context_type" in df.columns else df.iloc[0:0].copy())
        if len(wrong):
            wrong["answer_support"] = wrong.apply(lambda r: answer_support(contexts, r), axis=1)
            wrong["context_len"] = wrong["qid"].map(lambda q: len(contexts.get(str(q), "")))
            wrong["failure_hint"] = wrong["answer_support"].map({
                "MISSING": "파서 실패 의심 (정답 수치가 context 에 없음)",
                "PARTIAL": "파서 실패 의심 (정답 수치 일부 누락)",
                "IN_CONTEXT": "모델 실패 의심 (정답 수치가 context 에 존재)",
                "NA": "판별 불가 (정답에 수치 없음 — 수동 확인)",
                "NO_CONTEXT": "context 없음",
            })
            cols = ["qid", "domain", "context_type", "question", "target_answer",
                    "model_answer", "judge_reason", "answer_support", "context_len",
                    "failure_hint"]
            suspect_path = REPORT_DIR / f"parser_suspect_{run}.csv"
            wrong[[c for c in cols if c in wrong.columns]].to_csv(suspect_path, index=False)

            counts = wrong["failure_hint"].value_counts().reset_index()
            counts.columns = ["구분", "건수"]
            lines += [f"### image·table 오답 {len(wrong)}건의 실패 원인 후보", "",
                      md_table(counts), "",
                      f"상세 목록: `{suspect_path.relative_to(REPORT_DIR.parent)}` "
                      f"(상위 {args.suspect_samples}건은 육안 확인 권장)", ""]

    axis_df = pd.DataFrame(axis_rows)
    axis_df.to_csv(REPORT_DIR / f"{args.out}_by_axis.csv", index=False)

    if axis_df["run"].nunique() > 1:
        pivot = axis_df.pivot_table(index=["axis", "group"], columns="run",
                                    values="accuracy").round(4).reset_index()
        lines += [f"## {len(files) + 1}. 모델 간 비교 (accuracy)", "",
                  md_table(pivot), ""]
        lines += pairwise_comparison(run_frames)

    summary_path = REPORT_DIR / f"{args.out}.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {summary_path}, {REPORT_DIR / f'{args.out}_by_axis.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
