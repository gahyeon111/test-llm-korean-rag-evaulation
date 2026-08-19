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
    args = ap.parse_args()

    files = sorted(Path(args.scored_dir).glob("*.csv"))
    if not files:
        raise SystemExit("채점 결과가 없습니다. judge.py 를 먼저 실행하세요.")

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
    for path in files:
        df = pd.read_csv(path).fillna("")
        run = path.stem
        model = df["model"].iloc[0] if "model" in df.columns and len(df) else run
        effort = df["reasoning_effort"].iloc[0] if "reasoning_effort" in df.columns and len(df) else ""
        providers = (df.loc[df.get("status", "") == "ok", "provider"].value_counts()
                     if "provider" in df.columns else pd.Series(dtype=int))
        n_failed = int((df.get("status", pd.Series(dtype=str)) == "failed").sum())
        n_judge_failed = int((~df["verdict"].isin(["O", "X"])).sum())

        lines += [f"## 1. {run}", "",
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
    axis_df.to_csv(REPORT_DIR / "summary_by_axis.csv", index=False)

    if axis_df["run"].nunique() > 1:
        pivot = axis_df.pivot_table(index=["axis", "group"], columns="run",
                                    values="accuracy").round(4).reset_index()
        lines += ["## 2. 모델 간 비교 (accuracy)", "", md_table(pivot), ""]

    summary_path = REPORT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {summary_path}, {REPORT_DIR / 'summary_by_axis.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
