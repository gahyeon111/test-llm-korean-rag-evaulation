"""1단계: documents.csv 의 URL 로 PDF 를 내려받는다.

원문 URL 만료 케이스가 있으므로 실패 문서와 그 문서에 걸린 문항 수를 로그로 남긴다.
여기서 나온 실패 문서 = 평가 제외 문항 확정 (스펙 §7 게이트 전 단계).

출력
  data/pdfs/<target_file_name>
  reports/download_log.csv       문서별 성공/실패 + 걸린 문항 수
  reports/excluded_questions.csv 다운로드 실패로 제외되는 문항 목록
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from common import (DATASET_CSV, DOCUMENTS_ALIASES, DOCUMENTS_CSV, PDF_DIR,
                    REPORT_DIR, normalize_columns)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; allganize-rag-eval/1.0)"}


def download_one(url: str, dest: Path, timeout: int, retries: int) -> tuple[str, str]:
    """(status, detail) 반환. status ∈ {ok, skipped, failed}"""
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
            else:
                body = resp.content
                if not body.startswith(b"%PDF"):
                    last_error = "PDF 헤더 아님 (만료 페이지/HTML 응답 의심)"
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(body)
                    return "ok", f"{len(body)} bytes"
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(2 ** attempt)
    return "failed", last_error


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="선행 확인용 N개만 다운로드")
    args = ap.parse_args()

    if not DOCUMENTS_CSV.exists():
        raise SystemExit(f"{DOCUMENTS_CSV} 가 없습니다. 먼저 fetch_dataset.py 를 실행하세요.")

    documents = normalize_columns(pd.read_csv(DOCUMENTS_CSV), DOCUMENTS_ALIASES,
                                  required=["file_name"])
    if "url" not in documents.columns:
        raise SystemExit("documents.csv 에 url 컬럼이 없습니다.")

    dataset = pd.read_csv(DATASET_CSV) if DATASET_CSV.exists() else pd.DataFrame()
    q_per_doc = (dataset.groupby("target_file_name").size().to_dict()
                 if len(dataset) else {})

    rows = []
    docs = documents.head(args.limit) if args.limit else documents
    for i, doc in enumerate(docs.itertuples(index=False), start=1):
        file_name = str(doc.file_name).strip()
        url = str(getattr(doc, "url", "") or "").strip()
        dest = PDF_DIR / file_name
        n_q = int(q_per_doc.get(file_name, 0))

        if dest.exists() and dest.stat().st_size > 0:
            status, detail = "skipped", "이미 존재"
        elif not url or url.lower() == "nan":
            status, detail = "failed", "URL 없음"
        else:
            status, detail = download_one(url, dest, args.timeout, args.retries)

        print(f"[{i}/{len(docs)}] {status:7s} {file_name} ({n_q}문항) {detail}")
        rows.append({
            "domain": getattr(doc, "domain", ""),
            "file_name": file_name,
            "url": url,
            "status": status,
            "detail": detail,
            "n_questions": n_q,
        })

    log = pd.DataFrame(rows)
    log_path = REPORT_DIR / "download_log.csv"
    log.to_csv(log_path, index=False)

    failed = log[log["status"] == "failed"]
    ok = log[log["status"].isin(["ok", "skipped"])]
    print(f"\n성공/보유 {len(ok)}건, 실패 {len(failed)}건 → {log_path}")

    if len(failed) and len(dataset):
        excluded = dataset[dataset["target_file_name"].isin(set(failed["file_name"]))]
        excl_path = REPORT_DIR / "excluded_questions.csv"
        excluded.to_csv(excl_path, index=False)
        print(f"제외 문항 {len(excluded)}건 → {excl_path}")
        print(failed[["file_name", "n_questions", "detail"]].to_string(index=False))
    else:
        (REPORT_DIR / "excluded_questions.csv").write_text(
            ",".join(dataset.columns) + "\n" if len(dataset) else "", encoding="utf-8")
        print("제외 문항 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
