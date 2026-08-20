"""1단계: PDF 원본을 확보한다.

documents.csv 의 url 은 대부분 PDF 직링크가 아니라 **게시판 상세페이지**다.
따라서 기본 동작은: 직접 받아보고 → PDF 가 아니면 페이지에서 첨부 링크를 찾아
target 파일명과 가장 잘 맞는 첨부를 받는다 (한 게시글에 첨부가 여러 개인 경우 대응).

  --from-hf   HF 데이터셋 repo 에 PDF 가 있으면 거기서 직접 (2025-08 기준 repo 에는 없음)
  --no-scrape 첨부 스크래핑 없이 URL 직접 다운로드만 시도

원문 URL 만료 케이스가 있으므로 실패 문서와 그 문서에 걸린 문항 수를 로그로 남긴다.
여기서 나온 실패 문서 = 평가 제외 문항 확정 (스펙 §7 게이트 전 단계).

출력
  data/pdfs/<target_file_name>
  reports/download_log.csv       문서별 성공/실패 + 응답 정체 + 걸린 문항 수
  reports/excluded_questions.csv 다운로드 실패로 제외되는 문항 목록
  reports/failed_html/           PDF 가 아닌 응답 본문 (원인 확인용, 최대 --keep-html 건)
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd
import requests

from attachment_scraper import BROWSER_HEADERS, nfc, scrape_pdf
from common import (DATASET_CSV, DOCUMENTS_ALIASES, DOCUMENTS_CSV, PDF_DIR,
                    REPORT_DIR, normalize_columns)

REPO_ID = "allganize/RAG-Evaluation-Dataset-KO"
HEADERS = dict(BROWSER_HEADERS, Accept="application/pdf,application/octet-stream,*/*")
HTML_DIR = REPORT_DIR / "failed_html"
# 파일명 유사도가 이보다 낮으면 다른 첨부를 받았을 수 있어 검수 대상으로 표시한다.
LOW_SCORE = 0.6


def hf_pdf_index(repo_id: str) -> dict[str, str]:
    """HF repo 안의 PDF 를 {파일명(NFC): repo 내 경로} 로 인덱싱."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, repo_type="dataset")
    return {nfc(Path(f).name): f for f in files if f.lower().endswith(".pdf")}


def fetch_from_hf(repo_id: str, path_in_repo: str, dest: Path) -> tuple[str, dict]:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type="dataset")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    ok = dest.read_bytes()[:4] == b"%PDF"
    return ("ok" if ok else "failed"), {
        "detail": f"HF repo: {path_in_repo}" if ok else "HF repo 파일이 PDF 가 아님",
        "http_status": "", "content_type": "", "final_url": f"hf://{repo_id}/{path_in_repo}",
    }


def download_one(session: requests.Session, url: str, dest: Path, timeout: int,
                 retries: int, html_budget: list[int]) -> tuple[str, dict]:
    """(status, meta) 반환. status ∈ {ok, failed}

    HTTP 200 인데 PDF 가 아닌 응답은 재시도해도 결과가 같으므로 즉시 실패 처리한다.
    """
    meta = {"detail": "", "http_status": "", "content_type": "", "final_url": ""}
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        except requests.RequestException as exc:
            meta["detail"] = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(2 ** attempt)
            continue

        meta.update({
            "http_status": resp.status_code,
            "content_type": resp.headers.get("content-type", ""),
            "final_url": resp.url if resp.url != url else "",
        })
        body = resp.content

        if resp.status_code != 200:
            meta["detail"] = f"HTTP {resp.status_code}"
            if attempt < retries and resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            return "failed", meta

        if body.startswith(b"%PDF"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            meta["detail"] = f"{len(body)} bytes"
            return "ok", meta

        # 200 + non-PDF → 만료 안내 페이지/로그인 리다이렉트/차단. 재시도 무의미.
        meta["detail"] = f"PDF 아님 ({meta['content_type'] or '?'}, {len(body)} bytes)"
        if html_budget[0] > 0:
            HTML_DIR.mkdir(parents=True, exist_ok=True)
            (HTML_DIR / (dest.stem[:80] + ".html")).write_bytes(body[:200_000])
            html_budget[0] -= 1
        return "failed", meta

    return "failed", meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-hf", action="store_true",
                    help="HF 데이터셋 repo 에 있는 PDF 를 우선 사용 (없는 문서만 URL 로 폴백)")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="선행 확인용 N개만 다운로드")
    ap.add_argument("--keep-html", type=int, default=3,
                    help="PDF 아닌 응답 본문을 몇 건까지 저장할지")
    ap.add_argument("--no-scrape", action="store_true",
                    help="게시판 페이지에서 첨부 링크를 찾지 않음")
    ap.add_argument("--max-candidates", type=int, default=8,
                    help="상세페이지에서 시도할 첨부 링크 최대 개수")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="문서 간 대기(초) — 기관 사이트 부담 완화")
    args = ap.parse_args()

    if not DOCUMENTS_CSV.exists():
        raise SystemExit(f"{DOCUMENTS_CSV} 가 없습니다. 먼저 fetch_dataset.py 를 실행하세요.")

    documents = normalize_columns(pd.read_csv(DOCUMENTS_CSV), DOCUMENTS_ALIASES,
                                  required=["file_name"])
    if "url" not in documents.columns:
        documents["url"] = ""

    hf_index: dict[str, str] = {}
    if args.from_hf:
        try:
            hf_index = hf_pdf_index(args.repo_id)
            print(f"HF repo PDF {len(hf_index)}개 확인")
        except Exception as exc:  # noqa: BLE001
            print(f"HF repo 조회 실패 ({exc}) → URL 다운로드로 진행")
        if not hf_index:
            print("HF repo 에 PDF 가 없습니다 → URL 다운로드로 진행")

    dataset = pd.read_csv(DATASET_CSV) if DATASET_CSV.exists() else pd.DataFrame()
    q_per_doc = ({nfc(k): v for k, v in
                  dataset.groupby("target_file_name").size().to_dict().items()}
                 if len(dataset) else {})

    html_budget = [args.keep_html]
    session = requests.Session()
    rows = []
    docs = documents.head(args.limit) if args.limit else documents
    for i, doc in enumerate(docs.itertuples(index=False), start=1):
        file_name = nfc(doc.file_name)
        url = str(getattr(doc, "url", "") or "").strip()
        dest = PDF_DIR / file_name
        n_q = int(q_per_doc.get(file_name, 0))
        meta = {"detail": "", "http_status": "", "content_type": "", "final_url": "",
                "picked_name": "", "picked_url": "", "name_score": "", "n_candidates": ""}

        if dest.exists() and dest.stat().st_size > 0 and dest.read_bytes()[:4] == b"%PDF":
            status, meta["detail"] = "skipped", "이미 존재"
        elif file_name in hf_index:
            status, meta = fetch_from_hf(args.repo_id, hf_index[file_name], dest)
        elif not url or url.lower() == "nan":
            status, meta["detail"] = "failed", "URL 없음"
        else:
            status, meta = download_one(session, url, dest, args.timeout,
                                        args.retries, html_budget)
            if status == "failed" and not args.no_scrape and meta.get("http_status") == 200:
                got = scrape_pdf(session, url, file_name, timeout=args.timeout,
                                 max_candidates=args.max_candidates)
                if got:
                    body, scrape_meta = got
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(body)
                    status = "ok"
                    meta.update({
                        "detail": f"첨부 스크래핑: {scrape_meta['picked_name']} "
                                  f"(유사도 {scrape_meta['score']}, {len(body)} bytes)",
                        "picked_name": scrape_meta["picked_name"],
                        "picked_url": scrape_meta["picked_url"],
                        "name_score": scrape_meta["score"],
                        "n_candidates": scrape_meta["candidates"],
                    })
                    if scrape_meta["score"] < LOW_SCORE:
                        meta["detail"] += "  ⚠️ 파일명 유사도 낮음 — 다른 첨부일 수 있음"
                else:
                    meta["detail"] += " / 페이지에서 PDF 첨부를 못 찾음"
            if args.sleep:
                time.sleep(args.sleep)

        print(f"[{i}/{len(docs)}] {status:7s} {file_name} ({n_q}문항) {meta['detail']}")
        rows.append({"domain": getattr(doc, "domain", ""), "file_name": file_name,
                     "url": url, "status": status, "n_questions": n_q, **meta})

    log = pd.DataFrame(rows)
    log_path = REPORT_DIR / "download_log.csv"
    log.to_csv(log_path, index=False)

    failed = log[log["status"] == "failed"]
    ok = log[log["status"].isin(["ok", "skipped"])]
    print(f"\n성공/보유 {len(ok)}건, 실패 {len(failed)}건 → {log_path}")

    excl_path = REPORT_DIR / "excluded_questions.csv"
    if len(failed) and len(dataset):
        failed_names = set(failed["file_name"])
        excluded = dataset[dataset["target_file_name"].map(nfc).isin(failed_names)]
        excluded.to_csv(excl_path, index=False)
        print(f"제외 문항 {len(excluded)}건 / 전체 {len(dataset)}건 → {excl_path}")
        print(failed[["file_name", "n_questions", "detail"]].head(10).to_string(index=False))
        if html_budget[0] < args.keep_html:
            print(f"\nPDF 아닌 응답 본문을 {HTML_DIR} 에 저장했습니다 — 열어서 원인을 확인하세요.")
    else:
        excl_path.write_text(",".join(dataset.columns) + "\n" if len(dataset) else "",
                             encoding="utf-8")
        print("제외 문항 없음")

    low = log[(log["status"] == "ok") &
              (pd.to_numeric(log["name_score"], errors="coerce") < LOW_SCORE)]
    if len(low):
        print(f"\n⚠️ 파일명 유사도가 낮은 {len(low)}건 — 엉뚱한 첨부를 받았을 수 있으니 "
              "PDF 를 열어 확인하세요:")
        print(low[["file_name", "picked_name", "name_score"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
