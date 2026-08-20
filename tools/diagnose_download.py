"""다운로드 실패 원인 진단.

1) HF 데이터셋 repo 안에 PDF 원본이 들어있는지 확인 (있으면 URL 없이 바로 받을 수 있다)
2) documents.csv 의 URL 을 몇 건 찔러 보고 실제 응답 정체를 출력
   (HTTP 상태 / content-type / 최종 리다이렉트 URL / 본문 앞부분 / HTML <title>)

실행: python tools/diagnose_download.py
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from common import DOCUMENTS_ALIASES, DOCUMENTS_CSV, normalize_columns  # noqa: E402

REPO_ID = "allganize/RAG-Evaluation-Dataset-KO"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def check_repo_files(repo_id: str) -> None:
    print(f"=== 1. HF repo 파일 목록 ({repo_id}) ===")
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        print(f"조회 실패: {exc}\n")
        return
    ext = Counter(Path(f).suffix.lower() or "(없음)" for f in files)
    print(f"총 {len(files)}개 파일: " + ", ".join(f"{k} {v}" for k, v in ext.most_common(8)))
    pdfs = [f for f in files if f.lower().endswith(".pdf")]
    if pdfs:
        print(f"★ repo 안에 PDF {len(pdfs)}개가 있습니다 → `python src/download_pdfs.py --from-hf` 로 받으세요.")
        for f in pdfs[:5]:
            print(f"   {f}")
    else:
        print("repo 에 PDF 없음 → 원문 URL 로 받아야 합니다.")
        for f in files[:15]:
            print(f"   {f}")
    print()


def probe_urls(n: int, timeout: int) -> None:
    print("=== 2. documents.csv URL 응답 확인 ===")
    if not DOCUMENTS_CSV.exists():
        print(f"{DOCUMENTS_CSV} 없음\n")
        return
    docs = normalize_columns(pd.read_csv(DOCUMENTS_CSV), DOCUMENTS_ALIASES,
                             required=["file_name"])
    if "url" not in docs.columns:
        print("url 컬럼 자체가 없습니다 → documents.csv 에 URL 이 안 들어있는 구조입니다.\n")
        print(docs.head(3).to_string())
        return

    print(f"URL 샘플 {min(n, len(docs))}건:\n")
    for doc in docs.head(n).itertuples(index=False):
        url = str(getattr(doc, "url", "") or "").strip()
        print(f"- {doc.file_name}")
        print(f"  url: {url[:160] or '(빈 값)'}")
        if not url or url.lower() == "nan":
            print()
            continue
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        except requests.RequestException as exc:
            print(f"  요청 실패: {type(exc).__name__}: {exc}\n")
            continue
        body = resp.content
        print(f"  HTTP {resp.status_code} | content-type: {resp.headers.get('content-type', '-')}"
              f" | {len(body)} bytes")
        if resp.url != url:
            print(f"  최종 URL: {resp.url[:160]}")
        print(f"  첫 바이트: {body[:40]!r}")
        if not body.startswith(b"%PDF"):
            title = re.search(rb"<title[^>]*>(.{0,120}?)</title>", body, re.S | re.I)
            if title:
                print(f"  HTML title: {title.group(1).decode('utf-8', 'replace').strip()}")
        else:
            print("  → 정상 PDF")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--n", type=int, default=3, help="URL 을 몇 건 찔러볼지")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()
    check_repo_files(args.repo_id)
    probe_urls(args.n, args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
