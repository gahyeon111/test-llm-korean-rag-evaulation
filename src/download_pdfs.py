"""1단계: PDF 원본을 확보한다.

documents.csv 의 url 은 PDF 직링크가 아니라 **기관 게시판의 상세페이지**이고,
PDF 는 그 안의 첨부파일이다. 게다가 한 게시글에 첨부가 여러 개라 서로 다른 문서가
같은 url 을 갖는 경우가 있다.

그래서 url 단위로 묶어서 처리한다:
  1) 페이지의 PDF 첨부를 **한 번만** 전부 받아 cache/attachments/ 에 저장
  2) 그 url 에 걸린 target 파일명들에 첨부를 **1:1 로 배정** (유사도 높은 쌍부터)
  3) 유사도가 낮으면 저장은 하되 검수 대상으로 표시

출력
  data/pdfs/<target_file_name>
  cache/attachments/            페이지에서 받은 첨부 원본 (재실행 시 재사용)
  reports/download_log.csv      문서별 상태 + 어떤 첨부를 골랐는지 + 유사도
  reports/needs_review.csv      ★ 유사도가 낮아 눈으로 확인해야 하는 문서
  reports/excluded_questions.csv 실패 문서에 걸린 제외 문항
  reports/failed_html/          첨부를 못 찾은 페이지 본문 (원인 확인용)
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pandas as pd
import requests

from attachment_scraper import assign_attachments, collect_attachments, nfc
from common import (DATASET_CSV, DOCUMENTS_ALIASES, DOCUMENTS_CSV, REPORT_DIR,
                    ROOT, normalize_columns, pdf_path, safe_filename)

ATTACH_CACHE = ROOT / "cache" / "attachments"
HTML_DIR = REPORT_DIR / "failed_html"
LOW_SCORE = 0.6   # 이보다 낮으면 다른 첨부일 수 있음 → 검수 대상


def is_pdf(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and path.read_bytes()[:4] == b"%PDF"


def save_failed_html(session: requests.Session, url: str, name: str, budget: list[int],
                     timeout: int) -> None:
    """첨부를 못 찾은 페이지 본문을 남겨 패턴을 추가할 수 있게 한다."""
    if budget[0] <= 0:
        return
    try:
        resp = session.get(url, timeout=timeout)
    except requests.RequestException:
        return
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    (HTML_DIR / safe_filename(Path(name).stem + ".html")).write_bytes(resp.content[:400_000])
    budget[0] -= 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0, help="선행 확인용 N개 문서만")
    ap.add_argument("--max-candidates", type=int, default=10,
                    help="상세페이지에서 시도할 첨부 링크 최대 개수")
    ap.add_argument("--min-score", type=float, default=LOW_SCORE,
                    help="이 유사도 미만이면 검수 대상으로 표시")
    ap.add_argument("--keep-html", type=int, default=20,
                    help="첨부를 못 찾은 페이지 본문을 몇 건까지 저장할지")
    ap.add_argument("--follow-links", type=int, default=3,
                    help="목록페이지로 보일 때 따라 들어갈 상세페이지 수 (0=사용 안 함)")
    ap.add_argument("--sleep", type=float, default=1.0, help="페이지 간 대기(초)")
    ap.add_argument("--force", action="store_true", help="이미 받은 PDF 도 다시 받기")
    args = ap.parse_args()

    if not DOCUMENTS_CSV.exists():
        raise SystemExit(f"{DOCUMENTS_CSV} 가 없습니다. 먼저 fetch_dataset.py 를 실행하세요.")

    documents = normalize_columns(pd.read_csv(DOCUMENTS_CSV), DOCUMENTS_ALIASES,
                                  required=["file_name"])
    if "url" not in documents.columns:
        documents["url"] = ""
    documents["file_name"] = documents["file_name"].map(nfc)
    documents["url"] = documents["url"].fillna("").astype(str).str.strip()
    if args.limit:
        documents = documents.head(args.limit)

    dataset = pd.read_csv(DATASET_CSV) if DATASET_CSV.exists() else pd.DataFrame()
    q_per_doc = ({nfc(k): int(v) for k, v in
                  dataset.groupby("target_file_name").size().to_dict().items()}
                 if len(dataset) else {})

    session = requests.Session()
    html_budget = [args.keep_html]
    rows: list[dict] = []

    groups = documents.groupby("url", sort=False)
    print(f"문서 {len(documents)}건 / 상세페이지 {len(groups)}개\n")

    for gi, (url, group) in enumerate(groups, start=1):
        targets = list(group["file_name"])
        base = {t: {"domain": group.loc[group["file_name"] == t, "domain"].iloc[0]
                    if "domain" in group.columns else "",
                    "url": url, "n_questions": q_per_doc.get(t, 0)} for t in targets}
        head = f"[{gi}/{len(groups)}] {url[:80] or '(URL 없음)'} — 문서 {len(targets)}건"

        pending = [t for t in targets if args.force or not is_pdf(pdf_path(t))]
        for done in (t for t in targets if t not in pending):
            print(f"{head}\n    skipped  {done} (이미 존재)")
            rows.append(base[done] | {"file_name": done, "status": "skipped",
                                      "detail": "이미 존재"})
        if not pending:
            continue
        print(head)

        if not url or url.lower() == "nan":
            for t in pending:
                print(f"    failed   {t} — URL 없음")
                rows.append(base[t] | {"file_name": t, "status": "failed",
                                       "detail": "URL 없음"})
            continue

        attachments, meta = collect_attachments(
            session, url, cache_dir=ATTACH_CACHE, targets=pending,
            timeout=args.timeout, max_candidates=args.max_candidates,
            follow_links=args.follow_links, min_score=args.min_score)
        print(f"    페이지 {meta['page_status']} | 첨부 후보 {meta['n_candidates']} → "
              f"PDF {meta['n_attachments']}개")
        for followed in meta["followed"]:
            print(f"      ↳ 목록페이지로 보고 상세 진입: {followed['title'][:50]} "
                  f"(PDF {followed['n_attachments']}개)")

        if not attachments:
            save_failed_html(session, url, pending[0], html_budget, args.timeout)
            for t in pending:
                print(f"    failed   {t} — PDF 첨부를 못 찾음")
                rows.append(base[t] | {
                    "file_name": t, "status": "failed",
                    "detail": f"PDF 첨부 없음 (page {meta['page_status']}, "
                              f"후보 {meta['n_candidates']}, "
                              f"상세진입 {len(meta['followed'])})",
                    "n_attachments": 0})
            if args.sleep:
                time.sleep(args.sleep)
            continue

        assigned = assign_attachments(pending, attachments)
        for ti, t in enumerate(pending):
            if ti not in assigned:
                print(f"    failed   {t} — 배정할 첨부가 부족 (첨부 {len(attachments)}개)")
                rows.append(base[t] | {"file_name": t, "status": "failed",
                                       "detail": f"첨부 부족 ({len(attachments)}개)",
                                       "n_attachments": len(attachments)})
                continue
            ai, score = assigned[ti]
            att = attachments[ai]
            dest = pdf_path(t)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(att["path"], dest)

            single = len(attachments) == 1 and len(pending) == 1
            status = "ok" if score >= args.min_score else "review"
            mark = "" if status == "ok" else "  ⚠️ 검수 필요"
            print(f"    {status:8s} {t}\n             ← {att['name']} "
                  f"(유사도 {score}, {att['size']:,} bytes){mark}")
            rows.append(base[t] | {
                "file_name": t, "status": status,
                "detail": f"{att['size']} bytes",
                "picked_name": att["name"], "picked_url": att["url"],
                "name_score": score, "n_attachments": len(attachments),
                "single_attachment": single,
            })
        if args.sleep:
            time.sleep(args.sleep)

    log = pd.DataFrame(rows)
    for col in ["picked_name", "picked_url", "name_score", "n_attachments",
                "single_attachment"]:
        if col not in log.columns:
            log[col] = ""
    log_path = REPORT_DIR / "download_log.csv"
    log.to_csv(log_path, index=False)

    ok = log[log["status"].isin(["ok", "skipped"])]
    review = log[log["status"] == "review"]
    failed = log[log["status"] == "failed"]
    print(f"\n{'='*60}\n확보 {len(ok)}건 / 검수필요 {len(review)}건 / 실패 {len(failed)}건"
          f" → {log_path}")

    # 같은 첨부가 두 문서에 배정됐는지 (배정 로직상 없어야 하지만 최종 확인)
    picked = log[log["picked_url"].astype(str).str.len() > 0]
    dupes = picked[picked.duplicated("picked_url", keep=False)]
    if len(dupes):
        print(f"\n❗ 같은 첨부가 여러 문서에 배정됨 {len(dupes)}건 — 반드시 확인:")
        print(dupes[["file_name", "picked_name", "name_score"]].to_string(index=False))

    if len(review):
        review_path = REPORT_DIR / "needs_review.csv"
        review.to_csv(review_path, index=False)
        print(f"\n⚠️ 파일명이 안 맞는 {len(review)}건 — PDF 를 열어 내용이 맞는지 확인하세요"
              f" → {review_path}")
        print(review[["file_name", "picked_name", "name_score",
                      "n_attachments"]].to_string(index=False))

    excl_path = REPORT_DIR / "excluded_questions.csv"
    if len(failed) and len(dataset):
        excluded = dataset[dataset["target_file_name"].map(nfc).isin(set(failed["file_name"]))]
        excluded.to_csv(excl_path, index=False)
        print(f"\n실패 문서 {len(failed)}건 → 제외 문항 {len(excluded)}건 / 전체 {len(dataset)}건")
        print(failed[["file_name", "n_questions", "detail"]].to_string(index=False))
        if html_budget[0] < args.keep_html:
            print(f"\n첨부를 못 찾은 페이지 본문 → {HTML_DIR}")
            print("  `python tools/inspect_html.py` 로 링크 패턴을 확인할 수 있습니다.")
    elif len(dataset):
        excl_path.write_text(",".join(dataset.columns) + "\n", encoding="utf-8")
        print("\n제외 문항 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
