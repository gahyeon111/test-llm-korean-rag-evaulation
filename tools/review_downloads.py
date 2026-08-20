"""받은 PDF 가 정말 그 문서인지 자동 교차검증한다.

파일명 유사도(name_score)만으로는 "엉뚱한 첨부"를 가릴 수 없어서, PDF 내용으로 확인한다.

  1) 페이지 수 검사 — dataset 의 target_page_no 가 PDF 페이지 수를 넘으면 다른 파일이다 (확정적)
  2) 정답 대조   — target 페이지 텍스트에 정답의 수치·날짜가 있는지 (강한 신호)
  3) 첫 페이지 미리보기 — 위 둘로 판정이 안 될 때 눈으로 볼 근거

출력
  reports/download_review.md   문서별 판정 + 첫 페이지 미리보기
  reports/download_review.csv  판정 요약
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402
import pymupdf  # noqa: E402

from attachment_scraper import name_similarity  # noqa: E402
from common import DATASET_CSV, PDF_DIR, REPORT_DIR, pdf_path  # noqa: E402
from judge import extract_dates, extract_numbers  # noqa: E402


def answer_tokens(text: str) -> set[str]:
    return set(extract_numbers(text)) | set(extract_dates(text))


def find_answer_pages(page_texts: list[str], target_answer: str,
                      page_base: int) -> list[int]:
    """정답의 수치가 문서의 어느 페이지에 있는지. 페이지 오프셋 vs 다른 파일 구분용."""
    tokens = answer_tokens(target_answer)
    if not tokens:
        return []
    return [i + page_base for i, text in enumerate(page_texts)
            if tokens & answer_tokens(text)]


def coverage_report(dataset: pd.DataFrame) -> list[str]:
    """문항이 가리키는 파일과 실제로 받은 파일이 어긋나 있는지 확인.

    파일명이 조금만 달라도 문항이 통째로 제외되므로, 비슷한 이름이 디스크에
    있으면 후보로 제시한다.
    """
    lines = ["## 파일명 대조", ""]
    on_disk = sorted(p.name for p in PDF_DIR.glob("*.pdf"))
    per_file = dataset.groupby("target_file_name").size()

    missing = [(name, int(n)) for name, n in per_file.items()
               if not pdf_path(name).exists()]
    lines.append(f"- 문항이 가리키는 문서 {len(per_file)}개 중 PDF 확보 "
                 f"{len(per_file) - len(missing)}개, 미확보 {len(missing)}개 "
                 f"(문항 {sum(n for _, n in missing)}건)")

    suggestions = []
    for name, n in missing:
        best = max(((name_similarity(name, disk), disk) for disk in on_disk),
                   default=(0.0, ""))
        if best[0] >= 0.7:
            suggestions.append((name, n, best[1], round(best[0], 3)))
    if suggestions:
        lines += ["", "### ⚠️ 이름만 다른 파일이 디스크에 있을 수 있음", "",
                  "| 문항이 찾는 파일 | 문항 수 | 디스크의 비슷한 파일 | 유사도 |",
                  "|---|---|---|---|"]
        lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in suggestions]

    orphan = [p.name for p in PDF_DIR.glob("*.pdf")
              if not any(pdf_path(f).name == p.name for f in per_file.index)]
    if orphan:
        lines += ["", f"- 받았지만 문항이 없는 문서 {len(orphan)}개 (평가와 무관): "
                      + ", ".join(orphan[:10])]
    return lines


def page_text(doc: pymupdf.Document, page_no: int, page_base: int) -> str:
    index = int(page_no) - page_base
    if index < 0 or index >= doc.page_count:
        return ""
    return doc.load_page(index).get_text()


def answer_hit(text: str, target_answer: str) -> bool | None:
    """정답의 수치·날짜가 그 페이지 텍스트에 있는지. 대조 불가면 None."""
    tokens = set(extract_numbers(target_answer)) | set(extract_dates(target_answer))
    if not tokens or not text.strip():
        return None
    found = set(extract_numbers(text)) | set(extract_dates(text))
    return bool(tokens & found)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-review", action="store_true",
                    help="status=review 인 문서만 (기본: 받은 문서 전부)")
    ap.add_argument("--page-base", type=int, default=1)
    ap.add_argument("--preview-chars", type=int, default=400)
    ap.add_argument("--render", action="store_true",
                    help="확인이 필요한 문서의 첫 페이지를 PNG 로 저장 (스캔 PDF 눈검사용)")
    args = ap.parse_args()

    log_path = REPORT_DIR / "download_log.csv"
    if not log_path.exists():
        raise SystemExit(f"{log_path} 가 없습니다. download_pdfs.py 를 먼저 실행하세요.")
    log = pd.read_csv(log_path).fillna("")
    dataset = pd.read_csv(DATASET_CSV).fillna("")

    targets = log[log["status"].isin(["review"] if args.only_review
                                     else ["ok", "skipped", "review"])]
    rows, sections = [], []

    for item in targets.itertuples(index=False):
        file_name = str(item.file_name)
        path = pdf_path(file_name)
        questions = dataset[dataset["target_file_name"] == file_name]
        row = {"file_name": file_name, "status": item.status,
               "picked_name": getattr(item, "picked_name", ""),
               "name_score": getattr(item, "name_score", ""),
               "n_questions": len(questions), "page_count": "",
               "max_target_page": "", "pages_ok": "", "answer_hits": "",
               "answer_checked": "", "answer_elsewhere": "", "verdict": ""}

        if not path.exists():
            row["verdict"] = "파일 없음"
            rows.append(row)
            continue

        try:
            doc = pymupdf.open(path)
        except Exception as exc:  # noqa: BLE001
            row["verdict"] = f"열기 실패: {exc}"
            rows.append(row)
            continue

        with doc:
            row["page_count"] = doc.page_count
            pages = pd.to_numeric(questions["target_page_no"], errors="coerce").dropna()
            row["max_target_page"] = int(pages.max()) if len(pages) else ""
            row["pages_ok"] = (int(pages.max()) - args.page_base < doc.page_count
                               if len(pages) else True)

            page_texts = [doc.load_page(i).get_text() for i in range(doc.page_count)]
            hits = checked = 0
            elsewhere: list[str] = []
            for q in questions.itertuples(index=False):
                page_no = pd.to_numeric(q.target_page_no, errors="coerce")
                if pd.isna(page_no):
                    continue
                result = answer_hit(page_text(doc, page_no, args.page_base),
                                    q.target_answer)
                if result is None:
                    continue
                checked += 1
                hits += int(result)
                if not result:
                    found = find_answer_pages(page_texts, q.target_answer,
                                              args.page_base)
                    elsewhere.append(
                        f"{q.qid}: target p{int(page_no)} → 정답 수치가 "
                        + (f"p{', p'.join(str(x) for x in found[:5])} 에 있음"
                           if found else "문서 어디에도 없음"))
            row["answer_hits"], row["answer_checked"] = hits, checked
            row["answer_elsewhere"] = " / ".join(elsewhere[:5])
            preview = " ".join(page_texts[0].split()) if page_texts else ""
            render_target = doc.load_page(0) if (args.render and doc.page_count) else None
            png = render_target.get_pixmap(dpi=110).tobytes("png") if render_target else None

        if row["pages_ok"] is False:
            row["verdict"] = "❌ 다른 파일 (target 페이지가 PDF 범위 밖)"
        elif checked and hits / checked >= 0.5:
            row["verdict"] = "✅ 내용 일치 (정답 수치가 target 페이지에 있음)"
        elif checked and any("문서 어디에도 없음" not in e for e in elsewhere):
            row["verdict"] = "🔁 문서는 맞는데 페이지가 어긋남 — 확인 필요"
        elif checked:
            row["verdict"] = "⚠️ 정답 수치가 문서에 없음 — 다른 파일 의심"
        else:
            row["verdict"] = "❓ 자동 대조 불가 (스캔 PDF 등) — 눈으로 확인"
        rows.append(row)

        if row["verdict"].startswith("✅") and str(item.status) == "ok":
            continue   # 문제없는 건 상세 섹션에서 생략
        if png:
            page_dir = REPORT_DIR / "review_pages"
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / (Path(file_name).stem[:60] + ".png")).write_bytes(png)
        sections.append(
            f"### {file_name}\n\n"
            f"- 판정: **{row['verdict']}**\n"
            f"- 받은 첨부: `{row['picked_name']}` (파일명 유사도 {row['name_score']})\n"
            f"- 페이지 수 {row['page_count']} / 문항이 가리키는 최대 페이지 "
            f"{row['max_target_page']} / 문항 {row['n_questions']}건\n"
            f"- 정답 대조: {row['answer_hits']}/{row['answer_checked']}건 일치\n"
            + (f"- 불일치 문항: {row['answer_elsewhere']}\n" if row.get("answer_elsewhere") else "")
            + "\n"
            f"첫 페이지 미리보기:\n\n> {preview[:args.preview_chars] or '(텍스트 없음 — 스캔 PDF)'}\n")

    result = pd.DataFrame(rows)
    result.to_csv(REPORT_DIR / "download_review.csv", index=False)
    md = ["# 다운로드 검수 리포트", "",
          f"대상 {len(result)}건", "",
          "## 판정 요약", "", "```",
          result["verdict"].value_counts().to_string(), "```", ""]
    md += coverage_report(dataset) + ["", "## 확인이 필요한 문서", ""] + sections
    (REPORT_DIR / "download_review.md").write_text("\n".join(md), encoding="utf-8")

    print(result["verdict"].value_counts().to_string())
    bad = result[(~result["verdict"].str.startswith("✅")) & (result["n_questions"] > 0)]
    skip = result[(~result["verdict"].str.startswith("✅")) & (result["n_questions"] == 0)]
    if len(skip):
        print(f"\n(문항이 0건이라 볼 필요 없는 문서 {len(skip)}개는 제외)")
    if len(bad):
        print(f"\n실제로 확인이 필요한 문서 {len(bad)}건 (문항 {bad['n_questions'].sum()}건):")
        print(bad[["file_name", "n_questions", "verdict"]].to_string(index=False))
    for line in coverage_report(dataset):
        if line.startswith(("- ", "| ")) or line.startswith("### "):
            print(line)
    print(f"\n상세 → {REPORT_DIR / 'download_review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
