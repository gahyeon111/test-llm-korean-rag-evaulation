"""첨부를 못 찾은 페이지(reports/failed_html/*.html)의 링크 패턴을 뜯어본다.

새 사이트 패턴을 attachment_scraper 에 추가하려면 이 출력이 필요하다.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import REPORT_DIR  # noqa: E402

A_TAG_RE = re.compile(r"<a\b[^>]*>", re.I)
ATTR_RE = re.compile(r"""(href|onclick|data-\w+)\s*=\s*["']([^"']*)["']""", re.I)
JS_FUNC_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(\s*['\"][^)]{0,120}\)")
FILEISH_RE = re.compile(r"(file|down|atch|attach|pdf|첨부)", re.I)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(REPORT_DIR / "failed_html"))
    ap.add_argument("--max-lines", type=int, default=25)
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.html"))
    if not files:
        raise SystemExit(f"{args.dir} 에 HTML 이 없습니다.")

    for path in files:
        html = path.read_text(encoding="utf-8", errors="replace")
        print("=" * 70)
        print(f"{path.name}  ({len(html):,} chars)")
        print("=" * 70)

        print("\n[파일/다운로드로 보이는 속성]")
        shown = 0
        for tag in A_TAG_RE.findall(html):
            attrs = [(k, v) for k, v in ATTR_RE.findall(tag) if FILEISH_RE.search(v)]
            for key, value in attrs:
                print(f"  {key}={value[:150]}")
                shown += 1
            if shown >= args.max_lines:
                break
        if not shown:
            print("  (없음 — 첨부가 iframe/JS 로 나중에 그려질 수 있음)")

        print("\n[JS 함수 호출]")
        calls = {m.group(0)[:120] for m in JS_FUNC_RE.finditer(html)
                 if FILEISH_RE.search(m.group(1))}
        for call in list(calls)[:args.max_lines] or ["  (없음)"]:
            print(f"  {call}")

        print("\n['pdf' 가 들어간 줄]")
        lines = [ln.strip()[:150] for ln in html.splitlines() if ".pdf" in ln.lower()]
        for line in lines[:args.max_lines] or ["  (없음)"]:
            print(f"  {line}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
