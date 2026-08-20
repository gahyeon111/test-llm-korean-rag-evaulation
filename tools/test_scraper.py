"""첨부 스크래퍼 오프라인 테스트 — 네트워크 없이 가짜 세션으로 검증.

한 게시글에 PDF 첨부가 여러 개일 때 target 파일명에 맞는 것을 고르는지 확인한다.
(bok.or.kr 처럼 서로 다른 두 문서가 같은 상세페이지 URL 을 갖는 실제 케이스)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attachment_scraper import (candidate_urls, content_disposition_filename,  # noqa: E402
                                name_similarity, scrape_pdf)

PAGE_URL = "https://www.bok.or.kr/portal/bbs/B0000156/view.do?nttId=10082951&menuNo=200067"
PAGE_HTML = """
<html><body>
  <a href="/portal/main.do">홈</a>
  <a href="/portal/cmmn/file/fileDown.do?atchFileId=FILE_001&amp;fileSn=0">
     2. 통합신용정보 운영규약.pdf (1.2MB)</a>
  <a href="javascript:fn_egov_downFile('FILE_002','1')">3. 신용정보 관리규약.pdf</a>
  <a href="/portal/bbs/list.do">목록</a>
</body></html>
"""
PDF_A = b"%PDF-1.4 aaa" + b"0" * 500
PDF_B = b"%PDF-1.7 bbb" + b"1" * 500

RESPONSES = {
    PAGE_URL: (200, {"content-type": "text/html;charset=utf-8"}, PAGE_HTML.encode()),
    "https://www.bok.or.kr/portal/cmmn/file/fileDown.do?atchFileId=FILE_001&fileSn=0":
        (200, {"content-type": "application/pdf",
               "content-disposition": 'attachment; filename="2. 통합신용정보 운영규약.pdf"'}, PDF_A),
    "https://www.bok.or.kr/cmm/fms/FileDown.do?atchFileId=FILE_002&fileSn=1":
        (200, {"content-type": "application/pdf",
               "content-disposition": "attachment; filename*=UTF-8''3.%20%EC%8B%A0%EC%9A%A9"
                                      "%EC%A0%95%EB%B3%B4%20%EA%B4%80%EB%A6%AC%EA%B7%9C%EC%95%BD.pdf"},
         PDF_B),
}


class StubResponse:
    def __init__(self, url, status, headers, body):
        self.url, self.status_code, self.headers, self._body = url, status, headers, body

    @property
    def text(self):
        return self._body.decode("utf-8")

    encoding = "utf-8"
    apparent_encoding = "utf-8"

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class StubSession:
    calls: list[str] = []

    def get(self, url, **kwargs):
        StubSession.calls.append(url)
        status, headers, body = RESPONSES.get(url, (404, {}, b""))
        return StubResponse(url, status, headers, body)


def main() -> int:
    # 1. 링크 수집: href 형태 + javascript 호출 형태 모두 잡히는지
    urls = candidate_urls(PAGE_HTML, PAGE_URL)
    assert any("FILE_001" in u for u in urls), urls
    assert any("FILE_002" in u for u in urls), urls
    assert not any("list.do" in u for u in urls), f"본문 링크가 후보에 섞임: {urls}"

    # 2. Content-Disposition 파일명 파싱 (RFC5987 / 평문)
    assert content_disposition_filename(
        "attachment; filename*=UTF-8''%ED%95%9C%EA%B8%80.pdf") == "한글.pdf"
    assert content_disposition_filename('attachment; filename="a b.pdf"') == "a b.pdf"

    # 3. 파일명 유사도
    assert name_similarity("2024년 3월_3. 신용정보 관리규약.pdf",
                           "3. 신용정보 관리규약.pdf") > 0.9
    assert name_similarity("2024년 3월_3. 신용정보 관리규약.pdf",
                           "2. 통합신용정보 운영규약.pdf") < 0.9

    # 4. 같은 페이지 URL, 다른 target → 각각 맞는 첨부를 골라야 한다
    session = StubSession()
    got_b = scrape_pdf(session, PAGE_URL, "2024년 3월_3. 신용정보 관리규약.pdf")
    assert got_b and got_b[0] == PDF_B, "3번 문서가 엉뚱한 첨부를 받음"
    assert got_b[1]["score"] > 0.9, got_b[1]

    got_a = scrape_pdf(session, PAGE_URL, "2024년 3월_2. 통합신용정보 운영규약.pdf")
    assert got_a and got_a[0] == PDF_A, "2번 문서가 엉뚱한 첨부를 받음"

    # 5. PDF 가 없는 페이지면 None
    empty = scrape_pdf(session, "https://example.com/none", "x.pdf")
    assert empty is None

    print("스크래퍼 테스트 통과 (첨부 선택 5개 항목)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
