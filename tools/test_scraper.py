"""첨부 스크래퍼 오프라인 테스트 — 네트워크 없이 가짜 세션으로 검증.

한 게시글에 PDF 첨부가 여러 개일 때 target 파일명에 맞는 것을 고르는지 확인한다.
(bok.or.kr 처럼 서로 다른 두 문서가 같은 상세페이지 URL 을 갖는 실제 케이스)
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attachment_scraper import (assign_attachments, candidate_urls,  # noqa: E402
                                collect_attachments, content_disposition_filename,
                                name_similarity)

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
    def content(self):
        return self._body

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

    # 4. %20 이 남은 파일명도 풀린다 (유사도가 깎이면 엉뚱한 첨부를 고른다)
    assert content_disposition_filename(
        'attachment; filename="2.%20통합신용정보%20운영규약.pdf"') == "2. 통합신용정보 운영규약.pdf"

    # 5. 한 페이지의 첨부는 한 번만 받아서 목록으로 돌려준다
    cache = Path(tempfile.mkdtemp(prefix="attach-cache-"))
    session = StubSession()
    StubSession.calls.clear()
    attachments, meta = collect_attachments(session, PAGE_URL, cache_dir=cache)
    assert meta["n_attachments"] == 2, meta
    assert sorted(a["size"] for a in attachments) == sorted([len(PDF_A), len(PDF_B)])

    # 6. 같은 페이지에 걸린 두 문서 → 서로 다른 첨부가 배정돼야 한다
    targets = ["2024년 3월_2. 통합신용정보 운영규약.pdf",
               "2024년 3월_3. 신용정보 관리규약.pdf"]
    assigned = assign_attachments(targets, attachments)
    assert len(assigned) == 2, assigned
    picked = {ti: attachments[ai]["path"].read_bytes() for ti, (ai, _) in assigned.items()}
    assert picked[0] != picked[1], "두 문서에 같은 첨부가 배정됨"
    assert picked[0] == PDF_A and picked[1] == PDF_B, "배정이 서로 뒤바뀜"
    assert all(score > 0.9 for _, score in assigned.values()), assigned

    # 7. 두 번째 호출은 캐시를 쓰므로 첨부를 다시 받지 않는다
    StubSession.calls.clear()
    collect_attachments(session, PAGE_URL, cache_dir=cache)
    assert StubSession.calls.count(PAGE_URL) == 1, StubSession.calls
    assert not any("FileDown" in u for u in StubSession.calls), \
        f"캐시가 있는데 첨부를 다시 받음: {StubSession.calls}"

    # 8. PDF 가 없는 페이지면 빈 목록
    empty, _ = collect_attachments(session, "https://example.com/none", cache_dir=cache)
    assert empty == []

    shutil.rmtree(cache, ignore_errors=True)
    print("스크래퍼 테스트 통과 (링크수집·파일명·유사도·1:1 배정·캐시 8개 항목)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
