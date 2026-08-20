"""첨부 스크래퍼 오프라인 테스트 — 네트워크 없이 가짜 세션으로 검증.

실제 실행에서 나온 세 가지 케이스를 그대로 재현한다.
  A. 한 게시글에 첨부 여러 개 (bok) → 문서별로 서로 다른 첨부를 배정해야 한다
  B. down.do?...&file_seq=1 형태 링크 (kofia) → 후보로 잡혀야 한다
  C. 목록페이지 (fsc) → 제목이 맞는 상세페이지로 한 번 더 들어가야 한다
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attachment_scraper import (assign_attachments, candidate_urls,  # noqa: E402
                                collect_attachments, content_disposition_filename,
                                follow_candidates, name_similarity)

# ---- A. 한 게시글, 첨부 2개 ------------------------------------------------
BOK_URL = "https://www.bok.or.kr/portal/bbs/B0000156/view.do?nttId=10082951&menuNo=200067"
BOK_HTML = """
<html><body>
  <a href="/portal/main.do">홈</a>
  <a href="/portal/cmmn/file/fileDown.do?atchFileId=FILE_001&amp;fileSn=0">
     2. 통합신용정보 운영 규약.pdf (1.2MB)</a>
  <a href="javascript:fn_egov_downFile('FILE_002','1')">3. 향후 통합신용정보 방향.pdf</a>
  <a href="/portal/bbs/list.do">목록</a>
</body></html>
"""
PDF_A = b"%PDF-1.4 aaa" + b"0" * 500
PDF_B = b"%PDF-1.7 bbb" + b"1" * 500

# ---- B. kofia: ./down.do?brd_id=...&file_seq=1 ----------------------------
KOFIA_URL = "https://www.kofia.or.kr/brd/m_52/view.do?seq=252"
KOFIA_HTML = """
<html><body>
  <a href="./list.do?brd_id=www_default">목록</a>
  <a href="./down.do?brd_id=www_default&amp;seq=252&amp;data_tp=A&amp;file_seq=1">
     *2019 제1회 증시분석 자료집_최종*.pdf</a>
</body></html>
"""
PDF_KOFIA = b"%PDF-1.5 kofia" + b"2" * 300

# ---- C. fsc: 목록페이지 → 상세페이지 --------------------------------------
FSC_LIST_URL = "https://www.fsc.go.kr/po010101?srchCtgry=1&curPage=1"
FSC_LIST_HTML = """
<html><body>
  <ul>
    <li><a href="/no010101/80123">금융위, 핀테크 투자 생태계 활성화 나선다</a></li>
    <li><a href="/no010101/80999">전혀 다른 보도자료 제목</a></li>
  </ul>
</body></html>
"""
FSC_DETAIL_URL = "https://www.fsc.go.kr/no010101/80123"
FSC_DETAIL_HTML = """
<html><body>
  <a href="/comm/getFile?srvcId=BBSTY1&amp;fileTy=ATTACH&amp;fileNo=1">
     240409(보도자료) 금융위 핀테크 투자 생태계 활성화 나선다.pdf</a>
</body></html>
"""
PDF_FSC = b"%PDF-1.6 fsc" + b"3" * 400

RESPONSES = {
    BOK_URL: (200, {"content-type": "text/html;charset=utf-8"}, BOK_HTML.encode()),
    "https://www.bok.or.kr/portal/cmmn/file/fileDown.do?atchFileId=FILE_001&fileSn=0":
        (200, {"content-type": "application/pdf",
               "content-disposition": 'attachment; filename="2.%20통합신용정보%20운영%20규약.pdf"'},
         PDF_A),
    "https://www.bok.or.kr/cmm/fms/FileDown.do?atchFileId=FILE_002&fileSn=1":
        (200, {"content-type": "application/pdf",
               "content-disposition": "attachment; filename*=UTF-8''3.%20%ED%96%A5%ED%9B%84"
                                      "%20%ED%86%B5%ED%95%A9%EC%8B%A0%EC%9A%A9%EC%A0%95%EB%B3%B4"
                                      "%20%EB%B0%A9%ED%96%A5.pdf"},
         PDF_B),
    KOFIA_URL: (200, {"content-type": "text/html;charset=utf-8"}, KOFIA_HTML.encode()),
    "https://www.kofia.or.kr/brd/m_52/down.do?brd_id=www_default&seq=252&data_tp=A&file_seq=1":
        (200, {"content-type": "application/octet-stream"}, PDF_KOFIA),
    FSC_LIST_URL: (200, {"content-type": "text/html;charset=utf-8"}, FSC_LIST_HTML.encode()),
    FSC_DETAIL_URL: (200, {"content-type": "text/html;charset=utf-8"}, FSC_DETAIL_HTML.encode()),
    "https://www.fsc.go.kr/comm/getFile?srvcId=BBSTY1&fileTy=ATTACH&fileNo=1":
        (200, {"content-type": "application/pdf"}, PDF_FSC),
}


class StubResponse:
    encoding = "utf-8"
    apparent_encoding = "utf-8"

    def __init__(self, url, status, headers, body):
        self.url, self.status_code, self.headers, self._body = url, status, headers, body

    @property
    def content(self):
        return self._body

    @property
    def text(self):
        return self._body.decode("utf-8")

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
    cache = Path(tempfile.mkdtemp(prefix="attach-cache-"))
    session = StubSession()

    # 1. 링크 수집: href / javascript 호출 모두, 네비게이션 링크는 제외
    urls = [c["url"] for c in candidate_urls(BOK_HTML, BOK_URL)]
    assert any("FILE_001" in u for u in urls) and any("FILE_002" in u for u in urls), urls
    assert not any("list.do" in u or "main.do" in u for u in urls), urls

    # 2. Content-Disposition 파일명 (RFC5987 / 평문 / %20 잔여)
    assert content_disposition_filename(
        "attachment; filename*=UTF-8''%ED%95%9C%EA%B8%80.pdf") == "한글.pdf"
    assert content_disposition_filename(
        'attachment; filename="2.%20통합신용정보%20운영.pdf"') == "2. 통합신용정보 운영.pdf"

    # 3. 유사도
    assert name_similarity("2024년 3월_3. 향후 통합신용정보 방향.pdf",
                           "3. 향후 통합신용정보 방향.pdf") > 0.9
    assert name_similarity("2024년 3월_3. 향후 통합신용정보 방향.pdf",
                           "2. 통합신용정보 운영 규약.pdf") < 0.9

    # 4. A: 같은 페이지에 걸린 두 문서 → 서로 다른 첨부가 배정돼야 한다
    attachments, meta = collect_attachments(session, BOK_URL, cache_dir=cache)
    assert meta["n_attachments"] == 2, meta
    targets = ["2024년 3월_2. 통합신용정보 운영 규약.pdf",
               "2024년 3월_3. 향후 통합신용정보 방향.pdf"]
    assigned = assign_attachments(targets, attachments)
    picked = {ti: attachments[ai]["path"].read_bytes() for ti, (ai, _) in assigned.items()}
    assert picked[0] != picked[1], "두 문서에 같은 첨부가 배정됨"
    assert picked[0] == PDF_A and picked[1] == PDF_B, "배정이 서로 뒤바뀜"
    assert all(score > 0.9 for _, score in assigned.values()), assigned

    # 5. 캐시: 두 번째 호출은 첨부를 다시 받지 않는다
    StubSession.calls.clear()
    collect_attachments(session, BOK_URL, cache_dir=cache)
    assert not any("FileDown" in u for u in StubSession.calls), StubSession.calls

    # 6. B(kofia): down.do?...&file_seq= 형태도 후보로 잡혀야 한다
    kofia_target = "*2019 제1회 증시분석 자료집_최종*.pdf"
    attachments, meta = collect_attachments(session, KOFIA_URL, cache_dir=cache,
                                            targets=[kofia_target])
    assert meta["n_attachments"] == 1, f"kofia 첨부를 못 찾음: {meta}"
    assert attachments[0]["path"].read_bytes() == PDF_KOFIA
    # 서버가 파일명을 안 주면 링크 텍스트를 파일명으로 쓴다
    assert name_similarity(kofia_target, attachments[0]["name"]) > 0.9, attachments[0]["name"]

    # 7. C(fsc): 목록페이지면 제목이 맞는 상세페이지로 들어가 첨부를 찾는다
    fsc_target = "240409(보도자료) 금융위 핀테크 투자 생태계 활성화 나선다.pdf"
    attachments, meta = collect_attachments(session, FSC_LIST_URL, cache_dir=cache,
                                            targets=[fsc_target])
    assert meta["followed"], f"목록페이지에서 상세로 진입하지 않음: {meta}"
    assert meta["followed"][0]["url"] == FSC_DETAIL_URL, meta["followed"]
    assigned = assign_attachments([fsc_target], attachments)
    ai, score = assigned[0]
    assert attachments[ai]["path"].read_bytes() == PDF_FSC, "엉뚱한 첨부"
    assert score > 0.9, score

    # 8. 제목이 안 맞으면 아무 상세페이지나 따라 들어가지 않는다
    assert follow_candidates(FSC_LIST_HTML, FSC_LIST_URL, ["전혀 상관없는 문서.pdf"]) == []

    # 9. PDF 가 없는 페이지면 빈 목록
    empty, _ = collect_attachments(session, "https://example.com/none", cache_dir=cache)
    assert empty == []

    shutil.rmtree(cache, ignore_errors=True)
    print("스크래퍼 테스트 통과 (링크수집·파일명·1:1 배정·캐시·kofia·목록페이지 9개 항목)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
