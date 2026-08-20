"""게시판 상세페이지 HTML 에서 PDF 첨부파일을 찾아 내려받는다.

documents.csv 의 url 은 PDF 직링크가 아니라 게시글 상세페이지다.
한 게시글에 첨부가 여러 개 붙는 경우가 있으므로(같은 URL 이 서로 다른 문서에
매핑됨) 반드시 target 파일명과 가장 잘 맞는 첨부를 골라야 한다.

절차: 상세페이지 GET → 다운로드로 보이는 링크 수집 → 각각 받아보고
%PDF 인 것만 남김 → Content-Disposition 파일명과 target 파일명 유사도로 선택.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
import urllib.parse
from typing import Iterable

import requests

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

ATTR_RE = re.compile(r"""(?:href|src|data-url)\s*=\s*["']([^"'<>]+)["']""", re.I)
# 표준프레임워크 계열: fn_egov_downFile('FILE_000000000012345','0')
JS_CALL_RE = re.compile(
    r"""(?:fn_egov_downFile|fn_download|fnDownload|downFile|fileDown|goDownload|
         fn_fileDown|f_fileDown|fileDownload)\s*\(([^)]{0,200})\)""",
    re.I | re.X)
JS_ARG_RE = re.compile(r"""['"]([^'"]*)['"]""")
FILE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{6,}$")
DOWNLOAD_HINT = re.compile(
    r"(file\s*down|filedown|file_down|download|atch|attach|/fms/|/fileDown|\.pdf)", re.I)
DOWNLOAD_PATH_HINT = re.compile(r"(filedown|file_down|/fms/|filedownload)", re.I)
SKIP_SCHEME = re.compile(r"^(mailto:|tel:|#|javascript:void)", re.I)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def _norm_for_match(name: str) -> str:
    """파일명 비교용 정규화: 확장자·공백·구분자 제거."""
    name = nfc(name)
    name = re.sub(r"\.pdf$", "", name, flags=re.I)
    return re.sub(r"[\s_\-()\[\].,·]+", "", name).lower()


def name_similarity(a: str, b: str) -> float:
    na, nb = _norm_for_match(a), _norm_for_match(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.95
    return difflib.SequenceMatcher(None, na, nb).ratio()


def content_disposition_filename(header: str) -> str:
    """Content-Disposition 에서 파일명 추출. 한글 인코딩 깨짐까지 복원 시도."""
    if not header:
        return ""
    star = re.search(r"filename\*\s*=\s*([^;]+)", header, re.I)
    if star:
        value = star.group(1).strip().strip('"')
        encoding, _, encoded = value.partition("''")
        try:
            return nfc(urllib.parse.unquote(encoded or value,
                                            encoding=encoding or "utf-8"))
        except (LookupError, UnicodeDecodeError):
            pass
    plain = re.search(r'filename\s*=\s*"?([^";]+)"?', header, re.I)
    if not plain:
        return ""
    raw = plain.group(1).strip()
    candidates = [raw]
    try:  # 서버가 latin-1 로 흘려보낸 EUC-KR/UTF-8 파일명 복원
        as_bytes = raw.encode("latin-1")
        for enc in ("utf-8", "euc-kr", "cp949"):
            try:
                candidates.append(as_bytes.decode(enc))
            except UnicodeDecodeError:
                continue
    except UnicodeEncodeError:
        pass
    for enc in ("utf-8", "euc-kr"):
        try:
            candidates.append(urllib.parse.unquote(raw, encoding=enc))
        except (LookupError, UnicodeDecodeError):
            continue
    for cand in candidates:  # 한글이 살아있는 후보를 우선
        if re.search(r"[가-힣]", cand):
            return nfc(cand)
    return nfc(candidates[0])


def _js_download_urls(base_url: str, args_text: str, known_paths: list[str]) -> list[str]:
    """fn_egov_downFile('FILE_000...','0') 같은 호출을 실제 다운로드 URL 로 바꾼다.

    다운로드 경로는 사이트마다 달라서(bok 은 /portal/cmmn/file/fileDown.do 등)
    같은 페이지의 href 에서 발견한 경로를 우선 재사용하고, 표준프레임워크 기본
    경로를 마지막 후보로 붙인다.
    """
    args = [a.strip() for a in JS_ARG_RE.findall(args_text) if a.strip()]
    if not args or not FILE_ID_RE.match(args[0]):
        return []
    file_id = args[0]
    file_sn = next((a for a in args[1:] if a.isdigit()), "0")
    parts = urllib.parse.urlsplit(base_url)
    paths = list(dict.fromkeys([*known_paths, "/cmm/fms/FileDown.do"]))
    return [urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, path,
         f"atchFileId={file_id}&fileSn={file_sn}", "")) for path in paths]


def candidate_urls(html: str, base_url: str, limit: int = 15) -> list[str]:
    """상세페이지에서 첨부 다운로드로 보이는 링크를 우선순위대로 수집."""
    direct: list[str] = []
    known_paths: list[str] = []
    js_args: list[str] = []

    for raw in ATTR_RE.findall(html):
        raw = raw.strip().replace("&amp;", "&")
        if SKIP_SCHEME.match(raw):
            continue
        if raw.lower().startswith("javascript:"):
            js_args += [m.group(1) for m in JS_CALL_RE.finditer(raw)]
            continue
        if DOWNLOAD_HINT.search(raw):
            absolute = urllib.parse.urljoin(base_url, raw)
            direct.append(absolute)
            path = urllib.parse.urlsplit(absolute).path
            if DOWNLOAD_PATH_HINT.search(path):
                known_paths.append(path)

    js_args += [m.group(1) for m in JS_CALL_RE.finditer(html)]
    js_urls: list[str] = []
    for args_text in js_args:
        js_urls += _js_download_urls(base_url, args_text, known_paths)

    ordered, seen = [], set()
    for url in direct + js_urls:
        if url and url not in seen:
            seen.add(url)
            ordered.append(url)
    # .pdf / filedown 이 명시된 링크를 앞으로
    ordered.sort(key=lambda u: (0 if re.search(r"\.pdf|filedown|/fms/", u, re.I) else 1))
    return ordered[:limit]


def fetch_pdf_bytes(session: requests.Session, url: str, referer: str,
                    timeout: int, max_bytes: int) -> tuple[bytes, str] | None:
    """URL 이 PDF 를 주면 (본문, 파일명) 반환. 아니면 None."""
    headers = dict(BROWSER_HEADERS)
    headers["Referer"] = referer
    try:
        with session.get(url, headers=headers, timeout=timeout,
                         stream=True, allow_redirects=True) as resp:
            if resp.status_code != 200:
                return None
            if "text/html" in resp.headers.get("content-type", "").lower():
                return None
            # 스트림 이터레이터는 하나만 만들어 이어서 소비한다
            # (두 번 만들면 구현에 따라 앞부분을 다시 읽거나 건너뛴다).
            stream = resp.iter_content(65536)
            head = next(stream, b"")
            if not head.startswith(b"%PDF"):
                return None
            chunks, total = [head], len(head)
            for chunk in stream:
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    return None
            name = content_disposition_filename(resp.headers.get("content-disposition", ""))
            if not name:
                name = urllib.parse.unquote(urllib.parse.urlsplit(resp.url).path.rsplit("/", 1)[-1])
            return b"".join(chunks), name
    except requests.RequestException:
        return None


def scrape_pdf(session: requests.Session, page_url: str, target_name: str, *,
               timeout: int = 60, max_candidates: int = 8,
               max_bytes: int = 120_000_000,
               accept_threshold: float = 0.85,
               log: Iterable[str] | None = None) -> tuple[bytes, dict] | None:
    """상세페이지에서 target_name 에 가장 맞는 PDF 첨부를 받아 온다."""
    meta: dict = {"page_status": "", "candidates": 0, "picked_url": "",
                  "picked_name": "", "score": 0.0}
    try:
        page = session.get(page_url, headers=BROWSER_HEADERS, timeout=timeout,
                           allow_redirects=True)
    except requests.RequestException as exc:
        meta["page_status"] = f"{type(exc).__name__}: {exc}"
        return None
    meta["page_status"] = page.status_code
    if page.status_code != 200:
        return None

    page.encoding = page.encoding or page.apparent_encoding
    candidates = candidate_urls(page.text, page.url)
    meta["candidates"] = len(candidates)

    best: tuple[float, bytes, str, str] | None = None
    for url in candidates[:max_candidates]:
        got = fetch_pdf_bytes(session, url, page.url, timeout, max_bytes)
        if not got:
            continue
        body, name = got
        score = name_similarity(target_name, name)
        if best is None or score > best[0]:
            best = (score, body, name, url)
        if score >= accept_threshold:
            break

    if best is None:
        return None
    score, body, name, url = best
    meta |= {"picked_url": url, "picked_name": name, "score": round(score, 3)}
    return body, meta
