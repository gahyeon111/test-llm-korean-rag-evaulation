"""게시판 상세페이지 HTML 에서 PDF 첨부파일을 찾아 내려받는다.

documents.csv 의 url 은 PDF 직링크가 아니라 게시글 상세페이지다.
한 게시글에 첨부가 여러 개 붙는 경우가 있으므로(같은 URL 이 서로 다른 문서에
매핑됨) 반드시 target 파일명과 가장 잘 맞는 첨부를 골라야 한다.

절차: 상세페이지 GET → 다운로드로 보이는 링크 수집 → 각각 받아보고
%PDF 인 것만 남김 → Content-Disposition 파일명과 target 파일명 유사도로 선택.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import unicodedata
import urllib.parse
from pathlib import Path

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
            return nfc(_unquote_leftover(cand))
    return nfc(_unquote_leftover(candidates[0]))


def _unquote_leftover(name: str) -> str:
    """한글은 살아있는데 공백 등만 %XX 로 남은 경우를 마저 푼다."""
    if "%" not in name:
        return name
    try:
        decoded = urllib.parse.unquote(name, errors="strict")
    except (UnicodeDecodeError, LookupError):
        return name
    return decoded or name


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


def collect_attachments(session: requests.Session, page_url: str, *,
                        cache_dir: Path, timeout: int = 60,
                        max_candidates: int = 10,
                        max_bytes: int = 300_000_000) -> tuple[list[dict], dict]:
    """상세페이지의 PDF 첨부를 **전부** 받아 캐시에 저장하고 목록을 돌려준다.

    한 게시글에 여러 문서가 걸려 있는 경우가 있어서, 문서마다 페이지를 다시
    긁는 대신 첨부를 한 번만 받아 두고 호출부에서 배정한다.

    반환: ([{url, name, path, size}], meta)
    """
    meta: dict = {"page_status": "", "n_candidates": 0, "n_attachments": 0}
    try:
        page = session.get(page_url, headers=BROWSER_HEADERS, timeout=timeout,
                           allow_redirects=True)
    except requests.RequestException as exc:
        meta["page_status"] = f"{type(exc).__name__}: {exc}"
        return [], meta

    meta["page_status"] = page.status_code
    if page.status_code != 200:
        return [], meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    if page.content[:4] == b"%PDF":   # url 이 이미 PDF 직링크인 경우
        name = content_disposition_filename(page.headers.get("content-disposition", "")) or \
            urllib.parse.unquote(urllib.parse.urlsplit(page.url).path.rsplit("/", 1)[-1])
        key = hashlib.sha1(page.url.encode("utf-8")).hexdigest()
        dest = cache_dir / f"{key}.pdf"
        dest.write_bytes(page.content)
        meta.update({"n_candidates": 1, "n_attachments": 1})
        return [{"url": page.url, "name": nfc(name), "path": dest,
                 "size": len(page.content)}], meta

    page.encoding = page.encoding or page.apparent_encoding
    candidates = candidate_urls(page.text, page.url)
    meta["n_candidates"] = len(candidates)

    index_path = cache_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

    attachments: list[dict] = []
    for url in candidates[:max_candidates]:
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        cached = index.get(key)
        if cached and (cache_dir / cached["file"]).exists():
            attachments.append({"url": url, "name": cached["name"],
                                "path": cache_dir / cached["file"],
                                "size": (cache_dir / cached["file"]).stat().st_size})
            continue
        got = fetch_pdf_bytes(session, url, page.url, timeout, max_bytes)
        if not got:
            continue
        body, name = got
        file_name = f"{key}.pdf"
        (cache_dir / file_name).write_bytes(body)
        index[key] = {"name": name, "file": file_name, "url": url}
        attachments.append({"url": url, "name": name,
                            "path": cache_dir / file_name, "size": len(body)})

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["n_attachments"] = len(attachments)
    return attachments, meta


def assign_attachments(targets: list[str], attachments: list[dict]) -> dict[int, tuple[int, float]]:
    """target 파일명들에 첨부를 1:1 로 배정한다 (유사도 높은 쌍부터 확정).

    같은 게시글에 걸린 문서가 여러 개일 때 모두 같은 첨부를 집는 사고를 막는다.
    반환: {target 인덱스: (첨부 인덱스, 유사도)}
    """
    pairs = sorted(
        ((name_similarity(t, a["name"]), ti, ai)
         for ti, t in enumerate(targets) for ai, a in enumerate(attachments)),
        key=lambda x: (-x[0], x[1], x[2]))
    assigned: dict[int, tuple[int, float]] = {}
    used_targets: set[int] = set()
    used_attachments: set[int] = set()
    for score, ti, ai in pairs:
        if ti in used_targets or ai in used_attachments:
            continue
        assigned[ti] = (ai, round(score, 3))
        used_targets.add(ti)
        used_attachments.add(ai)
    return assigned
