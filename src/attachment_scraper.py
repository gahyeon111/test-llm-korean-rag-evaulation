"""게시판 페이지에서 PDF 첨부를 찾아 내려받는다.

documents.csv 의 url 은 PDF 직링크가 아니다. 실제로는 세 종류가 섞여 있다:
  1) 게시글 상세페이지 — 첨부가 여러 개 붙어 있고, 서로 다른 문서가 같은 url 을 갖기도 한다
  2) 게시판 목록페이지 — 목표 문서의 상세페이지로 한 번 더 들어가야 한다
  3) (드물게) PDF 직링크

그래서 첨부 링크를 사이트별 하드코딩 없이 "받아보고 %PDF 인지 확인"하는 방식으로 찾고,
파일명 유사도로 어느 문서의 첨부인지 판정한다.
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

ANCHOR_RE = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
ATTR_RE = re.compile(r"""(?:href|src|data-url)\s*=\s*["']([^"'<>]+)["']""", re.I)
JS_CALL_RE = re.compile(
    r"""(?:fn_egov_downFile|fn_download|fnDownload|downFile|fileDown|goDownload|
         fn_fileDown|f_fileDown|fileDownload)\s*\(([^)]{0,200})\)""",
    re.I | re.X)
JS_ARG_RE = re.compile(r"""['"]([^'"]*)['"]""")
FILE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{6,}$")

# 다운로드 링크로 볼 만한 신호. 사이트마다 이름이 제각각이라
# (bok: fileDown.do / kofia: down.do?file_seq= / fsc: /file/...) 넓게 잡고,
# 실제 PDF 여부는 받아서 %PDF 헤더로 확인한다.
STRONG_HINT = re.compile(
    r"(filedown|file_down|file\s*down|download|/fms/|atchfile|atch_file|"
    r"down\.do|file_seq|fileseq|fileno|file_id|fileid|data_tp|\.pdf)", re.I)
WEAK_HINT = re.compile(r"(down|file|attach|atch|첨부)", re.I)
SKIP_SCHEME = re.compile(r"^(mailto:|tel:|#|javascript:void)", re.I)
NAV_HINT = re.compile(r"(list\.do|/list|login|logout|search|sitemap|privacy|rss)", re.I)
DOWNLOAD_PATH_HINT = re.compile(r"(filedown|file_down|/fms/|filedownload|down\.do)", re.I)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def _norm_for_match(name: str) -> str:
    """파일명 비교용 정규화: 확장자·공백·구분자 제거."""
    name = nfc(name)
    name = re.sub(r"\.(pdf|hwp|hwpx|docx?|zip)$", "", name, flags=re.I)
    return re.sub(r"[\s_\-()\[\]{}.,·*'\"]+", "", name).lower()


def name_similarity(a: str, b: str) -> float:
    na, nb = _norm_for_match(a), _norm_for_match(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.95
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _unquote_leftover(name: str) -> str:
    """한글은 살아있는데 공백 등만 %XX 로 남은 경우를 마저 푼다."""
    if "%" not in name:
        return name
    try:
        return urllib.parse.unquote(name, errors="strict") or name
    except (UnicodeDecodeError, LookupError):
        return name


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


def _anchor_text(inner_html: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", inner_html)).strip()


def _js_download_urls(base_url: str, args_text: str, known_paths: list[str]) -> list[str]:
    """fn_egov_downFile('FILE_000...','0') 같은 호출을 실제 다운로드 URL 로 바꾼다."""
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


def candidate_urls(html: str, base_url: str, limit: int = 15) -> list[dict]:
    """첨부 다운로드로 보이는 링크를 우선순위대로 수집.

    반환: [{url, name_hint}] — name_hint 는 링크 텍스트(파일명이 적혀 있는 경우가 많다).
    """
    collected: list[tuple[int, str, str]] = []   # (우선순위, url, name_hint)
    known_paths: list[str] = []
    js_args: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(raw: str, name_hint: str) -> None:
        raw = raw.strip().replace("&amp;", "&")
        if not raw or SKIP_SCHEME.match(raw):
            return
        if raw.lower().startswith("javascript:"):
            js_args.extend((m.group(1), name_hint)
                           for m in JS_CALL_RE.finditer(raw))
            return
        priority = 0 if STRONG_HINT.search(raw) else (1 if WEAK_HINT.search(raw) else None)
        if priority is None or NAV_HINT.search(raw):
            return
        absolute = urllib.parse.urljoin(base_url, raw)
        if absolute in seen:
            return
        seen.add(absolute)
        collected.append((priority, absolute, name_hint))
        path = urllib.parse.urlsplit(absolute).path
        if DOWNLOAD_PATH_HINT.search(path):
            known_paths.append(path)

    for attrs, inner in ANCHOR_RE.findall(html):
        hint = _anchor_text(inner)
        for value in ATTR_RE.findall(attrs):
            add(value, hint)
        for match in JS_CALL_RE.finditer(attrs):
            js_args.append((match.group(1), hint))

    for value in ATTR_RE.findall(html):   # a 태그 밖의 링크
        add(value, "")
    js_args.extend((m.group(1), "") for m in JS_CALL_RE.finditer(html))

    for args_text, hint in js_args:
        for url in _js_download_urls(base_url, args_text, known_paths):
            if url not in seen:
                seen.add(url)
                collected.append((0, url, hint))

    collected.sort(key=lambda item: item[0])
    return [{"url": url, "name_hint": hint} for _, url, hint in collected[:limit]]


def follow_candidates(html: str, base_url: str, targets: list[str],
                      limit: int = 3, min_score: float = 0.45) -> list[dict]:
    """목록 페이지에서 target 문서의 상세페이지로 보이는 링크를 찾는다.

    링크 텍스트(게시글 제목)와 target 파일명의 유사도로 고른다.
    """
    host = urllib.parse.urlsplit(base_url).netloc
    found: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for attrs, inner in ANCHOR_RE.findall(html):
        text = _anchor_text(inner)
        if len(text) < 6:
            continue
        score = max((name_similarity(t, text) for t in targets), default=0.0)
        if score < min_score:
            continue
        for value in ATTR_RE.findall(attrs):
            raw = value.strip().replace("&amp;", "&")
            if not raw or SKIP_SCHEME.match(raw) or raw.lower().startswith("javascript:"):
                continue
            absolute = urllib.parse.urljoin(base_url, raw)
            if absolute in seen or absolute.split("#")[0] == base_url.split("#")[0]:
                continue
            if urllib.parse.urlsplit(absolute).netloc != host:
                continue
            seen.add(absolute)
            found.append((score, absolute, text))
            break
    found.sort(key=lambda item: -item[0])
    return [{"url": url, "title": title, "score": round(score, 3)}
            for score, url, title in found[:limit]]


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
                name = urllib.parse.unquote(
                    urllib.parse.urlsplit(resp.url).path.rsplit("/", 1)[-1])
            return b"".join(chunks), nfc(name)
    except requests.RequestException:
        return None


def _fetch_page(session: requests.Session, url: str, timeout: int):
    try:
        page = session.get(url, headers=BROWSER_HEADERS, timeout=timeout,
                           allow_redirects=True)
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if page.status_code != 200:
        return None, page.status_code
    return page, page.status_code


def _cache_attachment(cache_dir: Path, index: dict, url: str, body: bytes,
                      name: str) -> dict:
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    path = cache_dir / f"{key}.pdf"
    path.write_bytes(body)
    index[key] = {"name": name, "file": path.name, "url": url}
    return {"url": url, "name": name, "path": path, "size": len(body)}


def _attachments_from_page(session: requests.Session, page, cache_dir: Path,
                           index: dict, timeout: int, max_candidates: int,
                           max_bytes: int) -> tuple[list[dict], int]:
    page.encoding = page.encoding or page.apparent_encoding
    candidates = candidate_urls(page.text, page.url)
    attachments: list[dict] = []
    for cand in candidates[:max_candidates]:
        url = cand["url"]
        key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        cached = index.get(key)
        if cached and (cache_dir / cached["file"]).exists():
            path = cache_dir / cached["file"]
            attachments.append({"url": url, "name": cached["name"], "path": path,
                                "size": path.stat().st_size})
            continue
        got = fetch_pdf_bytes(session, url, page.url, timeout, max_bytes)
        if not got:
            continue
        body, name = got
        # 서버가 파일명을 안 주면 링크 텍스트를 쓴다 (게시판은 보통 파일명을 적어둔다)
        if (not name or not name.lower().endswith(".pdf")) and cand["name_hint"]:
            name = cand["name_hint"]
        attachments.append(_cache_attachment(cache_dir, index, url, body, name))
    return attachments, len(candidates)


def collect_attachments(session: requests.Session, page_url: str, *,
                        cache_dir: Path, targets: list[str] | None = None,
                        timeout: int = 60, max_candidates: int = 10,
                        max_bytes: int = 300_000_000, follow_links: int = 3,
                        min_score: float = 0.6) -> tuple[list[dict], dict]:
    """페이지의 PDF 첨부를 전부 받아 캐시에 저장하고 목록을 돌려준다.

    targets 를 주면, 그 문서들에 맞는 첨부가 안 나올 때 목록페이지로 보고
    제목이 비슷한 상세페이지를 따라 들어가 한 번 더 찾는다.

    반환: ([{url, name, path, size}], meta)
    """
    targets = list(targets or [])
    meta: dict = {"page_status": "", "n_candidates": 0, "n_attachments": 0,
                  "followed": []}

    page, status = _fetch_page(session, page_url, timeout)
    meta["page_status"] = status
    if page is None:
        return [], meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

    if page.content[:4] == b"%PDF":   # url 이 이미 PDF 직링크인 경우
        name = content_disposition_filename(page.headers.get("content-disposition", "")) or \
            urllib.parse.unquote(urllib.parse.urlsplit(page.url).path.rsplit("/", 1)[-1])
        attachment = _cache_attachment(cache_dir, index, page.url, page.content, nfc(name))
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        meta.update({"n_candidates": 1, "n_attachments": 1})
        return [attachment], meta

    attachments, n_candidates = _attachments_from_page(
        session, page, cache_dir, index, timeout, max_candidates, max_bytes)
    meta["n_candidates"] = n_candidates

    # 목록페이지 대응: target 에 맞는 첨부가 없으면 상세페이지를 따라 들어간다
    if targets and follow_links and not _covers_targets(targets, attachments, min_score):
        for link in follow_candidates(page.text, page.url, targets, follow_links):
            sub_page, _ = _fetch_page(session, link["url"], timeout)
            if sub_page is None:
                continue
            sub, sub_n = _attachments_from_page(
                session, sub_page, cache_dir, index, timeout, max_candidates, max_bytes)
            meta["followed"].append({"url": link["url"], "title": link["title"],
                                     "n_attachments": len(sub)})
            meta["n_candidates"] += sub_n
            known = {a["url"] for a in attachments}
            attachments += [a for a in sub if a["url"] not in known]
            if _covers_targets(targets, attachments, min_score):
                break

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["n_attachments"] = len(attachments)
    return attachments, meta


def _covers_targets(targets: list[str], attachments: list[dict], min_score: float) -> bool:
    """모든 target 이 기준 유사도 이상의 첨부를 하나씩 가질 수 있는지."""
    if len(attachments) < len(targets):
        return False
    assigned = assign_attachments(targets, attachments)
    return len(assigned) == len(targets) and all(
        score >= min_score for _, score in assigned.values())


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
