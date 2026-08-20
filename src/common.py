"""공통 유틸리티: 경로 상수, 환경변수 로딩, OpenRouter 호출, CSV 스키마 정규화.

모든 모듈이 이 파일을 통해 OpenRouter를 호출한다.
재시도/백오프/응답 메타(provider, latency, token) 기록이 여기에 모여 있다.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import requests

# ---------------------------------------------------------------- 경로
# ALLGANIZE_EVAL_ROOT 로 작업 루트를 바꿀 수 있다 (스모크 테스트/멀티 실험용).
ROOT = Path(os.environ.get("ALLGANIZE_EVAL_ROOT") or Path(__file__).resolve().parent.parent)
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CACHE_DIR = ROOT / "cache" / "parsed"
CONTEXT_DIR = ROOT / "contexts"
RESULT_RAW_DIR = ROOT / "results" / "raw"
RESULT_SCORED_DIR = ROOT / "results" / "scored"
REPORT_DIR = ROOT / "reports"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

DOCUMENTS_CSV = DATA_DIR / "documents.csv"
EXCLUDED_FILES = DATA_DIR / "excluded_files.txt"
DATASET_CSV = DATA_DIR / "dataset.csv"
CONTEXTS_JSONL = CONTEXT_DIR / "contexts.jsonl"

for _d in (DATA_DIR, PDF_DIR, CACHE_DIR, CONTEXT_DIR, RESULT_RAW_DIR, RESULT_SCORED_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 환경변수
def load_env(env_path: Path | None = None) -> None:
    """.env 를 읽어 os.environ 에 주입한다 (이미 설정된 값은 덮어쓰지 않음)."""
    path = env_path or (ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def require_api_key() -> str:
    load_env()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        sys.exit(
            "OPENROUTER_API_KEY 가 없습니다. .env 파일(.env.example 참고)에 키를 넣거나 "
            "환경변수로 지정하세요."
        )
    return key


# ---------------------------------------------------------------- OpenRouter
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}


class OpenRouterError(RuntimeError):
    """재시도해도 복구되지 않은 호출 실패."""


def build_provider(quantizations: Iterable[str] | None = None,
                   order: Iterable[str] | None = None,
                   allow_fallbacks: bool = False) -> dict[str, Any]:
    """provider 라우팅 옵션. 스펙 기본값은 fp8 고정 + fallback 금지."""
    provider: dict[str, Any] = {"allow_fallbacks": allow_fallbacks}
    quantizations = [q for q in (quantizations or []) if q]
    order = [o for o in (order or []) if o]
    if quantizations:
        provider["quantizations"] = list(quantizations)
    if order:
        provider["order"] = list(order)
    return provider


def chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    provider: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    retries: int = 3,
    timeout: int = 300,
    api_key: str | None = None,
) -> dict[str, Any]:
    """OpenRouter chat/completions 호출.

    reasoning 은 반드시 nested 형식(``{"reasoning": {"effort": "high"}}``)으로만 보낸다.
    flat 파라미터(``reasoning_effort``)와 혼용하면 400 이 난다.

    reasoning_effort 값의 의미:
      None    파라미터를 보내지 않는다 → **모델 기본값**을 따른다.
              하이브리드 모델(기본 thinking on)은 이때 그대로 추론한다.
      "off"   ``{"reasoning": {"enabled": false}}`` 로 추론을 끈다.
      그 외    ``{"reasoning": {"effort": <값>}}``

    반환: {text, raw, latency_s, provider, usage, attempts, error}
    """
    key = api_key or require_api_key()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if reasoning_effort == "off":
        payload["reasoning"] = {"enabled": False}
    elif reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    if provider:
        payload["provider"] = provider
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if response_format:
        payload["response_format"] = response_format
    if extra_body:
        payload.update(extra_body)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": "allganize-rag-eval",
    }

    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.time()
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 json=payload, timeout=timeout)
            latency = time.time() - started
            if resp.status_code >= 400:
                body = resp.text[:800]
                last_error = f"HTTP {resp.status_code}: {body}"
                if resp.status_code not in RETRYABLE_STATUS:
                    # 400/401/402/404 는 요청 자체가 잘못된 것 — 재시도 무의미
                    if "valid model" in body.lower() or "not found" in body.lower():
                        last_error += ("\n  → 모델 ID 가 카탈로그에 없습니다. "
                                       "`python tools/check_models.py` 로 확인하세요.")
                    raise OpenRouterError(last_error)
            else:
                data = resp.json()
                if "error" in data and not data.get("choices"):
                    last_error = f"API error: {json.dumps(data['error'], ensure_ascii=False)[:800]}"
                else:
                    return {
                        "text": extract_text(data),
                        "raw": data,
                        "latency_s": round(latency, 3),
                        "provider": data.get("provider", ""),
                        "usage": normalize_usage(data.get("usage") or {}),
                        "attempts": attempt,
                        "error": "",
                    }
        except OpenRouterError:
            raise
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except ValueError as exc:  # JSON 파싱 실패
            last_error = f"JSONDecodeError: {exc}"

        if attempt < retries:
            sleep_s = (2 ** attempt) + random.uniform(0, 1)
            print(f"  ! 실패({attempt}/{retries}) {last_error[:200]} → {sleep_s:.1f}s 후 재시도",
                  file=sys.stderr)
            time.sleep(sleep_s)

    raise OpenRouterError(last_error or "unknown error")


def extract_text(data: dict[str, Any]) -> str:
    """choices[0].message.content 를 문자열로 뽑는다 (content 가 리스트인 경우 포함)."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "\n".join(p for p in parts if p).strip()
    return ""


def normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def parse_json_block(text: str) -> dict[str, Any] | None:
    """모델 응답에서 JSON 객체를 최대한 관대하게 파싱한다."""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------- 스키마 정규화
DATASET_ALIASES = {
    "question": ["question", "query", "질문"],
    "target_answer": ["target_answer", "answer", "gt_answer", "정답"],
    "target_file_name": ["target_file_name", "file_name", "document_name", "doc_name"],
    "target_page_no": ["target_page_no", "page_no", "page", "target_page"],
    "context_type": ["context_type", "type", "answer_type"],
    "domain": ["domain", "category", "도메인"],
}
DOCUMENTS_ALIASES = {
    "domain": ["domain", "category"],
    "file_name": ["file_name", "target_file_name", "document_name", "doc_name", "name"],
    "url": ["url", "link", "document_url", "pdf_url", "source"],
}


def normalize_columns(df, aliases: dict[str, list[str]], *, required: Iterable[str] = ()):
    """컬럼명을 표준 이름으로 정규화한다. 원본 데이터셋 컬럼명 변경에 대한 방어."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for cand in candidates:
            if cand.lower() in lookup:
                rename[lookup[cand.lower()]] = canonical
                break
    df = df.rename(columns=rename)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"필수 컬럼 누락: {missing}\n실제 컬럼: {list(df.columns)}\n"
            "src/common.py 의 ALIASES 에 실제 컬럼명을 추가하세요."
        )
    return df


# 리눅스/ext4 의 파일명 한도는 255 바이트. 한글은 UTF-8 에서 글자당 3바이트라
# 85자만 넘어도 걸린다 (원본 데이터셋에 그런 파일명이 실제로 있다).
MAX_FILENAME_BYTES = 240


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).strip()


def load_excluded_files(path: Path | None = None) -> set[str]:
    """평가에서 뺄 문서 목록 (data/excluded_files.txt).

    한 줄에 target_file_name 하나. `#` 뒤는 주석이라 판정 근거를 적어둘 수 있고,
    확인 후 줄을 지우면 그 문서가 다시 평가에 들어온다.
    """
    path = path or EXCLUDED_FILES
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.add(nfc(name))
    return names


def safe_filename(name: str, max_bytes: int = MAX_FILENAME_BYTES) -> str:
    """길이 한도를 넘는 파일명을 잘라내되, 원본 이름 해시를 붙여 충돌을 막는다."""
    name = unicodedata.normalize("NFC", str(name)).strip()
    if len(name.encode("utf-8")) <= max_bytes:
        return name
    stem, ext = os.path.splitext(name)
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    budget = max_bytes - len(ext.encode("utf-8")) - len(digest) - 1
    truncated = stem.encode("utf-8")[:budget].decode("utf-8", "ignore").rstrip()
    return f"{truncated}_{digest}{ext}"


def pdf_path(file_name: str) -> Path:
    """target_file_name → 실제 저장 경로. 전 단계가 이 함수로만 경로를 만든다."""
    return PDF_DIR / safe_filename(file_name)


def slugify(name: str) -> str:
    """파일명을 캐시 키로 안전하게 변환 (한글은 유지)."""
    stem = Path(str(name)).stem
    return re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", stem).strip("_") or "doc"


def cache_path(file_name: str, page_no: int) -> Path:
    return CACHE_DIR / safe_filename(f"{slugify(file_name)}__p{int(page_no)}.json")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
