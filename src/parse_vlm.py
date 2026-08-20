"""2단계: target 페이지만 VLM 으로 파싱해 마크다운을 캐시한다.

전량 파싱 금지 — dataset.csv 의 (target_file_name, target_page_no) 유니크 페어만 처리한다.
캐시가 있으면 스킵하므로 재실행이 안전하다.

출력
  cache/parsed/<문서명>__p<페이지>.json   (원응답 raw 그대로 보존)
  cache/pages/<문서명>__p<페이지>.png     (--save-image 시, 육안 검수용)
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pymupdf

from common import (CACHE_DIR, DATASET_CSV, OpenRouterError, PROMPT_DIR,
                    cache_path, chat_completion, load_excluded_files, nfc,
                    pdf_path, require_api_key, safe_filename, slugify)

# 스펙의 google/gemini-3.1-pro 는 OpenRouter 카탈로그에 없다 (실제 ID 는
# gemini-3.1-pro-preview, $2/$12 per 1M). 파싱 100페이지 기준 pro 계열 ~$2.2 vs
# 3.7-flash ~$0.8 이고, 세대가 더 높은 flash 라 문서 파싱 품질도 크게 뒤지지 않는다.
# 표·차트 재현이 부족하면 --model google/gemini-3.1-pro-preview 로 올린다.
DEFAULT_MODEL = "google/gemini-3.7-flash"
PROMPT_VERSION = "parse_v1"
IMAGE_DIR = CACHE_DIR.parent / "pages"


def render_page(pdf_file: Path, page_no: int, dpi: int, page_base: int) -> bytes:
    """target_page_no 를 0-based 인덱스로 바꿔 해당 페이지만 PNG 로 렌더링."""
    with pymupdf.open(pdf_file) as doc:
        index = int(page_no) - page_base
        if index < 0 or index >= doc.page_count:
            raise IndexError(f"페이지 범위 초과: page_no={page_no} (문서 {doc.page_count}p)")
        page = doc.load_page(index)
        return page.get_pixmap(dpi=dpi).tobytes("png")


def parse_page(image_png: bytes, prompt: str, model: str, api_key: str,
               retries: int) -> dict:
    data_uri = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    }]
    return chat_completion(model, messages, temperature=0.0,
                           retries=retries, api_key=api_key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dpi", type=int, default=200, help="≥150 권장")
    ap.add_argument("--page-base", type=int, default=1,
                    help="target_page_no 의 시작값 (1=1-based, 0=0-based)")
    ap.add_argument("--limit", type=int, default=0, help="선행 검수용 N페어만 처리")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.0, help="호출 간 대기(초)")
    ap.add_argument("--save-image", action="store_true", help="렌더 이미지도 저장(육안 검수)")
    ap.add_argument("--force", action="store_true", help="캐시 무시하고 재파싱")
    ap.add_argument("--abort-after", type=int, default=3,
                    help="연속 N회 실패하면 중단 (모델 ID 오류 등 설정 문제)")
    args = ap.parse_args()

    if args.dpi < 150:
        raise SystemExit("dpi 는 150 이상이어야 합니다 (스펙 §4.2).")

    api_key = require_api_key()
    prompt = (PROMPT_DIR / f"{PROMPT_VERSION}.txt").read_text(encoding="utf-8")
    dataset = pd.read_csv(DATASET_CSV)

    excluded = load_excluded_files()
    if excluded:
        before = len(dataset)
        dataset = dataset[~dataset["target_file_name"].map(nfc).isin(excluded)]
        print(f"제외 목록 적용: 문서 {len(excluded)}개 → 문항 {before - len(dataset)}건 제외")

    pairs = (dataset[["target_file_name", "target_page_no"]]
             .dropna().drop_duplicates()
             .sort_values(["target_file_name", "target_page_no"]))
    print(f"유니크 (문서, 페이지) 페어: {len(pairs)}건")
    if args.limit:
        pairs = pairs.head(args.limit)

    stats = {"ok": 0, "cached": 0, "failed": 0}
    failures = []
    consecutive_failures = 0
    for i, pair in enumerate(pairs.itertuples(index=False), start=1):
        file_name = str(pair.target_file_name)
        page_no = int(pair.target_page_no)
        out_path = cache_path(file_name, page_no)
        tag = f"[{i}/{len(pairs)}] {file_name} p{page_no}"

        if out_path.exists() and not args.force:
            stats["cached"] += 1
            print(f"{tag} … 캐시 스킵")
            continue

        pdf_file = pdf_path(file_name)
        if not pdf_file.exists():
            stats["failed"] += 1
            failures.append((file_name, page_no, "PDF 없음(다운로드 실패 문서)"))
            print(f"{tag} … PDF 없음 → 스킵")
            continue

        try:
            image_png = render_page(pdf_file, page_no, args.dpi, args.page_base)
            if args.save_image:
                IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                (IMAGE_DIR / safe_filename(
                    f"{slugify(file_name)}__p{page_no}.png")).write_bytes(image_png)
            result = parse_page(image_png, prompt, args.model, api_key, args.retries)
        except (OpenRouterError, IndexError, RuntimeError) as exc:
            stats["failed"] += 1
            failures.append((file_name, page_no, str(exc)[:200]))
            print(f"{tag} … 실패: {str(exc)[:300]}")
            consecutive_failures += 1
            if isinstance(exc, OpenRouterError) and consecutive_failures >= args.abort_after:
                raise SystemExit(
                    f"\n연속 {consecutive_failures}회 실패 — 설정 문제로 보고 중단합니다.\n"
                    f"마지막 오류: {str(exc)[:400]}")
            continue

        out_path.write_text(json.dumps({
            "file_name": file_name,
            "page_no": page_no,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "dpi": args.dpi,
            "page_base": args.page_base,
            "parsed_at": datetime.now(timezone.utc).isoformat(),
            "markdown": result["text"],
            "usage": result["usage"],
            "latency_s": result["latency_s"],
            "provider": result["provider"],
            "raw_response": result["raw"],   # 원응답 보존, 후처리는 build_contexts
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        stats["ok"] += 1
        consecutive_failures = 0
        print(f"{tag} … ok ({len(result['text'])}자, {result['latency_s']}s)")
        if args.sleep:
            time.sleep(args.sleep)

    print(f"\n완료: 신규 {stats['ok']} / 캐시 {stats['cached']} / 실패 {stats['failed']}")
    for file_name, page_no, reason in failures:
        print(f"  실패: {file_name} p{page_no} — {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
