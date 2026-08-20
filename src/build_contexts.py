"""3단계: 파싱 캐시 + dataset 을 합쳐 oracle context 를 만든다 (게이트 B: 동결 대상).

레코드 스키마
  {qid, domain, context_type, question, target_answer,
   target_file, target_page, context}

동결 규칙: contexts.jsonl 이 이미 있으면 덮어쓰지 않는다.
수정이 필요하면 `--version v2` 로 contexts_v2.jsonl 을 새로 만든다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd

from common import (CONTEXT_DIR, CONTEXTS_JSONL, DATASET_CSV, REPORT_DIR,
                    cache_path, load_excluded_files, nfc, write_jsonl)

SCHEMA = ["qid", "domain", "context_type", "question", "target_answer",
          "target_file", "target_page", "context"]


def clean_markdown(text: str) -> str:
    """파서 원응답의 후처리: 감싸는 코드펜스 제거 + 공백 정리."""
    text = (text or "").strip()
    fence = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```$", text, re.S)
    if fence:
        text = fence.group(1)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha256_of(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_review_samples(rows: list[dict], per_group: int) -> None:
    """도메인×context_type 별 샘플을 마크다운으로 떨궈 육안 검수(게이트 A)에 쓴다."""
    df = pd.DataFrame(rows)
    lines = ["# contexts 검수 샘플", "",
             f"생성: {datetime.now(timezone.utc).isoformat()}", ""]
    for (domain, ctype), group in df.groupby(["domain", "context_type"]):
        lines.append(f"## {domain} / {ctype} ({len(group)}건 중 {min(per_group, len(group))}건)")
        for row in group.head(per_group).to_dict("records"):
            lines += [
                f"### {row['qid']} — {row['target_file']} p{row['target_page']}",
                f"- Q: {row['question']}",
                f"- A(target): {row['target_answer']}",
                f"- context 길이: {len(row['context'])}자",
                "",
                "```markdown",
                row["context"][:1500] + ("\n…(생략)" if len(row["context"]) > 1500 else ""),
                "```",
                "",
            ]
    (REPORT_DIR / "context_review_samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"검수 샘플 → {REPORT_DIR / 'context_review_samples.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="", help="예: v2 → contexts_v2.jsonl 로 저장")
    ap.add_argument("--samples", type=int, default=10, help="도메인×타입별 검수 샘플 수")
    ap.add_argument("--verify", action="store_true", help="기존 파일의 동결 해시만 검증")
    args = ap.parse_args()

    out_path = (CONTEXT_DIR / f"contexts_{args.version}.jsonl") if args.version else CONTEXTS_JSONL
    meta_path = out_path.parent / (out_path.name + ".meta.json")

    if args.verify:
        if not (out_path.exists() and meta_path.exists()):
            raise SystemExit("검증할 contexts/meta 파일이 없습니다.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual = sha256_of(out_path)
        ok = actual == meta.get("sha256")
        print(f"{'OK  ' if ok else 'FAIL'} {out_path.name}\n  기록 {meta.get('sha256')}\n  실제 {actual}")
        return 0 if ok else 1

    if out_path.exists():
        raise SystemExit(
            f"{out_path} 는 이미 동결되어 있습니다. 재생성이 필요하면 --version v2 를 쓰세요."
        )

    dataset = pd.read_csv(DATASET_CSV)
    if "qid" not in dataset.columns:
        dataset.insert(0, "qid", [f"q{i:04d}" for i in range(1, len(dataset) + 1)])

    excluded_files = load_excluded_files()
    if excluded_files:
        print(f"제외 목록: 문서 {len(excluded_files)}개 (data/excluded_files.txt)")

    rows, missing = [], []
    for item in dataset.itertuples(index=False):
        file_name = str(item.target_file_name)
        if nfc(file_name) in excluded_files:
            missing.append({"qid": item.qid, "target_file": file_name,
                            "target_page": getattr(item, "target_page_no", ""),
                            "reason": "제외 목록 (검수에서 제외)"})
            continue
        page_raw = getattr(item, "target_page_no", None)
        page_no = int(page_raw) if pd.notna(page_raw) else -1
        cache_file = cache_path(file_name, page_no) if page_no >= 0 else None

        context = ""
        if cache_file and cache_file.exists():
            context = clean_markdown(
                json.loads(cache_file.read_text(encoding="utf-8")).get("markdown", ""))
        if not context:
            missing.append({"qid": item.qid, "target_file": file_name,
                            "target_page": page_no,
                            "reason": "파싱 캐시 없음" if not (cache_file and cache_file.exists())
                                      else "파싱 결과 비어 있음"})
            continue

        rows.append({
            "qid": item.qid,
            "domain": item.domain,
            "context_type": item.context_type,
            "question": item.question,
            "target_answer": item.target_answer,
            "target_file": file_name,
            "target_page": page_no,
            "context": context,
        })

    if not rows:
        raise SystemExit("생성된 레코드가 0건입니다. parse_vlm.py 를 먼저 실행하세요.")

    bad = [r["qid"] for r in rows if set(r) != set(SCHEMA)]
    if bad:
        raise SystemExit(f"스키마 불일치: {bad[:5]}")

    write_jsonl(out_path, rows)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_records": len(rows),
        "n_excluded": len(missing),
        "dataset_rows": len(dataset),
        "sha256": sha256_of(out_path),
        "frozen": True,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if missing:
        pd.DataFrame(missing).to_csv(REPORT_DIR / "context_missing.csv", index=False)

    print(f"생성: {out_path} ({len(rows)}건 / 전체 {len(dataset)}문항, 제외 {len(missing)}건)")
    print(f"sha256 {meta['sha256']}  → {meta_path.name} (동결 기준)")
    counts = pd.DataFrame(rows).groupby(["domain", "context_type"]).size()
    print("\n[도메인 × context_type]")
    print(counts.to_string())
    write_review_samples(rows, args.samples)
    print("\n★ 게이트 B: 검수 샘플 확인 후 이 파일을 동결하세요. 이후 수정 금지.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
