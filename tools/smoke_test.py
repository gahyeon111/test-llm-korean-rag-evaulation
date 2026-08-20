"""네트워크 없이 도는 구간만 검증하는 스모크 테스트.

가짜 데이터셋/파싱 캐시/모델 응답을 임시 루트에 만들고
build_contexts.py → (오프라인 채점) → report.py 를 실제로 돌린다.
API 를 타는 fetch/parse/run/judge 호출 자체는 대상이 아니다.

실행: python3 tools/smoke_test.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

ROWS = [
    # qid, domain, ctype, question, target_answer, file, page, context, model_answer, expect
    ("q0001", "finance", "paragraph", "영업이익은?", "1,234억원", "a.pdf", 3,
     "2024년 영업이익은 1,234억원이다.", "영업이익은 1,234억원입니다.", "O"),
    ("q0002", "public", "table", "2023년 예산은?", "5,600억원", "a.pdf", 7,
     "| 연도 | 예산 |\n|---|---|\n| 2023 | 5,600억원 |", "5,600억원입니다.", "O"),
    ("q0003", "medical", "image", "그래프상 증가율은?", "12.5%", "b.pdf", 2,
     "[그림] 연도별 추이. 값 레이블 없음.", "약 8%입니다.", "X"),
    ("q0004", "law", "paragraph", "시행일은?", "2024년 3월 1일", "b.pdf", 5,
     "이 법은 2024년 3월 1일부터 시행한다.", "2025년 3월 1일입니다.", "X"),
]


def run(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"실패: {' '.join(cmd)}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="allganize-smoke-"))
    os.environ["ALLGANIZE_EVAL_ROOT"] = str(tmp)   # common.ROOT 확정보다 먼저
    env = os.environ | {"ALLGANIZE_EVAL_ROOT": str(tmp), "PYTHONPATH": str(SRC)}

    import pandas as pd  # noqa: E402
    from common import cache_path  # noqa: E402 - ROOT 를 임시 디렉터리로 잡은 뒤 import

    pd.DataFrame([{
        "qid": r[0], "domain": r[1], "context_type": r[2], "question": r[3],
        "target_answer": r[4], "target_file_name": r[5], "target_page_no": r[6],
    } for r in ROWS]).to_csv(tmp / "data" / "dataset.csv", index=False)

    for r in ROWS:
        path = cache_path(r[5], r[6])
        path.write_text(json.dumps({"file_name": r[5], "page_no": r[6],
                                    "markdown": r[7]}, ensure_ascii=False), encoding="utf-8")

    run([sys.executable, str(SRC / "build_contexts.py"), "--samples", "1"], env)
    contexts = (tmp / "contexts" / "contexts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(contexts) == len(ROWS), f"contexts {len(contexts)}건 (기대 {len(ROWS)})"

    # 동결 확인: 두 번째 실행은 반드시 거부되어야 한다
    again = subprocess.run([sys.executable, str(SRC / "build_contexts.py")],
                           env=env, capture_output=True, text=True)
    assert again.returncode != 0, "동결된 contexts 를 덮어썼습니다"

    verify = subprocess.run([sys.executable, str(SRC / "build_contexts.py"), "--verify"],
                            env=env, capture_output=True, text=True)
    assert verify.returncode == 0, f"해시 검증 실패: {verify.stdout}{verify.stderr}"

    # judge 를 타지 않고 기대 판정으로 scored CSV 를 만든 뒤 report 검증
    from judge import numeric_cross_check  # noqa: E402

    scored = []
    for r in ROWS:
        nm, n_t, n_h = numeric_cross_check(r[4], r[8])
        scored.append({
            "qid": r[0], "domain": r[1], "context_type": r[2], "question": r[3],
            "target_answer": r[4], "model_answer": r[8], "target_file": r[5],
            "target_page": r[6], "status": "ok", "provider": "TestProvider",
            "model": "test/model", "reasoning_effort": "high", "temperature": 0,
            "verdict": r[9], "judge_reason": "smoke", "numeric_match": nm,
            "numeric_targets": n_t, "numeric_hits": n_h,
            "judge_model": "test/judge", "judge_prompt_version": "judge_v1",
        })
    pd.DataFrame(scored).to_csv(
        tmp / "results" / "scored" / "openrouter__test-model__high.csv", index=False)

    # 긴 한글 파일명(리눅스 255바이트 한도)이 경로 생성에서 터지지 않는지
    from common import cache_path as _cp, pdf_path, safe_filename  # noqa: E402

    long_name = "특례 " + "가" * 200 + " 사건.pdf"
    assert len(pdf_path(long_name).name.encode()) <= 240
    assert len(_cp(long_name, 7).name.encode()) <= 240
    assert safe_filename("특례 " + "가" * 200 + " A.pdf") != \
        safe_filename("특례 " + "가" * 200 + " B.pdf"), "긴 이름이 같은 파일로 뭉개짐"
    pdf_path(long_name).write_bytes(b"%PDF-1.4 ok")   # 실제 쓰기까지 확인

    run([sys.executable, str(SRC / "report.py")], env)
    summary = (tmp / "reports" / "summary.md").read_text(encoding="utf-8")
    assert "50.0%" in summary, "전체 정확도 집계가 기대와 다릅니다"
    assert "파서 실패 의심" in summary, "image/table 실패 원인 분류가 빠졌습니다"

    print(f"\n스모크 테스트 통과 (작업 루트: {tmp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
