"""OpenRouter 에 실제로 존재하는 모델 ID 인지 확인하고, provider·양자화를 조회한다.

스펙에 적힌 모델 ID 가 카탈로그에 없으면 400 (`not a valid model ID`) 이 난다.
파싱을 돌리기 전에 세 모델(파서/평가대상/judge)을 한 번에 확인하는 용도.

  python tools/check_models.py                      # 설정된 기본 모델 3개 확인
  python tools/check_models.py --search gemini      # 이름으로 후보 찾기
  python tools/check_models.py --endpoints <model>  # provider·양자화 목록 (fp8 고정 확인)
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests  # noqa: E402

from common import require_api_key  # noqa: E402

MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULTS = {
    "파서 (parse_vlm)": "google/gemini-3.1-pro",
    "평가 대상 (run_model)": "deepseek/deepseek-v4-flash-0731",
    "judge (judge)": "google/gemini-3.7-flash",
}


def fetch_models(api_key: str) -> list[dict]:
    resp = requests.get(MODELS_URL, headers={"Authorization": f"Bearer {api_key}"},
                        timeout=60)
    resp.raise_for_status()
    return resp.json().get("data", [])


def price_of(model: dict) -> str:
    pricing = model.get("pricing") or {}
    try:
        prompt = float(pricing.get("prompt", 0)) * 1_000_000
        completion = float(pricing.get("completion", 0)) * 1_000_000
    except (TypeError, ValueError):
        return "-"
    return f"${prompt:.2f}/${completion:.2f} per 1M"


def suggest(model_id: str, ids: list[str], n: int = 10) -> list[str]:
    """카탈로그에 없을 때 대체 후보. 같은 계열 안에서 이름이 가까운 순으로."""
    family = model_id.split("/")[0]
    keyword = model_id.split("/")[-1].split("-")[0]
    same = [i for i in ids if i.startswith(family + "/") and keyword in i]
    # 알파벳순으로 자르면 구버전만 남는다 — 이름 유사도 순으로 정렬
    same.sort(key=lambda i: -difflib.SequenceMatcher(None, model_id, i).ratio())
    close = difflib.get_close_matches(model_id, ids, n=n, cutoff=0.4)
    return list(dict.fromkeys(same + close))[:n]


def is_vision(model: dict) -> bool:
    modalities = (model.get("architecture") or {}).get("input_modalities") or []
    return "image" in [str(m).lower() for m in modalities]


def show_endpoints(api_key: str, model_id: str) -> None:
    url = f"{MODELS_URL}/{model_id}/endpoints"
    resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    if resp.status_code != 200:
        print(f"endpoints 조회 실패: HTTP {resp.status_code} {resp.text[:200]}")
        return
    data = resp.json().get("data", {})
    endpoints = data.get("endpoints", [])
    print(f"\n{model_id} — provider {len(endpoints)}개")
    print(f"{'provider':<28} {'양자화':<10} {'context':>9}  가격(1M in/out)")
    for ep in endpoints:
        pricing = ep.get("pricing") or {}
        try:
            price = (f"${float(pricing.get('prompt', 0)) * 1e6:.2f}/"
                     f"${float(pricing.get('completion', 0)) * 1e6:.2f}")
        except (TypeError, ValueError):
            price = "-"
        print(f"{str(ep.get('provider_name', '?')):<28} "
              f"{str(ep.get('quantization') or '-'):<10} "
              f"{str(ep.get('context_length', '-')):>9}  {price}")
    quants = {str(ep.get("quantization")).lower() for ep in endpoints}
    if "fp8" in quants:
        print("\n✅ fp8 provider 있음 — run_model.py --quantization fp8 로 고정 가능")
    else:
        print(f"\n⚠️ fp8 provider 없음 (있는 양자화: {sorted(q for q in quants if q != 'none')})")
        print("   run_model.py --quantization <있는 값> 으로 바꾸거나 빈 값으로 고정 해제하세요.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", default="", help="모델 ID/이름에 이 문자열이 든 것만 나열")
    ap.add_argument("--vision", action="store_true",
                    help="이미지 입력이 가능한 모델만 (파서용 모델 고를 때)")
    ap.add_argument("--endpoints", default="", help="이 모델의 provider·양자화 조회")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    api_key = require_api_key()
    if args.endpoints:
        show_endpoints(api_key, args.endpoints)
        return 0

    models = fetch_models(api_key)
    ids = sorted(m["id"] for m in models)
    print(f"OpenRouter 카탈로그: 모델 {len(ids)}개\n")

    if args.search:
        needle = args.search.lower()
        hits = [m for m in models
                if needle in m["id"].lower() or needle in str(m.get("name", "")).lower()]
        if args.vision:
            hits = [m for m in hits if is_vision(m)]
        hits.sort(key=lambda m: m["id"])
        print(f"'{args.search}' 검색 결과 {len(hits)}건"
              + (" (이미지 입력 가능만)" if args.vision else "") + ":")
        for m in hits[:args.limit]:
            tag = "[vision]" if is_vision(m) else "        "
            print(f"  {m['id']:<50} {tag} {price_of(m)}")
        return 0

    ok = True
    for label, model_id in DEFAULTS.items():
        if model_id in ids:
            print(f"✅ {label}: {model_id}")
            continue
        ok = False
        print(f"❌ {label}: {model_id} — 카탈로그에 없음")
        by_id = {m["id"]: m for m in models}
        for cand in suggest(model_id, ids):
            tag = " [vision]" if is_vision(by_id.get(cand, {})) else ""
            print(f"     후보: {cand}{tag}")
    if not ok:
        print("\n실제 ID 를 확인한 뒤 각 스크립트의 --model / --judge-model 로 지정하거나,")
        print("src/parse_vlm.py·run_model.py·judge.py 의 DEFAULT_* 상수를 고치세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
