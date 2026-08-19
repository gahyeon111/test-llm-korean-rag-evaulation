"""0단계: HF 데이터셋을 data/dataset.csv, data/documents.csv 로 내려받는다.

착수 전 결정사항(스펙 §6 "데이터 수급 방식")의 구현:
기본은 `datasets` 라이브러리, 실패 시 `huggingface_hub` 로 원본 CSV 직접 다운로드.

출력
  data/dataset.csv    qid, domain, context_type, question, target_answer,
                      target_file_name, target_page_no
  data/documents.csv  domain, file_name, url
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from common import (DATASET_CSV, DOCUMENTS_ALIASES, DATASET_ALIASES, DOCUMENTS_CSV,
                    normalize_columns)

REPO_ID = "allganize/RAG-Evaluation-Dataset-KO"
DATASET_REQUIRED = ["question", "target_answer", "target_file_name",
                    "target_page_no", "context_type", "domain"]


def load_via_datasets(repo_id: str) -> pd.DataFrame:
    from datasets import load_dataset  # 지연 import: 이 경로에서만 필요

    ds = load_dataset(repo_id)
    frames = [split.to_pandas() for split in ds.values()]
    return pd.concat(frames, ignore_index=True)


def load_via_hub(repo_id: str, filename: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return pd.read_csv(path)


def derive_documents(dataset: pd.DataFrame) -> pd.DataFrame:
    """documents 파일을 못 구한 경우 dataset 에서 문서 목록만 뽑아 둔다 (url 은 공란)."""
    cols = ["domain", "target_file_name"] if "domain" in dataset.columns else ["target_file_name"]
    docs = dataset[cols].drop_duplicates().rename(columns={"target_file_name": "file_name"})
    if "domain" not in docs.columns:
        docs["domain"] = ""
    docs["url"] = ""
    return docs[["domain", "file_name", "url"]].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--source", choices=["hf", "csv"], default="hf",
                    help="hf=datasets 라이브러리, csv=repo 내 CSV 직접 다운로드")
    ap.add_argument("--dataset-file", default="rag_evaluation_result.csv",
                    help="--source csv 일 때 문항 CSV 파일명")
    ap.add_argument("--documents-file", default="documents.csv",
                    help="문서 메타 CSV 파일명 (없으면 dataset 에서 유도)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if DATASET_CSV.exists() and not args.overwrite:
        print(f"이미 존재: {DATASET_CSV} (덮어쓰려면 --overwrite)")
        return 0

    if args.source == "hf":
        try:
            dataset = load_via_datasets(args.repo_id)
        except Exception as exc:  # noqa: BLE001 - 네트워크/포맷 어떤 실패든 CSV 로 폴백
            print(f"datasets 로드 실패 ({exc}) → CSV 직접 다운로드로 폴백", file=sys.stderr)
            dataset = load_via_hub(args.repo_id, args.dataset_file)
    else:
        dataset = load_via_hub(args.repo_id, args.dataset_file)

    dataset = normalize_columns(dataset, DATASET_ALIASES, required=DATASET_REQUIRED)
    dataset = dataset[DATASET_REQUIRED].copy()
    dataset["target_page_no"] = (
        pd.to_numeric(dataset["target_page_no"], errors="coerce").astype("Int64")
    )
    bad_pages = int(dataset["target_page_no"].isna().sum())
    if bad_pages:
        print(f"경고: target_page_no 파싱 실패 {bad_pages}건 (해당 문항은 파싱 대상에서 제외됨)")
    # qid 는 여기서 한 번만 부여하고 이후 전 단계에서 그대로 사용한다.
    dataset.insert(0, "qid", [f"q{i:04d}" for i in range(1, len(dataset) + 1)])
    dataset.to_csv(DATASET_CSV, index=False)
    print(f"저장: {DATASET_CSV} ({len(dataset)}문항)")

    try:
        documents = load_via_hub(args.repo_id, args.documents_file)
        documents = normalize_columns(documents, DOCUMENTS_ALIASES, required=["file_name"])
        for col in ("domain", "url"):
            if col not in documents.columns:
                documents[col] = ""
        documents = documents[["domain", "file_name", "url"]]
    except Exception as exc:  # noqa: BLE001
        print(f"documents 파일 다운로드 실패 ({exc}) → dataset 에서 문서 목록 유도", file=sys.stderr)
        documents = derive_documents(dataset)
        print("주의: url 컬럼이 비어 있습니다. 수동으로 채운 뒤 download_pdfs.py 를 실행하세요.")

    documents.to_csv(DOCUMENTS_CSV, index=False)
    print(f"저장: {DOCUMENTS_CSV} ({len(documents)}문서)")

    print("\n[요약]")
    print(dataset.groupby(["domain", "context_type"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
