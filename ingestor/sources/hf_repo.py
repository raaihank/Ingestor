from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Generator, Iterable

from huggingface_hub import HfApi, hf_hub_download

from ..schema import extract_text_and_label, infer_split_from_path

DATA_EXTS = {".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet", ".arrow", ".txt"}


def _list_dataset_files(dataset_id: str) -> Iterable[str]:
    api = HfApi()
    # list_repo_files does not recurse; list_repo_tree provides paths
    try:
        tree = api.list_repo_tree(repo_id=dataset_id, repo_type="dataset", recursive=True)
        for e in tree:
            # mypy: RepoFile has attribute 'type'; RepoFolder has no 'type', so guard with getattr
            if getattr(e, "type", None) == "file" and Path(e.path).suffix.lower() in DATA_EXTS:
                yield e.path
    except Exception:
        # fallback
        for p in api.list_repo_files(repo_id=dataset_id, repo_type="dataset"):
            if Path(p).suffix.lower() in DATA_EXTS:
                yield p


def iter_hf_repo(dataset_id: str) -> Generator[Dict, None, None]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    for rel_path in _list_dataset_files(dataset_id):
        try:
            local_path = hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                filename=rel_path,
                token=token,
            )
        except Exception:
            # Likely gated
            print(
                f"[quality] gated_or_unavailable: https://huggingface.co/datasets/{dataset_id} — accept terms then re-run",
                flush=True,
            )
            continue

        p = Path(local_path)
        suffix = p.suffix.lower()
        split = infer_split_from_path(rel_path)

        try:
            if suffix in {".jsonl", ".ndjson"}:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            obj = {"text": line}
                        text, label = extract_text_and_label(obj)
                        if text is None:
                            text = json.dumps(obj, ensure_ascii=False)
                        yield {
                            "source": f"hf:{dataset_id}",
                            "source_id": f"{rel_path}:{idx}",
                            "raw": str(text),
                            "label": label,
                            "meta": {"path": rel_path, "dataset": dataset_id, **({"split": split} if split else {})},
                        }
            elif suffix == ".json":
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(obj, list):
                    for idx, row in enumerate(obj):
                        if isinstance(row, dict):
                            text, label = extract_text_and_label(row)
                            if text is None:
                                text = json.dumps(row, ensure_ascii=False)
                        else:
                            text = str(row)
                            label = None
                        yield {
                            "source": f"hf:{dataset_id}",
                            "source_id": f"{rel_path}:{idx}",
                            "raw": str(text),
                            "label": label,
                            "meta": {"path": rel_path, "dataset": dataset_id, **({"split": split} if split else {})},
                        }
                else:
                    text, label = extract_text_and_label(obj)
                    if text is None:
                        text = json.dumps(obj, ensure_ascii=False)
                    yield {
                        "source": f"hf:{dataset_id}",
                        "source_id": rel_path,
                        "raw": str(text),
                        "label": label,
                        "meta": {"path": rel_path, "dataset": dataset_id, **({"split": split} if split else {})},
                    }
            else:
                # treat as text-like (csv/tsv/parquet/arrow handled by datasets usually; fallback to raw read)
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                yield {
                    "source": f"hf:{dataset_id}",
                    "source_id": rel_path,
                    "raw": text,
                    "label": None,
                    "meta": {"path": rel_path, "dataset": dataset_id, **({"split": split} if split else {})},
                }
        except Exception:
            continue


