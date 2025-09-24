from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Generator, Iterable

import pandas as pd  # type: ignore
import pyarrow.ipc as pa_ipc  # type: ignore
from git import Repo

TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml", ".jsonl", ".ndjson", ".parquet", ".arrow"}


def _iter_text_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            yield path


def iter_git_repo(repo_url: str) -> Generator[Dict, None, None]:
    tmpdir = Path(tempfile.mkdtemp(prefix="ingest_git_"))
    try:
        Repo.clone_from(repo_url, tmpdir)
        for file_path in _iter_text_files(tmpdir):
            rel = file_path.relative_to(tmpdir)
            suffix = file_path.suffix.lower()
            try:
                if suffix in {".jsonl", ".ndjson"}:
                    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                        for idx, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                obj = {"text": line}
                            text = obj.get("text") or obj.get("prompt") or obj.get("content") or json.dumps(obj, ensure_ascii=False)
                            yield {
                                "source": f"git:{repo_url}",
                                "source_id": f"{rel}:{idx}",
                                "raw": str(text),
                                "label": obj.get("label"),
                                "meta": {"path": str(rel), "dataset": repo_url},
                            }
                elif suffix == ".json":
                    obj = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
                    if isinstance(obj, list):
                        for idx, row in enumerate(obj):
                            if isinstance(row, dict):
                                text = row.get("text") or row.get("prompt") or row.get("content") or json.dumps(row, ensure_ascii=False)
                                label = row.get("label")
                            else:
                                text = str(row)
                                label = None
                            yield {
                                "source": f"git:{repo_url}",
                                "source_id": f"{rel}:{idx}",
                                "raw": str(text),
                                "label": label,
                                "meta": {"path": str(rel), "dataset": repo_url},
                            }
                    else:
                        text = obj.get("text") or obj.get("prompt") or obj.get("content") or json.dumps(obj, ensure_ascii=False)
                        yield {
                            "source": f"git:{repo_url}",
                            "source_id": str(rel),
                            "raw": str(text),
                            "label": obj.get("label"),
                            "meta": {"path": str(rel), "dataset": repo_url},
                        }
                elif suffix in {".csv", ".tsv"}:
                    delimiter = "," if suffix == ".csv" else "\t"
                    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f, delimiter=delimiter)
                        for idx, row in enumerate(reader):
                            text = row.get("text") or row.get("prompt") or row.get("content")
                            if not text:
                                text = " ".join(str(v) for v in row.values())
                            yield {
                                "source": f"git:{repo_url}",
                                "source_id": f"{rel}:{idx}",
                                "raw": str(text),
                                "label": row.get("label"),
                                "meta": {"path": str(rel), "dataset": repo_url},
                            }
                elif suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                    for idx, row in df.iterrows():
                        row_dict = row.to_dict()
                        text = row_dict.get("text") or row_dict.get("prompt") or row_dict.get("content") or " ".join(str(v) for v in row_dict.values())
                        yield {
                            "source": f"git:{repo_url}",
                            "source_id": f"{rel}:{idx}",
                            "raw": str(text),
                            "label": row_dict.get("label"),
                            "meta": {"path": str(rel), "dataset": repo_url},
                        }
                elif suffix == ".arrow":
                    with file_path.open("rb") as f:
                        try:
                            reader2 = pa_ipc.open_file(f)
                        except Exception:
                            f.seek(0)
                            reader2 = pa_ipc.open_stream(f)
                        table = reader2.read_all()
                    df = table.to_pandas()
                    for idx, row in df.iterrows():
                        row_dict = row.to_dict()
                        text = row_dict.get("text") or row_dict.get("prompt") or row_dict.get("content") or " ".join(str(v) for v in row_dict.values())
                        yield {
                            "source": f"git:{repo_url}",
                            "source_id": f"{rel}:{idx}",
                            "raw": str(text),
                            "label": row_dict.get("label"),
                            "meta": {"path": str(rel), "dataset": repo_url},
                        }
                else:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    yield {
                        "source": f"git:{repo_url}",
                        "source_id": str(rel),
                        "raw": text,
                        "label": None,
                        "meta": {"path": str(rel), "dataset": repo_url},
                    }
            except Exception:
                continue
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

