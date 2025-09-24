from __future__ import annotations

import csv
import glob as pyglob
import json
from pathlib import Path
from typing import Dict, Generator, Iterable

import pandas as pd  # type: ignore
import pyarrow.ipc as pa_ipc  # type: ignore

from ..constants import STRUCTURED_EXTENSIONS, TEXT_EXTENSIONS
from ..schema import extract_text_and_label, infer_split_from_path

DATA_EXTS = STRUCTURED_EXTENSIONS | TEXT_EXTENSIONS


def _iter_paths(pattern: str) -> Iterable[Path]:
    for p in pyglob.glob(pattern, recursive=True):
        pth = Path(p)
        if pth.is_file() and pth.suffix.lower() in DATA_EXTS:
            yield pth


def iter_local(globs: list[str]) -> Generator[Dict, None, None]:
    for pattern in globs:
        for path in _iter_paths(pattern):
            suffix = path.suffix.lower()
            split = infer_split_from_path(str(path))
            try:
                if suffix in {".jsonl", ".ndjson"}:
                    with path.open("r", encoding="utf-8", errors="ignore") as f:
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
                                "source": "local",
                                "source_id": f"{path}:{idx}",
                                "raw": str(text),
                                "label": label,
                                "meta": {"path": str(path), **({"split": split} if split else {})},
                            }
                elif suffix == ".json":
                    obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
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
                                "source": "local",
                                "source_id": f"{path}:{idx}",
                                "raw": str(text),
                                "label": label,
                                "meta": {"path": str(path), **({"split": split} if split else {})},
                            }
                    else:
                        text, label = extract_text_and_label(obj)
                        if text is None:
                            text = json.dumps(obj, ensure_ascii=False)
                        yield {
                            "source": "local",
                            "source_id": str(path),
                            "raw": str(text),
                            "label": label,
                            "meta": {"path": str(path), **({"split": split} if split else {})},
                        }
                elif suffix in {".csv", ".tsv"}:
                    delimiter = "," if suffix == ".csv" else "\t"
                    with path.open("r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f, delimiter=delimiter)
                        for idx, row in enumerate(reader):
                            text, label = extract_text_and_label(row)
                            if text is None:
                                text = " ".join(str(v) for v in row.values())
                            yield {
                                "source": "local",
                                "source_id": f"{path}:{idx}",
                                "raw": str(text),
                                "label": label,
                                "meta": {"path": str(path), **({"split": split} if split else {})},
                            }
                elif suffix == ".parquet":
                    df = pd.read_parquet(path)
                    for idx, row in df.iterrows():
                        row_dict = row.to_dict()
                        text, label = extract_text_and_label(row_dict)
                        if text is None:
                            text = " ".join(str(v) for v in row_dict.values())
                        yield {
                            "source": "local",
                            "source_id": f"{path}:{idx}",
                            "raw": str(text),
                            "label": label,
                            "meta": {"path": str(path), **({"split": split} if split else {})},
                        }
                elif suffix == ".arrow":
                    with path.open("rb") as f:
                        try:
                            reader2 = pa_ipc.open_file(f)
                        except Exception:
                            f.seek(0)
                            reader2 = pa_ipc.open_stream(f)
                        table = reader2.read_all()
                    df = table.to_pandas()
                    for idx, row in df.iterrows():
                        row_dict = row.to_dict()
                        text, label = extract_text_and_label(row_dict)
                        if text is None:
                            text = " ".join(str(v) for v in row_dict.values())
                        yield {
                            "source": "local",
                            "source_id": f"{path}:{idx}",
                            "raw": str(text),
                            "label": label,
                            "meta": {"path": str(path), **({"split": split} if split else {})},
                        }
                else:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    yield {
                        "source": "local",
                        "source_id": str(path),
                        "raw": text,
                        "label": None,
                        "meta": {"path": str(path), **({"split": split} if split else {})},
                    }
            except Exception:
                continue
