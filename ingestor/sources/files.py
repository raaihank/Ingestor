"""Readers shared by the file-based sources (local, git, kaggle, HF repo crawl)."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import pyarrow as pa
import pyarrow.ipc as pa_ipc  # type: ignore
import pyarrow.parquet as pq  # type: ignore

from ..constants import STRUCTURED_EXTENSIONS
from ..logging_utils import log_warning
from ..schema import extract_text_and_label

# Metadata/config files that sit next to data but are not samples
NON_DATA_FILENAMES = frozenset({
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "jsconfig.json",
    "composer.json",
    "dataset_info.json",
    "dataset_infos.json",
    "dataset_dict.json",
    "state.json",
    "dataset-metadata.json",
    "datapackage.json",
})

_LICENSE_FILES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "LICENCE.md", "COPYING", "COPYING.md")
_LICENSE_PATTERNS = [
    ("MIT", re.compile(r"\bMIT License\b|Permission is hereby granted, free of charge", re.I)),
    ("Apache-2.0", re.compile(r"Apache License,?\s+Version 2\.0", re.I)),
    ("BSD-3-Clause", re.compile(r"Neither the name of .{0,200}? nor the names of its\s+contributors", re.I | re.S)),
    ("Unlicense", re.compile(r"This is free and unencumbered software released into the public domain", re.I)),
    ("CC0-1.0", re.compile(r"CC0 1\.0 Universal", re.I)),
    ("CC-BY-SA-4.0", re.compile(r"Attribution-ShareAlike 4\.0 International", re.I)),
    ("CC-BY-4.0", re.compile(r"Attribution 4\.0 International", re.I)),
    ("GPL-3.0", re.compile(r"GNU GENERAL PUBLIC LICENSE\s+Version 3", re.I)),
]


def _raise_csv_field_limit() -> None:
    # The default 128KB limit makes csv raise on long prompts
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()


def is_crawlable_data_file(rel_path: Path) -> bool:
    """Structured data file that isn't hidden or a known metadata file."""
    if any(part.startswith(".") for part in rel_path.parts):
        return False
    if rel_path.name.lower() in NON_DATA_FILENAMES:
        return False
    return rel_path.suffix.lower() in STRUCTURED_EXTENSIONS


def detect_license_file(root: Path) -> Optional[str]:
    """SPDX id of a well-known license file at the repository root, if recognizable."""
    for name in _LICENSE_FILES:
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")[:20000]
        for spdx, pattern in _LICENSE_PATTERNS:
            if pattern.search(text):
                return spdx
    return None


def _as_row(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        return {"text": obj}
    return {"text": json.dumps(obj, ensure_ascii=False)}


def _iter_arrow(path: Path) -> Iterator[Tuple[Optional[int], Dict[str, Any]]]:
    with path.open("rb") as f:
        try:
            reader = pa_ipc.open_file(f)
            batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
        except pa.ArrowInvalid:
            f.seek(0)
            batches = iter(pa_ipc.open_stream(f))
        idx = 0
        for batch in batches:
            for row in batch.to_pylist():
                yield idx, row
                idx += 1


def iter_rows(path: Path) -> Iterator[Tuple[Optional[int], Dict[str, Any]]]:
    """Yield (row index, row) for each record in a data file.

    Plain-text files and single-object JSON files yield one row with index None.
    """
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    obj = line
                yield idx, _as_row(obj)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
        if isinstance(data, list):
            for idx, obj in enumerate(data):
                yield idx, _as_row(obj)
        else:
            yield None, _as_row(data)
    elif suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            for idx, row in enumerate(csv.DictReader(f, delimiter=delimiter)):
                yield idx, dict(row)
    elif suffix == ".parquet":
        idx = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                yield idx, row
                idx += 1
    elif suffix == ".arrow":
        yield from _iter_arrow(path)
    else:
        yield None, {"text": path.read_text(encoding="utf-8", errors="ignore")}


def file_items(
    path: Path,
    *,
    source: str,
    id_prefix: str,
    meta: Dict[str, Any],
    spec: str,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Pipeline items for every row of a data file.

    `spec` is the configured source key (dataset spec, repo URL or glob) used to
    resolve overrides; it is not written to the output.
    """
    warned = False
    for idx, row in iter_rows(path):
        if text_column and text_column not in row and not warned:
            columns = ", ".join(str(k) for k in row)[:200]
            log_warning(f"{path}: text column {text_column!r} not found (columns: {columns}); such rows are rejected as empty")
            warned = True
        text, label = extract_text_and_label(row, text_column, label_column)
        yield {
            "source": source,
            "source_id": id_prefix if idx is None else f"{id_prefix}:{idx}",
            "raw": "" if text is None else text,
            "label": label,
            "meta": dict(meta),
            "spec": spec,
        }
