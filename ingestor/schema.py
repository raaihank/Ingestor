from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

TEXT_CANDIDATE_COLUMNS = [
    "text",
    "prompt",
    "content",
    "input",
    "instruction",
    "message",
    "question",
    "body",
]

LABEL_CANDIDATE_COLUMNS = [
    "label",
    "labels",
    "target",
    "class",
    "category",
    "injection_type",
    "is_malicious",
    "malicious",
    "y",
]

# Row identifiers: never part of the sample text
ID_COLUMNS = {"id", "_id", "idx", "index", "uuid", "guid"}


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def extract_text_and_label(
    row: Dict[str, Any],
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> Tuple[Optional[str], Optional[Any]]:
    """
    Pick the sample text and label from a row.

    Explicit columns win. Otherwise the first known text/label column is used, then
    the row's scalar fields are joined, then the remaining fields are dumped as JSON.
    Label and id columns are never folded into the text.
    """
    label: Optional[Any] = None
    label_key: Optional[str] = None
    if label_column:
        label, label_key = row.get(label_column), label_column
    else:
        for key in LABEL_CANDIDATE_COLUMNS:
            if row.get(key) is not None:
                label, label_key = row[key], key
                break

    if text_column:
        value = row.get(text_column)
        return (None if _is_blank(value) else str(value)), label

    for key in TEXT_CANDIDATE_COLUMNS:
        value = row.get(key)
        if not _is_blank(value):
            return str(value), label

    # Fallback: everything except label and id columns
    excluded = set(LABEL_CANDIDATE_COLUMNS) | ID_COLUMNS | {label_key}
    rest = {k: v for k, v in row.items() if k not in excluded and not _is_blank(v)}
    parts = [str(v) for v in rest.values() if isinstance(v, (str, int, float))]
    if parts:
        return " ".join(parts), label
    if rest:
        return json.dumps(rest, ensure_ascii=False, default=str), label
    return None, label


_SPLIT_PATTERNS = [
    (re.compile(r"(^|/)(train|training)(\.|/|$)", re.I), "train"),
    (re.compile(r"(^|/)(valid|validation|val|dev)(\.|/|$)", re.I), "validation"),
    (re.compile(r"(^|/)(test|testing)(\.|/|$)", re.I), "test"),
    (re.compile(r"(^|/)(eval|evaluation)(\.|/|$)", re.I), "eval"),
]


def infer_split_from_path(path_str: str) -> Optional[str]:
    for rx, name in _SPLIT_PATTERNS:
        if rx.search(path_str):
            return name
    return None

