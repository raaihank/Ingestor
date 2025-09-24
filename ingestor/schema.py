from __future__ import annotations

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


def extract_text_and_label(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
    # Prefer explicit text-like columns
    text: Optional[str] = None
    for key in TEXT_CANDIDATE_COLUMNS:
        if key in row and row[key] not in (None, ""):
            try:
                text = str(row[key])
                break
            except Exception:
                continue

    # Fallback: join string-like fields
    if text is None:
        parts = []
        for k, v in row.items():
            if isinstance(v, (str, int, float)):
                parts.append(str(v))
            elif isinstance(v, (list, dict)):
                continue
        if parts:
            text = " ".join(parts)

    label: Optional[Any] = None
    for key in LABEL_CANDIDATE_COLUMNS:
        if key in row and row[key] is not None:
            label = row[key]
            break

    # Convert boolean-like malicious fields to 0/1
    if label is None and "malicious" in row:
        label = row["malicious"]

    return text, label


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


