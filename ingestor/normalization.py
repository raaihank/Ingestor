from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional, Union, cast

from unidecode import unidecode

HOMOGLYPH_MAP: Dict[str, str] = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "—": "-",
    "–": "-",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    # Cast to the full translation map type expected by maketrans
    trans_map = cast(
        Dict[Union[str, int], Optional[Union[str, int]]],
        dict(HOMOGLYPH_MAP),
    )
    text = text.translate(str.maketrans(trans_map))
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label: str) -> str:
    """
    Normalize label to lowercase with underscores instead of spaces.
    
    Examples:
        "Prompt Injection" -> "prompt_injection"
        "Safe Content" -> "safe_content"
        "JAILBREAK" -> "jailbreak"
        "benign-text" -> "benign_text"
    """
    if not label:
        return label
    
    # Convert to string and strip whitespace
    label = str(label).strip()
    
    # Convert to lowercase
    label = label.lower()
    
    # Replace spaces, hyphens, and other separators with underscores
    label = re.sub(r"[\s\-\.]+", "_", label)
    
    # Remove any non-alphanumeric characters except underscores
    label = re.sub(r"[^a-z0-9_]", "", label)
    
    # Remove multiple consecutive underscores
    label = re.sub(r"_+", "_", label)
    
    # Remove leading/trailing underscores
    label = label.strip("_")
    
    return label
