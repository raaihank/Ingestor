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
