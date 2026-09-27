from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Set, Tuple

from unidecode import unidecode

HOMOGLYPH_MAP: Dict[str, str] = {
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u2014": "-",  # em dash
    "\u2013": "-",  # en dash
}

# Invisible characters used to hide or reorder text in evasion attacks
ZERO_WIDTH_CHARS = frozenset({
    "\u200b",  # Zero Width Space
    "\u200c",  # Zero Width Non-Joiner
    "\u200d",  # Zero Width Joiner
    "\u2060",  # Word Joiner
    "\ufeff",  # Zero Width No-Break Space
})
BIDI_CHARS = frozenset({
    "\u202a",  # Left-to-Right Embedding
    "\u202b",  # Right-to-Left Embedding
    "\u202c",  # Pop Directional Formatting
    "\u202d",  # Left-to-Right Override
    "\u202e",  # Right-to-Left Override
    "\u2066",  # Left-to-Right Isolate
    "\u2067",  # Right-to-Left Isolate
    "\u2068",  # First Strong Isolate
    "\u2069",  # Pop Directional Isolate
})
EVASION_PATTERNS = ZERO_WIDTH_CHARS | BIDI_CHARS

_INVISIBLE_RE = re.compile("[" + "".join(re.escape(c) for c in sorted(EVASION_PATTERNS)) + "]")
_ZERO_WIDTH_RE = re.compile("[" + "".join(re.escape(c) for c in sorted(ZERO_WIDTH_CHARS)) + "]")
_BIDI_RE = re.compile("[" + "".join(re.escape(c) for c in sorted(BIDI_CHARS)) + "]")
_NON_ASCII_RE = re.compile("[^" + chr(0) + "-" + chr(127) + "]")
_HEX_TOKEN_RE = re.compile(r"(?:0x)?[0-9a-f]+", re.I)
_BASE64_TOKEN_RE = re.compile(r"[a-z0-9+/_-]+={0,2}", re.I)


def normalize_text_light(text: str) -> str:
    """
    Light normalization for near-duplicate detection that preserves evasion variants.

    - Uses NFC (not NFKC) to preserve more character distinctions
    - No homoglyph mapping to preserve visual attack variants
    - No unidecode to preserve mixed-script evasions
    - Keeps zero-width and BiDi characters that may be part of evasions
    """
    if not text:
        return text

    # Use NFC (not NFKC) to preserve more distinctions
    text = unicodedata.normalize("NFC", text)

    # Convert to lowercase for case-insensitive comparison
    text = text.lower()

    # Only collapse whitespace, preserve other characters
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_text_heavy(text: str) -> str:
    """
    Heavy normalization for exact duplicate detection.

    This is the original normalize_text function - applies aggressive
    normalization to catch exact duplicates with different encodings.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFKC", text)
    # Apply homoglyph replacements
    for old_char, new_char in HOMOGLYPH_MAP.items():
        text = text.replace(old_char, new_char)
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Alias for backward compatibility
normalize_text = normalize_text_heavy


def label_key(value: Any) -> str:
    """Text form of a label value: True -> "true", 1.0 -> "1", "x" -> "x"."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


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


def strip_invisible(text: str) -> str:
    """Remove zero-width and BiDi control characters."""
    return _INVISIBLE_RE.sub("", text)


def fold_typography(text: str) -> str:
    """Map typographic quotes and dashes to their ASCII forms."""
    for old_char, new_char in HOMOGLYPH_MAP.items():
        text = text.replace(old_char, new_char)
    return text


def _fold_char(match: "re.Match[str]") -> str:
    # Characters without an ASCII form (e.g. emoji) are kept so they still distinguish texts
    ascii_form = unidecode(match.group())
    return ascii_form if ascii_form.strip() else match.group()


def _fold_lookalikes(text: str) -> str:
    """Fold compatibility forms, homoglyphs and accents to ASCII."""
    text = _NON_ASCII_RE.sub(_fold_char, unicodedata.normalize("NFKC", text))
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokens(text: str) -> Set[str]:
    """Whitespace tokens without surrounding punctuation."""
    return {token.strip("\"'`()[]{}<>,.;:!?") for token in text.split()}


def _looks_encoded(token: str) -> bool:
    """Heuristic for a base64/hex payload token."""
    if len(token) < 16:
        return False
    if _HEX_TOKEN_RE.fullmatch(token):
        return True
    # Require a digit or padding so long or slash-joined plain words don't qualify
    return bool(_BASE64_TOKEN_RE.fullmatch(token)) and (
        any(c.isdigit() for c in token) or token.endswith("=")
    )


def classify_evasion(original: str, candidate: str) -> str:
    """
    Name the evasion technique that distinguishes two near-identical texts.

    Args:
        original: Text already kept (light normalized)
        candidate: Text being checked (light normalized)

    Returns:
        "zero_width", "bidi_override", "zero_width_bidi", "homoglyph" or
        "encoding_wrap"; "" when the texts don't differ by an evasion technique.
    """
    if not original or not candidate or original == candidate:
        return ""
    # Curly vs straight quotes and dash styles are formatting, not evasion
    if fold_typography(original) == fold_typography(candidate):
        return ""

    if strip_invisible(original) == strip_invisible(candidate):
        both = original + candidate
        has_zw = _ZERO_WIDTH_RE.search(both) is not None
        has_bidi = _BIDI_RE.search(both) is not None
        if has_zw and has_bidi:
            return "zero_width_bidi"
        return "zero_width" if has_zw else "bidi_override"

    # Same text once compatibility forms, homoglyphs and accents are folded
    if _fold_lookalikes(original) == _fold_lookalikes(candidate):
        return "homoglyph"

    # A differing token that looks like an encoded payload
    differing = _tokens(original) ^ _tokens(candidate)
    if any(_looks_encoded(token) for token in differing):
        return "encoding_wrap"

    return ""


def has_evasion_markers(text1: str, text2: str) -> Tuple[bool, str]:
    """
    Check if two texts differ primarily by evasion techniques.

    Args:
        text1: First text (light normalized)
        text2: Second text (light normalized)

    Returns:
        (is_evasion_variant, evasion_type) tuple
    """
    evasion_type = classify_evasion(text1, text2)
    return bool(evasion_type), evasion_type


def shingles(text: str, k: int) -> Set[str]:
    """Character k-grams of `text`; a text shorter than k is a single shingle."""
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def get_shingle_size(text_length: int) -> int:
    """
    Get appropriate shingle size based on text length.

    Args:
        text_length: Length of the text

    Returns:
        Appropriate k-gram size (3, 4, or 5)
    """
    if text_length < 40:
        return 3
    elif text_length <= 200:
        return 4
    else:
        return 5


def get_similarity_threshold(text_length: int) -> float:
    """
    Get appropriate similarity threshold based on text length.

    Args:
        text_length: Length of the text

    Returns:
        Similarity threshold for near-duplicate detection
    """
    if text_length < 40:
        return 0.95  # Very high threshold for short texts
    elif text_length <= 200:
        return 0.91  # Medium threshold for medium texts
    else:
        return 0.89  # Lower threshold for long texts
