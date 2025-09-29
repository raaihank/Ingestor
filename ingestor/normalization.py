from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional, Union, cast, Tuple

from unidecode import unidecode

HOMOGLYPH_MAP: Dict[str, str] = {
    """: '"',
    """: '"',
    "'": "'",
    "'": "'",
    "—": "-",
    "–": "-",
}

# Extended homoglyph patterns for evasion detection
EVASION_PATTERNS = {
    # Zero-width characters
    "\u200b",  # Zero Width Space
    "\u200c",  # Zero Width Non-Joiner
    "\u200d",  # Zero Width Joiner
    "\u2060",  # Word Joiner
    "\ufeff",  # Zero Width No-Break Space
    # BiDi override characters
    "\u202a",  # Left-to-Right Embedding
    "\u202b",  # Right-to-Left Embedding
    "\u202c",  # Pop Directional Formatting
    "\u202d",  # Left-to-Right Override
    "\u202e",  # Right-to-Left Override
    "\u2066",  # Left-to-Right Isolate
    "\u2067",  # Right-to-Left Isolate
    "\u2068",  # First Strong Isolate
    "\u2069",  # Pop Directional Isolate
}


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


def has_evasion_markers(text1: str, text2: str) -> Tuple[bool, str]:
    """
    Check if two texts differ primarily by evasion techniques.

    Args:
        text1: First text (light normalized)
        text2: Second text (light normalized)

    Returns:
        (is_evasion_variant, evasion_type) tuple
    """
    if not text1 or not text2:
        return False, ""

    # Remove all evasion characters from both texts
    clean_text1 = text1
    clean_text2 = text2

    for pattern in EVASION_PATTERNS:
        clean_text1 = clean_text1.replace(pattern, "")
        clean_text2 = clean_text2.replace(pattern, "")

    # If texts are identical after removing evasion characters,
    # they're likely evasion variants
    if clean_text1 == clean_text2:
        # Determine evasion type
        evasion_list = list(EVASION_PATTERNS)
        has_zw = any(pattern in text1 or pattern in text2
                     for pattern in evasion_list[:5])  # Zero-width chars
        has_bidi = any(pattern in text1 or pattern in text2
                       for pattern in evasion_list[5:])  # BiDi chars

        if has_zw and has_bidi:
            return True, "zero_width_bidi"
        elif has_zw:
            return True, "zero_width"
        elif has_bidi:
            return True, "bidi_override"
        else:
            return True, "other_evasion"

    # Check for Base64/hex wrapping patterns
    base64_pattern = r"^[A-Za-z0-9+/]+=*$"
    hex_pattern = r"^[0-9a-fA-F]+$"
    
    if (re.match(base64_pattern, text1.replace(" ", "")) or
        re.match(hex_pattern, text1.replace(" ", "")) or
        re.match(base64_pattern, text2.replace(" ", "")) or
        re.match(hex_pattern, text2.replace(" ", ""))):
        return True, "encoding_wrap"

    return False, ""


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
