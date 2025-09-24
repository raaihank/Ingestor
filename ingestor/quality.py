from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasketch import MinHash, MinHashLSH

try:
    import fasttext  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fasttext = None  # type: ignore

try:
    from langdetect import detect_langs  # type: ignore
except Exception:  # pragma: no cover
    detect_langs = None  # type: ignore


def calculate_entropy(text: str) -> float:
    freq = Counter(text)
    length = len(text) or 1
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def filter_by_entropy(text: str, min_entropy: float = 2.5) -> bool:
    return calculate_entropy(text) >= min_entropy


def validate_length(text: str, min_len: int = 10, max_len: int = 10000) -> bool:
    text_length = len(text.strip())
    if text_length < min_len:
        return False
    if text_length > max_len:
        return False
    return True


@dataclass
class NearDuplicateDetector:
    threshold: float = 0.85
    num_perm: int = 128

    def __post_init__(self) -> None:
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.seen_hashes: Dict[str, MinHash] = {}

    def create_minhash(self, text: str) -> MinHash:
        minhash = MinHash(num_perm=self.num_perm)
        text_lower = text.lower()
        for i in range(max(0, len(text_lower) - 2)):
            shingle = text_lower[i : i + 3]
            minhash.update(shingle.encode("utf-8"))
        return minhash

    def is_duplicate(self, text: str) -> Tuple[bool, float]:
        minhash = self.create_minhash(text)
        similar = self.lsh.query(minhash)
        if similar:
            closest = similar[0]
            similarity = minhash.jaccard(self.seen_hashes[closest])
            return True, similarity
        doc_id = f"doc_{len(self.seen_hashes)}"
        self.lsh.insert(doc_id, minhash)
        self.seen_hashes[doc_id] = minhash
        return False, 0.0


class LicenseValidator:
    def __init__(self) -> None:
        self.approved_licenses = {
            "MIT",
            "Apache-2.0",
            "BSD-3-Clause",
            "CC0-1.0",
            "CC-BY-4.0",
            "CC-BY-SA-4.0",
            "Unlicense",
        }
        self.rejected_licenses = {"GPL-3.0", "CC-BY-NC", "PROPRIETARY", "UNKNOWN"}

    def validate_source_license(self, source_metadata: Dict) -> bool:
        license_id = source_metadata.get("license", "UNKNOWN")
        if license_id in self.approved_licenses:
            return True
        if license_id in self.rejected_licenses:
            return False
        return False


class LanguageFilter:
    def __init__(
        self,
        allowed_languages: Optional[List[str]] = None,
        confidence: float = 0.7,
        model_path: Optional[Path] = None,
    ) -> None:
        self.allowed = set(allowed_languages or ["en"])
        self.confidence = confidence
        self.model = None
        self.last_lang: Optional[str] = None
        self.last_conf: Optional[float] = None
        if fasttext is not None:
            lid_path = model_path or Path(os.getenv("FASTTEXT_LID_PATH", "lid.176.bin"))
            if lid_path.exists():
                try:
                    # Loading can be expensive; avoid if file missing.
                    self.model = fasttext.load_model(str(lid_path))  # type: ignore
                except Exception:
                    self.model = None

    def is_allowed_language(self, text: str) -> bool:
        if not text:
            self.last_lang, self.last_conf = None, None
            return False
        # Prefer FastText if available
        if self.model is not None:
            labels, probs = self.model.predict(text, k=1)  # type: ignore
            primary_lang = labels[0].replace("__label__", "")
            primary_conf = float(probs[0])
            self.last_lang, self.last_conf = primary_lang, primary_conf
            return primary_lang in self.allowed and primary_conf >= self.confidence

        # Fallback to langdetect
        if detect_langs is not None:
            try:
                detections = detect_langs(text)
                if not detections:
                    self.last_lang, self.last_conf = None, None
                    return False
                primary = detections[0]
                self.last_lang, self.last_conf = primary.lang, float(primary.prob)
                return primary.lang in self.allowed and float(primary.prob) >= self.confidence
            except Exception:
                self.last_lang, self.last_conf = None, None
                return False

        # If no detectors installed, allow by default
        self.last_lang, self.last_conf = None, None
        return True

