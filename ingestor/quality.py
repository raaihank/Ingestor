from __future__ import annotations

import math
import os
import re
import sqlite3
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
from datasketch import LeanMinHash, MinHash, MinHashLSH

from .normalization import (
    classify_evasion,
    get_shingle_size,
    get_similarity_threshold,
    normalize_text_light,
    shingles,
)
from .state import connect

try:
    import fasttext  # type: ignore

    # load_model prints a deprecation notice on every load
    fasttext.FastText.eprint = lambda *args, **kwargs: None  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fasttext = None  # type: ignore

try:
    from langdetect import DetectorFactory, detect_langs  # type: ignore

    # langdetect is randomized unless seeded; seed it for reproducible runs
    DetectorFactory.seed = 0
except Exception:  # pragma: no cover
    detect_langs = None  # type: ignore

MINHASH_SEED = 1


class DuplicateResult(NamedTuple):
    """Result from duplicate detection."""
    is_duplicate: bool
    similarity: float
    duplicate_of: Optional[str] = None
    reason: str = ""
    evasion_type: str = ""


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


def new_minhash_template(num_perm: int) -> MinHash:
    """Empty MinHash whose permutations are shared by every signature built from it."""
    return MinHash(num_perm=num_perm, seed=MINHASH_SEED)


def minhash_signature(text_light: str, template: MinHash) -> np.ndarray:
    """MinHash hash values of a light-normalized text, using length-aware shingles."""
    minhash = template.copy()
    grams = shingles(text_light, get_shingle_size(len(text_light)))
    minhash.update_batch([g.encode("utf-8") for g in grams])
    return np.asarray(minhash.hashvalues)


class _Kept(NamedTuple):
    text: str
    label: str
    source: str


@dataclass
class EnhancedNearDuplicateDetector:
    """Enhanced near-duplicate detector with evasion awareness.

    Signatures, their texts and the duplicate log are persisted in SQLite. Pass `conn`
    to share a connection (the owner commits); otherwise the detector opens
    `state_db_path` and commits in batches (call `commit()` when done).
    """

    # Configuration
    num_perm: int = 256  # Increased from 128 for better accuracy
    state_db_path: Optional[Path] = None
    memory_limit: int = 1_000_000  # Max signatures kept in the in-memory index
    threshold: Optional[float] = None  # Fixed threshold; None = length-aware
    preserve_evasion_variants: bool = True
    enable_logging: bool = True
    conn: Optional[sqlite3.Connection] = None
    commit_every: int = 1000
    signatures: "OrderedDict[str, np.ndarray]" = field(default_factory=OrderedDict, init=False)

    def __post_init__(self) -> None:
        self._owns_conn = self.conn is None
        if self.conn is None:
            if self.state_db_path is None:
                self.state_db_path = Path(".state/near_dup_sigs.sqlite")
            self.conn = connect(self.state_db_path)
        self._db: sqlite3.Connection = self.conn
        self._pending = 0

        self._template = new_minhash_template(self.num_perm)
        self.scheme = self._template.scheme
        self._dtype = self._template.hashvalues.dtype
        # Candidate search runs below the decision threshold; candidates are re-checked
        lsh_threshold = 0.8 if self.threshold is None else max(0.3, min(0.8, self.threshold - 0.1))
        self.lsh = MinHashLSH(threshold=lsh_threshold, num_perm=self.num_perm)

        self._init_db()
        self._load_existing_signatures()

    def _init_db(self) -> None:
        """Create the signature and duplicate-log tables."""
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS minhash_sig (
                doc_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                label TEXT NOT NULL,
                len_bucket INTEGER NOT NULL,
                text_length INTEGER NOT NULL,
                num_perm INTEGER NOT NULL,
                sig BLOB NOT NULL,
                scheme TEXT,
                text TEXT
            )
        """)
        # Databases written by older versions lack these columns
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(minhash_sig)")}
        for column in ("scheme", "text"):
            if column not in columns:
                self._db.execute(f"ALTER TABLE minhash_sig ADD COLUMN {column} TEXT")

        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_minhash_source
            ON minhash_sig(source)
        """)

        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_minhash_label
            ON minhash_sig(label, len_bucket)
        """)

        self._db.execute("""
            CREATE TABLE IF NOT EXISTS duplicate_log (
                kept_id TEXT NOT NULL,
                dropped_id TEXT NOT NULL,
                jaccard REAL NOT NULL,
                k INTEGER NOT NULL,
                reason TEXT NOT NULL,
                label_kept TEXT NOT NULL,
                label_dropped TEXT NOT NULL,
                source_kept TEXT NOT NULL,
                source_dropped TEXT NOT NULL,
                evasion_type TEXT,
                PRIMARY KEY (kept_id, dropped_id)
            )
        """)
        if self._owns_conn:
            self._db.commit()

    def _load_existing_signatures(self) -> None:
        """Index the most recent persisted signatures (up to memory_limit)."""
        limit = self.memory_limit if self.memory_limit > 0 else -1
        rows = self._db.execute(
            """
            SELECT doc_id, sig FROM minhash_sig
            WHERE num_perm = ? AND scheme = ?
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (self.num_perm, self.scheme, limit),
        ).fetchall()
        for doc_id, blob in reversed(rows):
            hashvalues = np.frombuffer(blob, dtype=self._dtype)
            if len(hashvalues) == self.num_perm:
                self._index(doc_id, hashvalues)

    def _lean(self, hashvalues: np.ndarray) -> LeanMinHash:
        return LeanMinHash(seed=MINHASH_SEED, hashvalues=hashvalues, scheme=self.scheme)

    def _index(self, doc_id: str, hashvalues: np.ndarray) -> None:
        """Add a signature to the in-memory index, evicting the oldest past memory_limit."""
        if doc_id in self.signatures:
            return
        if self.memory_limit > 0 and len(self.signatures) >= self.memory_limit:
            oldest, _ = self.signatures.popitem(last=False)
            self.lsh.remove(oldest)
        self.signatures[doc_id] = hashvalues
        self.lsh.insert(doc_id, self._lean(hashvalues), check_duplication=False)

    def signature(self, text_light: str) -> np.ndarray:
        return minhash_signature(text_light, self._template)

    def is_duplicate(self, text: str, doc_id: str, source: str,
                     label: str) -> DuplicateResult:
        """
        Enhanced duplicate detection with evasion awareness.

        Args:
            text: Raw text to check
            doc_id: Unique document identifier
            source: Source of the document
            label: Document label

        Returns:
            DuplicateResult with detailed information
        """
        text_light = normalize_text_light(text)
        return self.check(doc_id, text_light, self.signature(text_light), source, label)

    def check(self, doc_id: str, text_light: str, hashvalues: np.ndarray,
              source: str, label: str) -> DuplicateResult:
        """Compare a precomputed signature against the index; index it if it is unique."""
        text_length = len(text_light)
        threshold = self.threshold if self.threshold is not None else get_similarity_threshold(text_length)
        hv = np.asarray(hashvalues, dtype=self._dtype)

        # Most similar indexed document above the threshold (ties -> smallest id)
        best_id: Optional[str] = None
        best_sim = -1.0
        for candidate in self.lsh.query(self._lean(hv)):
            other = self.signatures.get(candidate)
            if candidate == doc_id or other is None:
                continue
            sim = float(np.count_nonzero(other == hv)) / self.num_perm
            if sim < threshold:
                continue
            if best_id is None or sim > best_sim or (sim == best_sim and candidate < best_id):
                best_id, best_sim = candidate, sim

        if best_id is None:
            self.add(doc_id, text_light, hv, source, label)
            return DuplicateResult(is_duplicate=False, similarity=0.0)

        kept = self._kept(best_id)
        k = get_shingle_size(text_length)
        evasion_type = classify_evasion(kept.text, text_light) if self.preserve_evasion_variants else ""
        if evasion_type:
            # Don't collapse evasion variants - keep both
            self.log_duplicate(best_id, doc_id, best_sim, k, "evasion_variant_kept",
                               kept.label, label, kept.source, source, evasion_type)
            return DuplicateResult(
                is_duplicate=False,
                similarity=best_sim,
                duplicate_of=best_id,
                reason="evasion_variant",
                evasion_type=evasion_type,
            )

        self.log_duplicate(best_id, doc_id, best_sim, k, "near_duplicate",
                           kept.label, label, kept.source, source)
        return DuplicateResult(
            is_duplicate=True,
            similarity=best_sim,
            duplicate_of=best_id,
            reason="near_duplicate",
        )

    def _kept(self, doc_id: str) -> _Kept:
        row = self._db.execute(
            "SELECT text, label, source FROM minhash_sig WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return _Kept("", "unknown", "unknown")
        return _Kept(row[0] or "", row[1], row[2])

    def add(self, doc_id: str, text_light: str, hashvalues: np.ndarray,
            source: str, label: str) -> None:
        """Persist and index a signature."""
        text_length = len(text_light)
        len_bucket = 0 if text_length < 40 else 1 if text_length <= 200 else 2
        self._db.execute(
            """
            INSERT OR REPLACE INTO minhash_sig
            (doc_id, source, label, len_bucket, text_length, num_perm, sig, scheme, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, source, label, len_bucket, text_length, self.num_perm,
             np.ascontiguousarray(hashvalues, dtype=self._dtype).tobytes(), self.scheme, text_light),
        )
        self._index(doc_id, np.asarray(hashvalues, dtype=self._dtype))
        self._wrote()

    def log_duplicate(self, kept_id: str, dropped_id: str, jaccard: float,
                      k: int, reason: str, label_kept: str, label_dropped: str,
                      source_kept: str, source_dropped: str,
                      evasion_type: str = "") -> None:
        """Log duplicate detection for auditing."""
        if not self.enable_logging:
            return
        self._db.execute("""
            INSERT OR REPLACE INTO duplicate_log
            (kept_id, dropped_id, jaccard, k, reason, label_kept, label_dropped,
             source_kept, source_dropped, evasion_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (kept_id, dropped_id, jaccard, k, reason, label_kept, label_dropped,
              source_kept, source_dropped, evasion_type))
        self._wrote()

    def _wrote(self) -> None:
        if not self._owns_conn:
            return
        self._pending += 1
        if self._pending >= self.commit_every:
            self.commit()

    def commit(self) -> None:
        self._db.commit()
        self._pending = 0

    def close(self) -> None:
        if self._owns_conn:
            self.commit()
            self._db.close()

    def get_duplicate_stats(self) -> Dict:
        """Get statistics about duplicates found."""
        cursor = self._db.execute("""
            SELECT reason, COUNT(*) as count, AVG(jaccard) as avg_similarity
            FROM duplicate_log
            GROUP BY reason
        """)

        stats = {}
        for reason, count, avg_sim in cursor:
            stats[reason] = {'count': count, 'avg_similarity': avg_sim}

        return stats


# Alias for backward compatibility - use enhanced version
NearDuplicateDetector = EnhancedNearDuplicateDetector

APPROVED_LICENSES = frozenset({
    "mit",
    "apache-2.0",
    "bsd-3-clause",
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "unlicense",
})

_LICENSE_ALIASES = {
    "mit license": "mit",
    "the mit license": "mit",
    "apache2": "apache-2.0",
    "apache-2": "apache-2.0",
    "apache license 2.0": "apache-2.0",
    "apache license, version 2.0": "apache-2.0",
    "cc0": "cc0-1.0",
    "cc0: public domain": "cc0-1.0",
    "public domain (cc0)": "cc0-1.0",
    "the unlicense": "unlicense",
    "attribution 4.0 international (cc by 4.0)": "cc-by-4.0",
    "attribution-sharealike 4.0 international (cc by-sa 4.0)": "cc-by-sa-4.0",
}


def canonical_license(value: Any) -> str:
    """Lowercase SPDX-style id for a license string ("Apache 2.0" -> "apache-2.0")."""
    name = re.sub(r"[\s_]+", " ", str(value).strip().lower())
    name = _LICENSE_ALIASES.get(name, name)
    return name.replace(" ", "-")


class LicenseValidator:
    def __init__(self) -> None:
        self.approved_licenses = set(APPROVED_LICENSES)

    def validate_source_license(self, source_metadata: Dict) -> bool:
        """True if the source license (or any of several) is approved; unknown is rejected."""
        value = source_metadata.get("license")
        values: List[Any] = list(value) if isinstance(value, (list, tuple)) else [value]
        return any(v and canonical_license(v) in self.approved_licenses for v in values)


class LanguageFilter:
    def __init__(
        self,
        allowed_languages: Optional[List[str]] = None,
        confidence: float = 0.7,
        model_path: Optional[Path] = None,
    ) -> None:
        # Support "*" as special marker for all languages
        if allowed_languages and "*" in allowed_languages:
            self.allowed = set(["*"])  # Special marker for all languages
        else:
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
                except Exception as e:
                    # Log the failure but continue with fallback
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Failed to load FastText model from {lid_path}: {e}"
                    )
                    self.model = None

    def is_allowed_language(self, text: str) -> bool:
        if not text:
            self.last_lang, self.last_conf = None, None
            return False
        # Prefer FastText if available
        if self.model is not None:
            try:
                labels, probs = self.model.predict(text, k=1)  # type: ignore
                primary_lang = labels[0].replace("__label__", "")
                primary_conf = float(probs[0])
                self.last_lang, self.last_conf = primary_lang, primary_conf
                # If "*" is in allowed, accept any language above confidence threshold
                if "*" in self.allowed:
                    return primary_conf >= self.confidence
                return primary_lang in self.allowed and primary_conf >= self.confidence
            except Exception as e:
                # Handle NumPy compatibility issues or other FastText errors
                import logging
                logging.getLogger(__name__).warning(
                    f"FastText prediction failed: {e}. Falling back to langdetect."
                )
                # Disable the model for future calls to avoid repeated errors
                self.model = None

        # Fallback to langdetect
        if detect_langs is not None:
            try:
                detections = detect_langs(text)
                if not detections:
                    self.last_lang, self.last_conf = None, None
                    return False
                primary = detections[0]
                self.last_lang, self.last_conf = primary.lang, float(primary.prob)
                # If "*" is in allowed, accept any language above confidence threshold
                if "*" in self.allowed:
                    return float(primary.prob) >= self.confidence
                return primary.lang in self.allowed and float(primary.prob) >= self.confidence
            except Exception:
                self.last_lang, self.last_conf = None, None
                return False

        # If no detectors installed, allow by default
        self.last_lang, self.last_conf = None, None
        return True
