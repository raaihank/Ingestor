from __future__ import annotations

import math
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple

from datasketch import MinHash, MinHashLSH

from .normalization import (
    normalize_text_light,
    has_evasion_markers,
    get_shingle_size,
    get_similarity_threshold
)

try:
    import fasttext  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fasttext = None  # type: ignore

try:
    from langdetect import detect_langs  # type: ignore
except Exception:  # pragma: no cover
    detect_langs = None  # type: ignore


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


@dataclass
class EnhancedNearDuplicateDetector:
    """Enhanced near-duplicate detector with evasion awareness and
    label-aware deduplication."""

    # Configuration
    num_perm: int = 256  # Increased from 128 for better accuracy
    state_db_path: Optional[Path] = None
    memory_limit: int = 1_000_000  # Max signatures to keep in memory

    def __post_init__(self) -> None:
        # LSH forest for similarity search
        # Lower threshold, we'll filter later
        self.lsh = MinHashLSH(threshold=0.8, num_perm=self.num_perm)

        # In-memory signature storage
        self.seen_hashes: Dict[str, MinHash] = {}
        self.doc_metadata: Dict[str, Dict] = {}  # Store metadata for each doc

        # Set up persistent state database
        if self.state_db_path is None:
            self.state_db_path = Path(".state/near_dup_sigs.sqlite")

        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_existing_signatures()

    def _init_db(self) -> None:
        """Initialize SQLite database for persistent signatures."""
        self._conn = sqlite3.connect(self.state_db_path)

        # Optimize for our use case
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=OFF")
            self._conn.execute("PRAGMA cache_size=-50000")  # 50MB cache
        except Exception:
            pass

        # Create tables
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS minhash_sig (
                doc_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                label TEXT NOT NULL,
                len_bucket INTEGER NOT NULL,
                text_length INTEGER NOT NULL,
                num_perm INTEGER NOT NULL,
                sig BLOB NOT NULL
            )
        """)
        
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_minhash_source
            ON minhash_sig(source)
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_minhash_label
            ON minhash_sig(label, len_bucket)
        """)

        self._conn.execute("""
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

        self._conn.commit()

    def _load_existing_signatures(self) -> None:
        """Load existing signatures from database into memory (up to limit)."""
        cursor = self._conn.execute("""
            SELECT doc_id, sig, source, label, text_length, len_bucket
            FROM minhash_sig
            ORDER BY rowid
            LIMIT ?
        """, (self.memory_limit,))

        for row in cursor:
            (doc_id, sig_blob, source, label,
             text_length, len_bucket) = row

            # Reconstruct MinHash from stored bytes
            minhash = MinHash(num_perm=self.num_perm)
            # Assuming sig_blob contains the hash values as bytes
            # In real implementation, you'd need to properly serialize/deserialize

            self.seen_hashes[doc_id] = minhash
            self.doc_metadata[doc_id] = {
                'source': source,
                'label': label,
                'text_length': text_length,
                'len_bucket': len_bucket
            }

            # Add to LSH
            self.lsh.insert(doc_id, minhash)

    def create_minhash(self, text: str, k: Optional[int] = None) -> MinHash:
        """Create MinHash with length-aware shingles."""
        text_light = normalize_text_light(text)

        if k is None:
            k = get_shingle_size(len(text_light))

        minhash = MinHash(num_perm=self.num_perm)

        # Create k-gram shingles
        for i in range(max(0, len(text_light) - k + 1)):
            shingle = text_light[i:i + k]
            minhash.update(shingle.encode('utf-8'))

        return minhash

    def is_duplicate(self, text: str, doc_id: str, source: str,
                     label: str) -> DuplicateResult:
        """
        Enhanced duplicate detection with evasion awareness and label checking.

        Args:
            text: Raw text to check
            doc_id: Unique document identifier
            source: Source of the document
            label: Document label

        Returns:
            DuplicateResult with detailed information
        """
        text_light = normalize_text_light(text)
        text_length = len(text_light)
        len_bucket = (
            0 if text_length < 40
            else 1 if text_length <= 200
            else 2
        )

        # Create MinHash
        k = get_shingle_size(text_length)
        minhash = self.create_minhash(text, k)

        # Query LSH for similar documents
        similar_docs = self.lsh.query(minhash)
        threshold = get_similarity_threshold(text_length)
        
        for similar_doc_id in similar_docs:
            if similar_doc_id not in self.seen_hashes:
                continue

            similar_minhash = self.seen_hashes[similar_doc_id]
            similar_meta = self.doc_metadata[similar_doc_id]
            jaccard_sim = minhash.jaccard(similar_minhash)

            # Skip if below threshold
            if jaccard_sim < threshold:
                continue

            # Check for evasion variants
            # Would need to store this too for evasion check
            similar_text = ""
            is_evasion, evasion_type = has_evasion_markers(
                text_light, similar_text
            )
            
            if is_evasion:
                # Don't collapse evasion variants - keep both
                self._log_duplicate(
                    kept_id=similar_doc_id,
                    dropped_id=doc_id, 
                    jaccard=jaccard_sim,
                    k=k,
                    reason="evasion_variant_kept",
                    label_kept=similar_meta['label'],
                    label_dropped=label,
                    source_kept=similar_meta['source'],
                    source_dropped=source,
                    evasion_type=evasion_type
                )
                
                return DuplicateResult(
                    is_duplicate=False,
                    similarity=jaccard_sim,
                    duplicate_of=similar_doc_id,
                    reason="evasion_variant",
                    evasion_type=evasion_type
                )
            
            # High similarity - this is a duplicate
            self._log_duplicate(
                kept_id=similar_doc_id,
                dropped_id=doc_id,
                jaccard=jaccard_sim,
                k=k,
                reason="near_duplicate",
                label_kept=similar_meta['label'],
                label_dropped=label,
                source_kept=similar_meta['source'], 
                source_dropped=source
            )
            
            return DuplicateResult(
                is_duplicate=True,
                similarity=jaccard_sim,
                duplicate_of=similar_doc_id,
                reason="near_duplicate"
            )
        
        # Not a duplicate - add to LSH and storage
        self._add_signature(
            doc_id, minhash, source, label, text_length, len_bucket
        )

        return DuplicateResult(
            is_duplicate=False,
            similarity=0.0
        )

    def _add_signature(self, doc_id: str, minhash: MinHash, source: str,
                      label: str, text_length: int,
                      len_bucket: int) -> None:
        """Add new signature to LSH and persistent storage."""

        # Check memory limit
        if len(self.seen_hashes) >= self.memory_limit:
            # In production, implement LRU eviction or sharding
            pass

        # Add to in-memory structures
        self.seen_hashes[doc_id] = minhash
        self.doc_metadata[doc_id] = {
            'source': source,
            'label': label,
            'text_length': text_length,
            'len_bucket': len_bucket
        }

        # Add to LSH
        self.lsh.insert(doc_id, minhash)

        # Persist to database
        # Note: In real implementation, you'd properly serialize the MinHash
        sig_blob = bytes(128)  # Placeholder - would serialize minhash.hashvalues

        self._conn.execute("""
            INSERT OR REPLACE INTO minhash_sig
            (doc_id, source, label, len_bucket, text_length,
             num_perm, sig)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, source, label, len_bucket, text_length,
              self.num_perm, sig_blob))
        self._conn.commit()

    def _log_duplicate(self, kept_id: str, dropped_id: str, jaccard: float,
                      k: int, reason: str, label_kept: str, label_dropped: str,
                      source_kept: str, source_dropped: str,
                      evasion_type: str = "") -> None:
        """Log duplicate detection for auditing."""
        self._conn.execute("""
            INSERT OR REPLACE INTO duplicate_log
            (kept_id, dropped_id, jaccard, k, reason, label_kept, label_dropped,
             source_kept, source_dropped, evasion_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (kept_id, dropped_id, jaccard, k, reason, label_kept, label_dropped,
              source_kept, source_dropped, evasion_type))
        self._conn.commit()

    def get_duplicate_stats(self) -> Dict:
        """Get statistics about duplicates found."""
        cursor = self._conn.execute("""
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

