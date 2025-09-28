from __future__ import annotations

import hashlib
import os
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional

from .config import IngestConfig
from .logging_utils import log_dataset, log_debug, log_success
from .normalization import normalize_text, normalize_label
from .quality import (
    LanguageFilter,
    LicenseValidator,
    NearDuplicateDetector,
    filter_by_entropy,
    validate_length,
)
from .sources.git import iter_git_repo
from .sources.huggingface import iter_huggingface
from .sources.kaggle import iter_kaggle
from .sources.local import iter_local
from .state import StateStore
from .writers.jsonl_writer import AtomicJSONLWriter


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_and_hash(text: str) -> tuple[str, str]:
    from .normalization import normalize_text as _norm

    normalized = _norm(text)
    return normalized, sha256_hex(normalized)


@dataclass
class IngestPipeline:
    config: IngestConfig

    def __post_init__(self) -> None:
        self.state = StateStore(db_path=Path(".state/ingest.sqlite"))
        self.dup_detector = NearDuplicateDetector(threshold=self.config.near_duplicate_threshold)
        self.license_validator = LicenseValidator()
        self.language_filter = LanguageFilter(
            allowed_languages=self.config.allowed_languages,
            confidence=self.config.language_confidence,
        )
        self.approved_count = 0
        self.rejected_count = 0
        # Fast in-run exact dedupe guard to complement SQLite idempotency
        self._seen_triplets: set[tuple[str, str, str]] = set()

    def _normalize_label_and_inject_category(self, label, meta: Dict, dataset_id: str) -> tuple:
        """Extract and normalize label, inject category into meta if needed."""
        # Apply label normalization maps (dataset-specific overrides then global)
        if label is not None:
            try:
                label_str = str(label)
                mapped = None
                if dataset_id and dataset_id in self.config.hf_label_maps:
                    mapped = self.config.hf_label_maps[dataset_id].get(label_str, None)
                if mapped is None and dataset_id and dataset_id in self.config.kaggle_label_maps:
                    mapped = self.config.kaggle_label_maps[dataset_id].get(label_str, None)
                if mapped is None:
                    mapped = self.config.global_label_map.get(label_str, None)
                if mapped is not None:
                    label = mapped
                
                # Apply final label normalization (lowercase, underscores)
                label = normalize_label(str(label))
            except Exception:
                pass

        # Inject category from overrides
        if dataset_id and "category" not in meta:
            cat = None
            if dataset_id in self.config.hf_overrides and \
               self.config.hf_overrides[dataset_id].category:
                cat = self.config.hf_overrides[dataset_id].category
            if dataset_id in self.config.kaggle_overrides and \
               self.config.kaggle_overrides[dataset_id].category:
                cat = self.config.kaggle_overrides[dataset_id].category
            if cat:
                meta["category"] = cat
        
        return label, meta

    def _build_record_from_normalized(self, item: Dict, normalized: str, prompt_hash: str) -> Optional[Dict]:
        source = str(item.get("source", "unknown"))
        source_id = str(item.get("source_id", ""))
        label = item.get("label")
        meta = item.get("meta", {})
        dataset_id = str(meta.get("dataset") or "")

        label, meta = self._normalize_label_and_inject_category(label, meta, dataset_id)

        if not self._passes_quality(normalized, meta):
            return None

        key = (source, source_id, prompt_hash)
        if key in self._seen_triplets:
            log_debug("[quality] rejected: duplicate (in-run)")
            return None
        # Cross-run idempotency
        if self.state.has_seen(source, source_id, prompt_hash):
            log_debug("[quality] rejected: duplicate (exact)")
            return None
        self._seen_triplets.add(key)
        self.state.mark_seen(source, source_id, prompt_hash)

        rec = {
            "id": f"{source}:{source_id}",
            "source": source,
            "source_id": source_id,
            "normalized_text": normalized,
            "prompt_hash": prompt_hash,
            "label": label,
            "meta": meta,
        }
        # Retain raw optionally
        if self.config.store_raw:
            rec["raw"] = str(item.get("raw", ""))
        return rec

    def _iter_sources(self) -> Iterable[Dict]:
        for hf_ds in self.config.hf:
            log_dataset(f"Loading HF dataset: {hf_ds}")
            yield from iter_huggingface(hf_ds, self.config)
        for repo in self.config.git:
            log_dataset(f"Loading Git repo: {repo}")
            yield from iter_git_repo(repo)
        for kg in self.config.kaggle:
            log_dataset(f"Loading Kaggle dataset: {kg}")
            yield from iter_kaggle(kg)
        if self.config.local:
            log_dataset("Loading local datasets")
            yield from iter_local(self.config.local)

    def _passes_quality(self, text: str, meta: Dict) -> bool:
        if not filter_by_entropy(text, self.config.min_entropy):
            log_debug("[quality] rejected: entropy")
            return False
        if not validate_length(text, self.config.min_length, self.config.max_length):
            log_debug("[quality] rejected: length")
            return False
        if not self.language_filter.is_allowed_language(text):
            log_debug(
                f"[quality] rejected: language lang={self.language_filter.last_lang} conf={self.language_filter.last_conf}"
            )
            return False
        if self.dup_detector.is_duplicate(text)[0]:
            log_debug("[quality] rejected: near-duplicate")
            return False
        if self.config.enforce_license and not self.license_validator.validate_source_license(meta):
            log_debug(f"[quality] rejected: license {meta.get('license')}")
            return False
        return True

    def _build_record(self, item: Dict) -> Optional[Dict]:
        raw: str = str(item.get("raw", ""))
        normalized = normalize_text(raw)
        prompt_hash = sha256_hex(normalized)
        source = str(item.get("source", "unknown"))
        source_id = str(item.get("source_id", ""))
        label = item.get("label")
        meta = item.get("meta", {})
        dataset_id = str(meta.get("dataset") or "")

        label, meta = self._normalize_label_and_inject_category(label, meta, dataset_id)

        if not self._passes_quality(normalized, meta):
            return None

        if self.state.has_seen(source, source_id, prompt_hash):
            log_debug("[quality] rejected: duplicate (exact)")
            return None
        self.state.mark_seen(source, source_id, prompt_hash)

        rec = {
            "id": f"{source}:{source_id}",
            "source": source,
            "source_id": source_id,
            "normalized_text": normalized,
            "prompt_hash": prompt_hash,
            "label": label,
            "meta": meta,
        }
        if self.config.store_raw:
            rec["raw"] = raw
        return rec

    def run(self, out_path: Path) -> Generator[Dict, None, None]:
        writer = AtomicJSONLWriter(out_path)
        buffer: list[Dict] = []
        # Heuristics / defaults
        cores = os.cpu_count() or 1
        io_workers = (
            self.config.io_workers
            if self.config.io_workers is not None
            else min(32, max(4, cores * 4))
        )
        cpu_workers = (
            self.config.cpu_workers
            if self.config.cpu_workers is not None
            else max(1, cores - 1)
        )
        batch_size = self.config.batch_size if self.config.batch_size is not None else 256

        # Bounded queue for backpressure
        q: queue.Queue = queue.Queue(maxsize=max(64, batch_size * 2))
        SENTINEL = object()

        def _produce_hf(spec: str) -> None:
            try:
                for it in iter_huggingface(spec, self.config):
                    q.put(it)
            finally:
                q.put(SENTINEL)

        def _produce_git(url: str) -> None:
            try:
                for it in iter_git_repo(url):
                    q.put(it)
            finally:
                q.put(SENTINEL)

        def _produce_kaggle(spec: str) -> None:
            try:
                for it in iter_kaggle(spec):
                    q.put(it)
            finally:
                q.put(SENTINEL)

        def _produce_local() -> None:
            try:
                if self.config.local:
                    for it in iter_local(self.config.local):
                        q.put(it)
            finally:
                q.put(SENTINEL)

        # Submit producers
        producers = 0
        try:
            with ThreadPoolExecutor(max_workers=io_workers) as io_pool, \
                 ProcessPoolExecutor(max_workers=cpu_workers) as cpu_pool:
                for ds in self.config.hf:
                    io_pool.submit(_produce_hf, ds)
                    producers += 1
                for url in self.config.git:
                    io_pool.submit(_produce_git, url)
                    producers += 1
                for kg in self.config.kaggle:
                    io_pool.submit(_produce_kaggle, kg)
                    producers += 1
                # Local as a single producer
                if self.config.local:
                    io_pool.submit(_produce_local)
                    producers += 1

                finished = 0
                batch_items: list[Dict] = []
                while finished < producers:
                    item = q.get()
                    if item is SENTINEL:
                        finished += 1
                        continue
                    batch_items.append(item)
                    if len(batch_items) >= batch_size:
                        # Dispatch batch to CPU pool for normalization/hash
                        texts = [str(it.get("raw", "")) for it in batch_items]
                        results = list(
                            cpu_pool.map(_normalize_and_hash, texts)
                        )
                        for it, (normalized, p_hash) in zip(batch_items, results):
                            dataset_id_yield = str(
                                it.get("meta", {}).get("dataset")
                                or it.get("source", "unknown")
                            )
                            rec = self._build_record_from_normalized(it, normalized, p_hash)
                            if rec is None:
                                self.rejected_count += 1
                                yield {
                                    "event": "rejected",
                                    "dataset": dataset_id_yield,
                                }
                                continue
                            self.approved_count += 1
                            log_success(f"approved: {rec['id']}")
                            buffer.append(rec)
                            yield rec
                        batch_items.clear()

                # Process any remaining items
                if batch_items:
                    texts = [str(it.get("raw", "")) for it in batch_items]
                    results = list(cpu_pool.map(_normalize_and_hash, texts))
                    for it, (normalized, p_hash) in zip(batch_items, results):
                        dataset_id_yield = str(
                            it.get("meta", {}).get("dataset")
                            or it.get("source", "unknown")
                        )
                        rec = self._build_record_from_normalized(it, normalized, p_hash)
                        if rec is None:
                            self.rejected_count += 1
                            yield {
                                "event": "rejected",
                                "dataset": dataset_id_yield,
                            }
                            continue
                        self.approved_count += 1
                        log_success(f"approved: {rec['id']}")
                        buffer.append(rec)
                        yield rec
        finally:
            # Ensure batched DB writes are persisted
            try:
                self.state.flush()
            except Exception:
                pass

        if buffer:
            writer.write(buffer)

