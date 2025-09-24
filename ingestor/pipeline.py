from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional

from .config import IngestConfig
from .logging_utils import log_dataset, log_debug, log_success
from .normalization import normalize_text
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

    def _iter_sources(self) -> Iterable[Dict]:
        for hf_ds in self.config.hf:
            log_dataset(f"Loading HF dataset: {hf_ds}")
            yield from iter_huggingface(hf_ds)
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
            except Exception:
                pass

        # Inject category from overrides
        if dataset_id and "category" not in meta:
            cat = None
            if dataset_id in self.config.hf_overrides and self.config.hf_overrides[dataset_id].category:
                cat = self.config.hf_overrides[dataset_id].category
            if dataset_id in self.config.kaggle_overrides and self.config.kaggle_overrides[dataset_id].category:
                cat = self.config.kaggle_overrides[dataset_id].category
            if cat:
                meta["category"] = cat

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
        buffer = []
        try:
            for item in self._iter_sources():
                dataset_id_yield = str(item.get("meta", {}).get("dataset") or item.get("source", "unknown"))
                rec = self._build_record(item)
                if rec is None:
                    self.rejected_count += 1
                    # Yield a lightweight rejection event for UI counters
                    yield {"event": "rejected", "dataset": dataset_id_yield}
                    continue
                self.approved_count += 1
                log_success(f"approved: {rec['id']}")
                buffer.append(rec)
                yield rec
            if buffer:
                writer.write(buffer)
        finally:
            # Ensure batched DB writes are persisted
            try:
                self.state.flush()
            except Exception:
                pass

