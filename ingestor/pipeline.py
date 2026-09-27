from __future__ import annotations

import os
import queue
import threading
import traceback
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple

from .config import IngestConfig, Override
from .logging_utils import log_dataset, log_debug, log_error, log_success, log_warning
from .normalization import classify_evasion, fold_typography, label_key, normalize_label
from .prepare import Prepared, PrepareParams, init_pool_worker, init_worker, prepare_batch
from .quality import LicenseValidator, NearDuplicateDetector, fasttext_available
from .sources.git import iter_git_repo
from .sources.huggingface import iter_huggingface
from .sources.kaggle import iter_kaggle
from .sources.local import expand_local_globs, iter_local_files
from .state import StateStore, remove_state, state_path_for
from .writers.jsonl_writer import AtomicJSONLWriter

_SENTINEL = object()
_KIND_NAMES = {"hf": "HF dataset", "git": "Git repo", "kaggle": "Kaggle dataset", "local": "local files"}


class Source(NamedTuple):
    kind: str  # hf | git | kaggle | local
    spec: str  # the configured identifier
    open: Callable[[], Iterable[Dict]]


def source_kind(item: Dict) -> str:
    return str(item.get("source", "")).split(":", 1)[0]


def dataset_label(item: Dict) -> str:
    """Per-dataset key used for progress and summaries."""
    meta = item.get("meta") or {}
    return str(meta.get("dataset") or item.get("source") or "unknown")


def _map_label(mapping: Dict[str, str], key: str) -> Optional[str]:
    if key in mapping:
        return mapping[key]
    # Tolerate case/separator differences ("Prompt Injection" vs "prompt_injection")
    wanted = normalize_label(key)
    for raw, mapped in mapping.items():
        if normalize_label(raw) == wanted:
            return mapped
    return None


def _fasttext_model_path(configured: Optional[str]) -> Optional[str]:
    """fasttext_lid_path (existence is checked by the config); fasttext must be importable."""
    if configured and not fasttext_available():
        raise RuntimeError("fasttext_lid_path is set but the fasttext package can't be imported")
    return configured


def _completed(value: Any) -> Future:
    fut: Future = Future()
    fut.set_result(value)
    return fut


@dataclass
class IngestPipeline:
    config: IngestConfig
    commit_every: int = 1000
    approved_count: int = field(default=0, init=False)
    rejected_count: int = field(default=0, init=False)
    existing_count: int = field(default=0, init=False)
    total_records: int = field(default=0, init=False)
    failed_sources: List[Tuple[str, str]] = field(default_factory=list, init=False)
    state_path: Optional[Path] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.license_validator = LicenseValidator()

    # ------------------------------------------------------------------ sources
    def iter_source_specs(self, exclude: Iterable[Path] = ()) -> List[Source]:
        """Configured sources in order; local files listed in `exclude` are skipped."""
        cfg = self.config
        sources: List[Source] = []
        for spec in cfg.hf:
            sources.append(Source("hf", spec, partial(iter_huggingface, spec, cfg)))
        for url in cfg.git:
            sources.append(Source("git", url, partial(iter_git_repo, url)))
        for spec in cfg.kaggle:
            kaggle = partial(
                iter_kaggle, spec, cfg.kaggle_overrides.get(spec), cfg.kaggle_username, cfg.kaggle_key
            )
            sources.append(Source("kaggle", spec, kaggle))
        for pattern, files in expand_local_globs(cfg.local, exclude=list(exclude)):
            if not files:
                log_warning(f"[local] {pattern} matched no data files")
            sources.append(
                Source("local", pattern, partial(iter_local_files, pattern, files, cfg.local_overrides))
            )
        return sources

    def iter_sources(self) -> Iterator[Dict]:
        for source in self.iter_source_specs():
            log_dataset(f"Loading {_KIND_NAMES[source.kind]}: {source.spec}")
            yield from source.open()

    # ------------------------------------------------------- labels / overrides
    def _lookup_keys(self, item: Dict) -> Tuple[str, str, str, Optional[str]]:
        """(kind, configured spec, meta.dataset, file path) of a source item."""
        meta = item.get("meta") or {}
        spec = str(item.get("spec") or "")
        return source_kind(item), spec, str(meta.get("dataset") or spec), meta.get("path")

    def override_for(self, item: Dict) -> Optional[Override]:
        kind, spec, dataset, path = self._lookup_keys(item)
        return self.config.override_for(kind, spec, dataset, path)

    def map_label(self, item: Dict) -> Optional[str]:
        """Dataset label map, then the global map, then lowercase_with_underscores."""
        label = item.get("label")
        if label is None:
            return None
        key = label_key(label)
        kind, spec, dataset, path = self._lookup_keys(item)
        for mapping in (self.config.label_map_for(kind, spec, dataset, path), self.config.global_label_map):
            mapped = _map_label(mapping, key)
            if mapped is not None:
                return normalize_label(mapped)
        return normalize_label(key)

    # -------------------------------------------------------------- per item
    def _rejected(self, item: Dict, reason: str, detail: str = "") -> Dict:
        self.rejected_count += 1
        log_debug(f"[quality] rejected: {reason} {detail}".rstrip())
        return {"event": "rejected", "dataset": dataset_label(item), "reason": reason}

    def _existing(self, item: Dict) -> Dict:
        self.existing_count += 1
        return {"event": "existing", "dataset": dataset_label(item)}

    def _replay(self, item: Dict, decision: str) -> Dict:
        """Outcome of an item decided by an earlier (possibly interrupted) run."""
        if decision == "accepted":
            return self._existing(item)
        return self._rejected(item, decision)

    def _process(self, item: Dict, prep: Prepared, state: StateStore, dups: NearDuplicateDetector) -> Dict:
        doc_id = f"{item.get('source', 'unknown')}:{item.get('source_id', '')}"
        decision = state.decision(doc_id)
        if decision is not None:
            return self._replay(item, decision)
        outcome = self._decide(item, prep, state, dups, doc_id)
        if outcome.get("event") == "rejected":
            state.reject(doc_id, outcome["reason"])
        state.checkpoint()
        return outcome

    def _decide(
        self, item: Dict, prep: Prepared, state: StateStore, dups: NearDuplicateDetector, doc_id: str
    ) -> Dict:
        source = str(item.get("source", "unknown"))
        source_id = str(item.get("source_id", ""))

        if prep.reject:
            return self._rejected(item, prep.reject, prep.detail)

        meta = dict(item.get("meta") or {})
        override = self.override_for(item)
        if override is not None:
            if override.license:
                meta["license"] = override.license
            if override.category and "category" not in meta:
                meta["category"] = override.category
        if self.config.enforce_license and not self.license_validator.validate_source_license(meta):
            return self._rejected(item, "license", str(meta.get("license")))

        label = self.map_label(item)
        label_str = "unknown" if label is None else label
        preserve = self.config.preserve_evasion_variants

        # Exact duplicate: same light text (or same heavy text when variants aren't kept)
        kept = state.find_by_light_hash(prep.light_hash) if preserve else state.find_by_prompt_hash(prep.prompt_hash)
        if kept is not None:
            dups.log_duplicate(kept.id, doc_id, 1.0, 0, "exact_duplicate",
                               kept.label, label_str, kept.source, source)
            return self._rejected(item, "duplicate_exact")

        variant_of: Optional[str] = None
        evasion_type = ""
        # Same heavy text but different light text: an evasion variant, a formatting-only
        # duplicate, or text the heavy view dropped (e.g. different emoji)
        twin = state.find_by_prompt_hash(prep.prompt_hash) if preserve else None
        if twin is not None:
            evasion_type = classify_evasion(twin.text, prep.light)
            if evasion_type:
                variant_of = twin.id
                dups.log_duplicate(twin.id, doc_id, 1.0, 0, "evasion_variant_kept",
                                   twin.label, label_str, twin.source, source, evasion_type)
            elif fold_typography(twin.text) == fold_typography(prep.light):
                dups.log_duplicate(twin.id, doc_id, 1.0, 0, "exact_duplicate",
                                   twin.label, label_str, twin.source, source)
                return self._rejected(item, "duplicate_exact")
        if variant_of is None:
            signature = prep.signature if prep.signature is not None else dups.signature(prep.light)
            result = dups.check(doc_id, prep.light, signature, source, label_str)
            if result.is_duplicate:
                return self._rejected(item, result.reason, f"(similarity={result.similarity:.3f})")
            if result.reason == "evasion_variant":
                variant_of, evasion_type = result.duplicate_of, result.evasion_type
                log_debug(f"[quality] kept evasion variant: {evasion_type} (similarity={result.similarity:.3f})")

        if variant_of and "evasion_variant_of" not in meta:
            meta["evasion_variant_of"] = variant_of
            meta["evasion_type"] = evasion_type

        rec = {
            "id": doc_id,
            "source": source,
            "source_id": source_id,
            "normalized_text": prep.light,
            "prompt_hash": prep.prompt_hash,
            "label": label,
            "meta": meta,
        }
        if self.config.store_raw:
            rec["raw"] = str(item.get("raw", ""))
        if not state.add(rec, prep.light_hash):
            return self._existing(item)
        self.approved_count += 1
        log_success(f"approved: {doc_id}")
        return rec

    # ------------------------------------------------------------- producers
    def _produce(self, source: Source, q: "queue.Queue[Any]", stop: threading.Event) -> None:
        def put(obj: Any) -> bool:
            while not stop.is_set():
                try:
                    q.put(obj, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        try:
            for item in source.open():
                item.setdefault("spec", source.spec)
                if not put(item):
                    return
        except Exception as e:
            self.failed_sources.append((source.spec, str(e)))
            log_error(f"[{source.kind}] {source.spec} failed: {e}")
            log_debug(traceback.format_exc())
        finally:
            put(_SENTINEL)

    # ------------------------------------------------------------------- run
    def run(self, out_path: Path, fresh: bool = False) -> Iterator[Dict]:
        """Ingest all sources, yielding approved records and rejected/existing events.

        Records are stored in a per-output SQLite state; the JSONL is exported from it
        when the run completes, so interrupted runs resume without losing records.
        """
        cfg = self.config
        lid_path = _fasttext_model_path(cfg.fasttext_lid_path)
        self.approved_count = self.rejected_count = self.existing_count = self.total_records = 0
        self.failed_sources = []
        self.state_path = state_path_for(out_path, Path(cfg.state_dir))
        if fresh:
            remove_state(self.state_path)
        state = StateStore(self.state_path, commit_every=self.commit_every)
        dups = NearDuplicateDetector(
            num_perm=cfg.near_dup_num_perm,
            memory_limit=cfg.near_dup_memory_limit,
            threshold=cfg.near_duplicate_threshold,
            preserve_evasion_variants=cfg.preserve_evasion_variants,
            enable_logging=cfg.enable_duplicate_logging,
            conn=state.conn,
        )

        cores = os.cpu_count() or 1
        io_workers = cfg.io_workers if cfg.io_workers is not None else min(32, max(4, cores * 4))
        cpu_workers = cfg.cpu_workers if cfg.cpu_workers is not None else max(1, cores - 1)
        batch_size = cfg.batch_size if cfg.batch_size is not None else 256
        params = PrepareParams(
            min_entropy=cfg.min_entropy,
            min_length=cfg.min_length,
            max_length=cfg.max_length,
            allowed_languages=list(cfg.allowed_languages),
            language_confidence=cfg.language_confidence,
            fasttext_lid_path=lid_path,
            num_perm=cfg.near_dup_num_perm,
        )

        # Never read this run's own output back in (e.g. test-data/*.jsonl)
        sources = self.iter_source_specs(exclude=[out_path])
        # One bounded queue per source: downloads overlap, but sources are consumed
        # in config order so the output doesn't depend on thread timing
        queues: List["queue.Queue[Any]"] = [queue.Queue(maxsize=max(64, batch_size * 2)) for _ in sources]
        stop = threading.Event()
        threads = [
            threading.Thread(target=self._produce, args=(src, q, stop), daemon=True)
            for src, q in zip(sources, queues)
        ]
        started = 0

        pool: Optional[ProcessPoolExecutor] = None
        if cpu_workers > 1:
            pool = ProcessPoolExecutor(
                max_workers=cpu_workers, initializer=init_pool_worker, initargs=(params,)
            )
        else:
            init_worker(params)
        inflight: Deque[Tuple[List[Dict], Future]] = deque()
        max_inflight = max(2, cpu_workers * 2)

        def submit(batch: List[Dict]) -> None:
            texts = [str(it.get("raw", "")) for it in batch]
            fut = pool.submit(prepare_batch, texts) if pool else _completed(prepare_batch(texts))
            inflight.append((batch, fut))

        def drain(keep: int) -> Iterator[Dict]:
            while len(inflight) > keep:
                batch, fut = inflight.popleft()
                for item, prep in zip(batch, fut.result()):
                    yield self._process(item, prep, state, dups)

        completed = False
        try:
            batch: List[Dict] = []
            for i, (source, q) in enumerate(zip(sources, queues)):
                while started < min(len(threads), i + max(1, io_workers)):
                    threads[started].start()
                    started += 1
                log_dataset(f"Loading {_KIND_NAMES[source.kind]}: {source.spec}")
                while True:
                    item = q.get()
                    if item is _SENTINEL:
                        break
                    # Decided by an earlier run (resume / re-run): skip the expensive work
                    decision = state.decision(f"{item.get('source', 'unknown')}:{item.get('source_id', '')}")
                    if decision is not None:
                        yield self._replay(item, decision)
                        continue
                    batch.append(item)
                    if len(batch) >= batch_size:
                        submit(batch)
                        batch = []
                        yield from drain(max_inflight)
            if batch:
                submit(batch)
            yield from drain(0)
            state.commit()
            completed = True
        finally:
            stop.set()
            if pool is not None:
                pool.shutdown(wait=completed, cancel_futures=True)
            if completed:
                for t in threads[:started]:
                    t.join()
            else:
                # Drop the partial batch; everything committed before it stays resumable
                state.rollback()
                state.close()

        try:
            self.total_records = AtomicJSONLWriter(out_path).write_lines(state.iter_json_lines())
        finally:
            state.close()
