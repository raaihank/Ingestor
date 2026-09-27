from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union
from typing import Dict as TypingDict

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


def glob_match(path: str, pattern: str) -> bool:
    """fnmatch-style match (`*` also crosses directories); a leading `**/` may match nothing."""
    path = path.replace(os.sep, "/")
    pattern = pattern.replace(os.sep, "/")
    paths = {path, os.path.abspath(path).replace(os.sep, "/")}
    patterns = {pattern, pattern[3:]} if pattern.startswith("**/") else {pattern}
    return any(fnmatch.fnmatchcase(p, pat) for p in paths for pat in patterns)


def lookup(mapping: Dict[str, T], keys: Sequence[str], path: Optional[str] = None) -> Optional[T]:
    """First entry whose key equals one of `keys`, else (if `path` given) whose key glob-matches it."""
    for key in keys:
        if key and key in mapping:
            return mapping[key]
    if path:
        for pattern, value in mapping.items():
            if glob_match(path, pattern):
                return value
    return None


class _Override(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_column: Optional[str] = None
    label_column: Optional[str] = None
    category: Optional[str] = None
    # Declared license; takes precedence over what the source reports
    license: Optional[str] = None


class HfOverride(_Override):
    split: Optional[str] = None


class KaggleOverride(_Override):
    include_globs: List[str] = Field(default_factory=list)


class LocalOverride(_Override):
    include_globs: List[str] = Field(default_factory=list)


Override = Union[HfOverride, KaggleOverride, LocalOverride]

_MAPPING_KEYS = (
    "hf_overrides",
    "kaggle_overrides",
    "local_overrides",
    "global_label_map",
    "hf_label_maps",
    "kaggle_label_maps",
    "local_label_maps",
)


class IngestConfig(BaseModel):
    # Unknown keys are errors so typos don't silently disable settings
    model_config = ConfigDict(extra="forbid")

    hf: List[str] = Field(default_factory=list)
    git: List[str] = Field(default_factory=list)
    kaggle: List[str] = Field(default_factory=list)
    local: List[str] = Field(default_factory=list)
    store_raw: bool = Field(default=False)
    allowed_languages: List[str] = Field(default_factory=lambda: ["en"])
    language_confidence: float = Field(default=0.7)
    enforce_license: bool = Field(default=False)
    # Quality thresholds
    min_entropy: float = Field(default=2.5)
    min_length: int = Field(default=10)
    max_length: int = Field(default=10000)
    # Fixed near-duplicate threshold; None uses the length-aware thresholds
    near_duplicate_threshold: Optional[float] = Field(default=None)
    # Enhanced deduplication settings
    near_dup_num_perm: int = Field(default=256)  # MinHash permutations
    near_dup_memory_limit: int = Field(default=1_000_000)  # Max signatures
    preserve_evasion_variants: bool = Field(default=True)  # Preserve evasion
    enable_duplicate_logging: bool = Field(default=True)  # Log decisions
    # Parallelism (optional; may be auto-calculated at runtime)
    io_workers: Optional[int] = Field(default=None)
    cpu_workers: Optional[int] = Field(default=None)
    batch_size: Optional[int] = Field(default=None)
    # Language detection model path
    fasttext_lid_path: Optional[str] = Field(default=None)
    # Verbosity level for logging (0,1,2)
    verbose: int = Field(default=0)
    # Directory holding the per-output resumable state
    state_dir: str = Field(default=".state")
    hf_token: Optional[str] = Field(default=None)
    kaggle_username: Optional[str] = Field(default=None)
    kaggle_key: Optional[str] = Field(default=None)
    # Dataset-specific overrides
    hf_overrides: TypingDict[str, HfOverride] = Field(default_factory=dict)
    kaggle_overrides: TypingDict[str, KaggleOverride] = Field(default_factory=dict)
    local_overrides: TypingDict[str, LocalOverride] = Field(default_factory=dict)
    # Label normalization (optional)
    global_label_map: TypingDict[str, str] = Field(default_factory=dict)
    hf_label_maps: TypingDict[str, TypingDict[str, str]] = Field(default_factory=dict)
    kaggle_label_maps: TypingDict[str, TypingDict[str, str]] = Field(default_factory=dict)
    local_label_maps: TypingDict[str, TypingDict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_lists(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        # Coerce single strings to lists and empty values to empty lists
        for key in ("hf", "git", "kaggle", "local"):
            if key in data:
                v = data[key]
                if v is None:
                    data[key] = []
                elif isinstance(v, str):
                    data[key] = [v]
        # An empty YAML entry (`key:`) means "nothing configured"
        for key in _MAPPING_KEYS:
            v = data.get(key)
            if key in data and v is None:
                data[key] = {}
            elif isinstance(v, dict) and key != "global_label_map":
                data[key] = {k: ({} if val is None else val) for k, val in v.items()}
        return data

    def override_for(
        self, kind: str, keys: Sequence[str], path: Optional[str] = None
    ) -> Optional[Override]:
        """Override for a source item; local overrides may also be keyed by a path glob."""
        if kind == "hf":
            return lookup(self.hf_overrides, keys)
        if kind == "kaggle":
            return lookup(self.kaggle_overrides, keys)
        if kind == "local":
            return lookup(self.local_overrides, keys, path)
        return None

    def label_map_for(
        self, kind: str, keys: Sequence[str], path: Optional[str] = None
    ) -> Dict[str, str]:
        """Dataset-specific label map for a source item (empty if none)."""
        if kind == "hf":
            found = lookup(self.hf_label_maps, keys)
        elif kind == "kaggle":
            found = lookup(self.kaggle_label_maps, keys)
        elif kind == "local":
            found = lookup(self.local_label_maps, keys, path)
        else:
            found = None
        return found or {}


def load_config(path: Path) -> IngestConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return IngestConfig.model_validate(data or {})
