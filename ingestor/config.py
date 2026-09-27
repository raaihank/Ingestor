from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union
from typing import Dict as TypingDict

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .normalization import label_key

T = TypeVar("T")


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader for config files.

    Mapping keys are always text, so `1:`, `true:` and `yes:` are label names rather than
    numbers or booleans (which would also collide: True == 1). Only true/false are booleans;
    yes/no/on/off stay words (`no` is the code for Norwegian).
    """

    def construct_mapping(self, node: Any, deep: bool = False) -> Any:
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)  # resolve `<<` merge keys before retagging
            for key_node, _ in node.value:
                if isinstance(key_node, yaml.ScalarNode):
                    key_node.tag = "tag:yaml.org,2002:str"
        return super().construct_mapping(node, deep=deep)


_ConfigLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def _label_map_text(mapping: Dict[Any, Any]) -> Dict[Any, Any]:
    """Label map with number/boolean keys and values in their label text form."""
    def text(value: Any) -> Any:
        return label_key(value) if isinstance(value, (bool, int, float)) else value

    return {text(k): text(v) for k, v in mapping.items()}


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
    language_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    enforce_license: bool = Field(default=False)
    # Quality thresholds
    min_entropy: float = Field(default=2.5, ge=0.0)
    min_length: int = Field(default=10, ge=0)
    max_length: int = Field(default=10000, ge=1)
    # Fixed near-duplicate threshold; None uses the length-aware thresholds
    near_duplicate_threshold: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    # Enhanced deduplication settings
    near_dup_num_perm: int = Field(default=256, ge=16)  # MinHash permutations
    near_dup_memory_limit: int = Field(default=1_000_000, ge=0)  # Max signatures; 0 = no limit
    preserve_evasion_variants: bool = Field(default=True)  # Preserve evasion
    enable_duplicate_logging: bool = Field(default=True)  # Log decisions
    # Parallelism (optional; may be auto-calculated at runtime)
    io_workers: Optional[int] = Field(default=None, ge=1)
    cpu_workers: Optional[int] = Field(default=None, ge=1)
    batch_size: Optional[int] = Field(default=None, ge=1)
    # Language detection model path
    fasttext_lid_path: Optional[str] = Field(default=None)
    # Verbosity level for logging (0,1,2)
    verbose: int = Field(default=0, ge=0, le=2)
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
        # Label maps built in Python may use 1 / True as keys or values
        if isinstance(data.get("global_label_map"), dict):
            data["global_label_map"] = _label_map_text(data["global_label_map"])
        for key in ("hf_label_maps", "kaggle_label_maps", "local_label_maps"):
            maps = data.get(key)
            if isinstance(maps, dict):
                data[key] = {
                    name: _label_map_text(m) if isinstance(m, dict) else m for name, m in maps.items()
                }
        return data

    @field_validator("hf", "git", "kaggle", "local")
    @classmethod
    def _no_blank_sources(cls, sources: List[str]) -> List[str]:
        if any(not s.strip() for s in sources):
            raise ValueError("source entries must not be empty")
        return sources

    @field_validator("fasttext_lid_path")
    @classmethod
    def _model_file_exists(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"file not found: {path}")
        return str(path)

    @field_validator("allowed_languages")
    @classmethod
    def _language_codes(cls, codes: List[str]) -> List[str]:
        codes = [c.strip().lower() for c in codes if c.strip()]
        if not codes:
            raise ValueError('list at least one language code, or ["*"] for all languages')
        return codes

    @model_validator(mode="after")
    def _length_bounds(self) -> "IngestConfig":
        if self.min_length > self.max_length:
            raise ValueError(
                f"min_length ({self.min_length}) must not exceed max_length ({self.max_length})"
            )
        return self

    def hf_override(self, spec: str, split: Optional[str] = None) -> Optional[HfOverride]:
        """HF override; a `name:split` entry refines the `name` entry field by field."""
        base = self.hf_overrides.get(spec)
        specific = self.hf_overrides.get(f"{spec}:{split}") if split else None
        if base is None or specific is None:
            return specific or base
        merged = {**base.model_dump(exclude_none=True), **specific.model_dump(exclude_none=True)}
        return HfOverride(**merged)

    def override_for(
        self, kind: str, spec: str, dataset: Optional[str] = None, path: Optional[str] = None
    ) -> Optional[Override]:
        """Override for a source item.

        `spec` is the configured source entry and `dataset` the item's `meta.dataset`
        (`name:split` for HF). Local overrides may also be keyed by a path glob.
        """
        if kind == "hf":
            return self.hf_override(spec, _split_of(spec, dataset))
        if kind == "kaggle":
            return self.kaggle_overrides.get(spec)
        if kind == "local":
            return lookup(self.local_overrides, [spec], path)
        return None

    def label_map_for(
        self, kind: str, spec: str, dataset: Optional[str] = None, path: Optional[str] = None
    ) -> Dict[str, str]:
        """Dataset-specific label map for a source item (empty if none)."""
        if kind == "hf":
            # A `name:split` map refines the `name` map
            split = _split_of(spec, dataset)
            return {
                **self.hf_label_maps.get(spec, {}),
                **(self.hf_label_maps.get(f"{spec}:{split}", {}) if split else {}),
            }
        if kind == "kaggle":
            return self.kaggle_label_maps.get(spec, {})
        if kind == "local":
            return lookup(self.local_label_maps, [spec], path) or {}
        return {}


def _split_of(spec: str, dataset: Optional[str]) -> Optional[str]:
    """Split name from an HF `meta.dataset` of the form `<spec>:<split>`."""
    if dataset and dataset.startswith(spec + ":"):
        return dataset[len(spec) + 1:]
    return None


def load_config(path: Path) -> IngestConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_ConfigLoader)  # _ConfigLoader is a SafeLoader subclass
    return IngestConfig.model_validate(data or {})
