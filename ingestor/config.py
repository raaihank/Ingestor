from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
from typing import Dict as TypingDict

import yaml
from pydantic import BaseModel, Field, model_validator


class HfOverride(BaseModel):
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    category: Optional[str] = None
    split: Optional[str] = None


class KaggleOverride(BaseModel):
    include_globs: List[str] = Field(default_factory=list)
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    category: Optional[str] = None


class LocalOverride(BaseModel):
    include_globs: List[str] = Field(default_factory=list)
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    category: Optional[str] = None


class IngestConfig(BaseModel):
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
    near_duplicate_threshold: float = Field(default=0.85)
    # Parallelism (optional; may be auto-calculated at runtime)
    io_workers: Optional[int] = Field(default=None)
    cpu_workers: Optional[int] = Field(default=None)
    batch_size: Optional[int] = Field(default=None)
    # Language detection model path
    fasttext_lid_path: Optional[str] = Field(default=None)
    # Verbosity level for logging (0,1,2)
    verbose: int = Field(default=0)
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
        # Coerce single strings to lists for hf/kaggle
        for key in ("hf", "kaggle"):
            if key in data:
                v = data[key]
                if v is None:
                    data[key] = []
                elif isinstance(v, str):
                    data[key] = [v]
        # Ensure git is a list
        if "git" in data and isinstance(data["git"], str):
            data["git"] = [data["git"]]
        # Ensure local is a list
        if "local" in data:
            v = data["local"]
            if v is None:
                data["local"] = []
            elif isinstance(v, str):
                data["local"] = [v]
        return data

    @classmethod
    def from_cli(
        cls,
        *,
        hf: List[str],
        git: List[str],
        kaggle: List[str],
        local: List[str],
        store_raw: bool,
        allowed_languages: Optional[List[str]] = None,
        language_confidence: float = 0.7,
        enforce_license: bool = False,
        min_entropy: float = 2.5,
        min_length: int = 10,
        max_length: int = 10000,
        near_duplicate_threshold: float = 0.85,
        fasttext_lid_path: Optional[str] = None,
        verbose: int = 0,
        hf_token: Optional[str] = None,
        kaggle_username: Optional[str] = None,
        kaggle_key: Optional[str] = None,
        io_workers: Optional[int] = None,
        cpu_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> "IngestConfig":
        return cls(
            hf=hf,
            git=git,
            kaggle=kaggle,
            local=local,
            store_raw=store_raw,
            allowed_languages=allowed_languages or ["en"],
            language_confidence=language_confidence,
            enforce_license=enforce_license,
            min_entropy=min_entropy,
            min_length=min_length,
            max_length=max_length,
            near_duplicate_threshold=near_duplicate_threshold,
            fasttext_lid_path=fasttext_lid_path,
            verbose=verbose,
            hf_token=hf_token,
            kaggle_username=kaggle_username,
            kaggle_key=kaggle_key,
            io_workers=io_workers,
            cpu_workers=cpu_workers,
            batch_size=batch_size,
        )


def load_config(path: Path) -> IngestConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return IngestConfig.model_validate(data)
