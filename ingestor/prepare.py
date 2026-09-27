"""Stateless per-text work, run in worker processes: normalize, hash, filter, sign."""
from __future__ import annotations

import hashlib
import signal
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import numpy as np

from .normalization import normalize_text_heavy, normalize_text_light
from .quality import (
    LanguageFilter,
    filter_by_entropy,
    minhash_signature,
    new_minhash_template,
    validate_length,
)


class PrepareParams(NamedTuple):
    min_entropy: float
    min_length: int
    max_length: int
    allowed_languages: List[str]
    language_confidence: float
    fasttext_lid_path: Optional[str]
    num_perm: int


class Prepared(NamedTuple):
    light: str  # light-normalized text (quality checks, near-dup, output)
    light_hash: str
    prompt_hash: str  # hash of the heavy-normalized text
    reject: Optional[str]  # failed quality check, None if the text passed
    detail: str  # e.g. detected language, for debug logs
    signature: Optional[np.ndarray]  # MinHash values; None when rejected


_WORKER: Dict[str, Any] = {}


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def init_worker(params: PrepareParams) -> None:
    """Set up the language model and MinHash template for this process."""
    model_path = Path(params.fasttext_lid_path) if params.fasttext_lid_path else None
    _WORKER["params"] = params
    _WORKER["language"] = LanguageFilter(
        allowed_languages=params.allowed_languages,
        confidence=params.language_confidence,
        model_path=model_path,
    )
    _WORKER["template"] = new_minhash_template(params.num_perm)


def init_pool_worker(params: PrepareParams) -> None:
    # Ctrl-C is handled by the parent, which shuts the pool down
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    init_worker(params)


def prepare_text(raw: str) -> Prepared:
    params: PrepareParams = _WORKER["params"]
    light = normalize_text_light(raw)
    heavy = normalize_text_heavy(raw)
    light_hash, prompt_hash = sha256_hex(light), sha256_hex(heavy)

    reject: Optional[str] = None
    detail = ""
    if not light:
        reject = "empty"
    elif not filter_by_entropy(light, params.min_entropy):
        reject = "entropy"
    elif not validate_length(light, params.min_length, params.max_length):
        reject = "length"
    else:
        language: LanguageFilter = _WORKER["language"]
        if not language.is_allowed_language(light):
            reject = "language"
        detail = f"lang={language.last_lang} conf={language.last_conf}"

    signature = None if reject else minhash_signature(light, _WORKER["template"])
    return Prepared(light, light_hash, prompt_hash, reject, detail, signature)


def prepare_batch(texts: Sequence[str]) -> List[Prepared]:
    return [prepare_text(text) for text in texts]
