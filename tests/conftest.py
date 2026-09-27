from __future__ import annotations

import os

# CI sets FORCE_COLOR, which makes rich add ANSI codes to captured output. Drop it before
# ingestor creates its console so tests (and the CLI subprocesses they start) see plain text.
os.environ.pop("FORCE_COLOR", None)

from pathlib import Path  # noqa: E402
from typing import Callable, Dict, List, Tuple  # noqa: E402

import pytest  # noqa: E402

from ingestor.config import IngestConfig  # noqa: E402
from ingestor.pipeline import IngestPipeline  # noqa: E402


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., IngestConfig]:
    """Permissive config (no quality filtering, inline workers) with state under tmp_path."""
    def _make(**overrides) -> IngestConfig:
        base = dict(
            allowed_languages=["*"],
            language_confidence=0.0,
            min_entropy=0.0,
            min_length=0,
            max_length=100_000,
            cpu_workers=1,
            batch_size=8,
            state_dir=str(tmp_path / ".state"),
        )
        base.update(overrides)
        return IngestConfig(**base)
    return _make


@pytest.fixture
def ingest() -> Callable[..., Tuple[IngestPipeline, List[Dict]]]:
    """Run a pipeline to completion; returns the pipeline and every yielded outcome."""
    def _run(cfg: IngestConfig, out: Path, **kwargs) -> Tuple[IngestPipeline, List[Dict]]:
        pipeline = IngestPipeline(config=cfg)
        return pipeline, list(pipeline.run(out_path=out, **kwargs))
    return _run
