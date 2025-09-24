from __future__ import annotations

import json
from pathlib import Path

from ingestor.config import IngestConfig
from ingestor.pipeline import IngestPipeline
from ingestor.sources.local import iter_local


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_iter_local_jsonl(tmp_path: Path):
    f = tmp_path / "dataset_train.jsonl"
    write_jsonl(f, [{"text": "Hello", "label": 1}, {"text": "World", "label": 0}])
    items = list(iter_local([str(tmp_path / "*.jsonl")]))
    assert len(items) == 2
    assert items[0]["raw"] == "Hello"
    assert items[0]["label"] == 1


def test_pipeline_local_only(tmp_path: Path, monkeypatch):
    # Minimal config with local source only and permissive filters
    cfg = IngestConfig(
        hf=[], git=[], kaggle=[], local=[str(tmp_path / "*.jsonl")],
        store_raw=False, allowed_languages=["en"], language_confidence=0.0,
        enforce_license=False, min_entropy=0.0, min_length=0, max_length=100000,
        near_duplicate_threshold=0.90,
    )
    # Create simple local data
    f = tmp_path / "data.jsonl"
    write_jsonl(f, [{"text": "A sample", "label": "1"}, {"text": "Another", "label": "0"}])

    p = IngestPipeline(config=cfg)
    out = tmp_path / "out.jsonl"
    approved = 0
    for rec in p.run(out_path=out):
        if isinstance(rec, dict) and rec.get("event") == "rejected":
            continue
        approved += 1
    # Depending on language detection fallback, at least one should pass
    assert approved >= 1
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1


