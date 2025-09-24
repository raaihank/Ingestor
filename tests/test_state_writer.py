from __future__ import annotations

from pathlib import Path

from ingestor.state import StateStore
from ingestor.writers.jsonl_writer import AtomicJSONLWriter


def test_state_store_idempotency(tmp_path: Path):
    db = tmp_path / "ingest.sqlite"
    s = StateStore(db_path=db)
    assert s.has_seen("s", "1", "h") is False
    s.mark_seen("s", "1", "h")
    s.flush()
    assert s.has_seen("s", "1", "h") is True


def test_atomic_writer(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    w = AtomicJSONLWriter(out)
    records = [{"a": 1}, {"b": 2}]
    w.write(records)
    content = out.read_text(encoding="utf-8").strip().splitlines()
    assert content[0] == "{\"a\":1}"
    assert content[1] == "{\"b\":2}"

    # Empty write should still create a file
    out2 = tmp_path / "empty.jsonl"
    AtomicJSONLWriter(out2).write([])
    assert out2.exists()
    assert out2.read_text(encoding="utf-8") == ""


