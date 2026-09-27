from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestor.state import StateStore, state_path_for
from ingestor.writers.jsonl_writer import AtomicJSONLWriter


def _record(rec_id: str, text: str = "hello world") -> dict:
    return {
        "id": rec_id,
        "source": "s",
        "source_id": rec_id.split(":", 1)[1],
        "normalized_text": text,
        "prompt_hash": f"ph-{text}",
        "label": "benign",
        "meta": {},
    }


def test_state_store_records(tmp_path: Path):
    db = tmp_path / "ingest.sqlite"
    s = StateStore(db_path=db)
    rec = _record("s:1")
    assert s.has_id("s:1") is False
    assert s.add(rec, "lh-1") is True
    assert s.add(rec, "lh-1") is False  # same id is stored once
    assert s.has_id("s:1") is True
    found = s.find_by_light_hash("lh-1")
    assert found is not None and found.id == "s:1" and found.text == "hello world"
    assert s.find_by_prompt_hash("ph-hello world") is not None
    s.commit()
    s.close()

    reopened = StateStore(db_path=db)
    assert [json.loads(line) for line in reopened.iter_json_lines()] == [rec]


def test_state_store_rollback_discards_uncommitted(tmp_path: Path):
    s = StateStore(db_path=tmp_path / "ingest.sqlite")
    s.add(_record("s:1"), "a")
    s.commit()
    s.add(_record("s:2", "second"), "b")
    s.rollback()
    assert s.count() == 1


def test_state_path_is_per_output(tmp_path: Path):
    state_dir = tmp_path / ".state"
    a = state_path_for(tmp_path / "a.jsonl", state_dir)
    assert a == state_path_for(tmp_path / "a.jsonl", state_dir)
    assert a != state_path_for(tmp_path / "b.jsonl", state_dir)
    assert a.parent == state_dir


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


def test_atomic_writer_keeps_target_on_error(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    out.write_text("previous\n", encoding="utf-8")

    def lines():
        yield b'{"a":1}'
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        AtomicJSONLWriter(out).write_lines(lines())
    assert out.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_state_store_remembers_decisions(tmp_path: Path):
    db = tmp_path / "ingest.sqlite"
    s = StateStore(db_path=db)
    s.add(_record("s:1"), "a")
    s.reject("s:2", "near_duplicate")
    assert s.decision("s:3") is None
    s.commit()
    s.close()
    reopened = StateStore(db_path=db)
    assert reopened.decision("s:1") == "accepted"
    assert reopened.decision("s:2") == "near_duplicate"
