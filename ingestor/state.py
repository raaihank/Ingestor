from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, NamedTuple, Optional

import orjson


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a state database (WAL; NORMAL sync is crash-safe in WAL mode)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-50000")
    return conn


def state_path_for(out_path: Path, state_dir: Path) -> Path:
    """State database for an output file: one per output, so outputs never share state."""
    resolved = Path(out_path).expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return Path(state_dir) / f"{resolved.stem}-{digest}.sqlite"


def remove_state(db_path: Path) -> None:
    """Delete a state database together with its WAL files."""
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)


def serialize_record(record: Dict) -> bytes:
    """JSON-encode a record; values JSON can't represent are stringified."""
    return orjson.dumps(record, default=str, option=orjson.OPT_SERIALIZE_NUMPY)


class StoredRecord(NamedTuple):
    id: str
    source: str
    label: str
    text: str


class StateStore:
    """Every accepted record for one output file.

    The output JSONL is exported from here, so an interrupted run can resume without
    losing records and re-runs rebuild the same file.
    """

    def __init__(self, db_path: Path, commit_every: int = 1000) -> None:
        self.db_path = db_path
        self.commit_every = commit_every
        self.conn = connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                label TEXT NOT NULL,
                light_hash TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                data BLOB NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_records_light ON records(light_hash)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_records_prompt ON records(prompt_hash)")
        # Rejections are remembered too, so a resumed run never re-decides an item
        # against records that were accepted after it
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS rejected (id TEXT PRIMARY KEY, reason TEXT NOT NULL)"
        )
        self.conn.commit()
        self._pending = 0

    def has_id(self, record_id: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM records WHERE id = ?", (record_id,))
        return cur.fetchone() is not None

    def decision(self, record_id: str) -> Optional[str]:
        """"accepted", the stored rejection reason, or None if the item wasn't decided yet."""
        if self.has_id(record_id):
            return "accepted"
        row = self.conn.execute("SELECT reason FROM rejected WHERE id = ?", (record_id,)).fetchone()
        return None if row is None else str(row[0])

    def reject(self, record_id: str, reason: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO rejected (id, reason) VALUES (?, ?)", (record_id, reason)
        )
        self._pending += 1

    def _find(self, column: str, value: str) -> Optional[StoredRecord]:
        row = self.conn.execute(
            f"SELECT id, source, label, data FROM records WHERE {column} = ? ORDER BY seq LIMIT 1",
            (value,),
        ).fetchone()
        if row is None:
            return None
        text = orjson.loads(row[3]).get("normalized_text") or ""
        return StoredRecord(row[0], row[1], row[2], text)

    def find_by_light_hash(self, light_hash: str) -> Optional[StoredRecord]:
        return self._find("light_hash", light_hash)

    def find_by_prompt_hash(self, prompt_hash: str) -> Optional[StoredRecord]:
        return self._find("prompt_hash", prompt_hash)

    def add(self, record: Dict, light_hash: str) -> bool:
        """Insert a record; False if its id is already stored."""
        label = record.get("label")
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO records (id, source, label, light_hash, prompt_hash, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["source"],
                "unknown" if label is None else str(label),
                light_hash,
                record["prompt_hash"],
                serialize_record(record),
            ),
        )
        if cur.rowcount == 0:
            return False
        self._pending += 1
        return True

    def checkpoint(self) -> None:
        """Commit once enough decisions are pending (call between items, never mid-item)."""
        if self._pending >= self.commit_every:
            self.commit()

    def commit(self) -> None:
        self.conn.commit()
        self._pending = 0

    def rollback(self) -> None:
        self.conn.rollback()
        self._pending = 0

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0])

    def iter_json_lines(self) -> Iterator[bytes]:
        """Serialized records in insertion order."""
        for (data,) in self.conn.execute("SELECT data FROM records ORDER BY seq"):
            yield bytes(data)

    def close(self) -> None:
        self.conn.close()
