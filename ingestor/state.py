from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class StateStore:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        # Speed-oriented PRAGMAs for local cache usage
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=OFF")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute("PRAGMA cache_size=-20000")
        except Exception:
            pass
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen (
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                PRIMARY KEY (source, source_id, prompt_hash)
            )
            """
        )
        self._conn.commit()
        # Batch buffer for inserts
        self._batch_buffer: List[Tuple[str, str, str]] = []
        self._batch_size: int = 1000

    def has_seen(self, source: str, source_id: str, prompt_hash: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM seen WHERE source=? AND source_id=? AND prompt_hash=?",
            (source, source_id, prompt_hash),
        )
        return cur.fetchone() is not None

    def mark_seen(self, source: str, source_id: str, prompt_hash: str) -> None:
        self._batch_buffer.append((source, source_id, prompt_hash))
        if len(self._batch_buffer) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._batch_buffer:
            return
        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN")
            cur.executemany(
                "INSERT OR IGNORE INTO seen (source, source_id, prompt_hash) VALUES (?, ?, ?)",
                self._batch_buffer,
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        finally:
            self._batch_buffer.clear()

