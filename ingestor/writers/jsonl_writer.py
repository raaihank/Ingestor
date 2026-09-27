from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

from ..state import serialize_record


class AtomicJSONLWriter:
    """Stream JSONL into a temp file beside the target, then atomically replace the target."""

    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self.tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")

    def write(self, records: Iterable[Dict]) -> int:
        return self.write_lines(serialize_record(rec) for rec in records)

    def write_lines(self, lines: Iterable[bytes]) -> int:
        """Write pre-serialized JSON lines (without trailing newlines); returns the count."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        try:
            with self.tmp_path.open("wb") as f:
                chunk: List[bytes] = []
                for line in lines:
                    chunk.append(line + b"\n")
                    count += 1
                    if len(chunk) >= 1000:
                        f.writelines(chunk)
                        chunk.clear()
                f.writelines(chunk)
                f.flush()
                os.fsync(f.fileno())
            os.replace(self.tmp_path, self.out_path)
        except BaseException:
            self.tmp_path.unlink(missing_ok=True)
            raise
        return count
