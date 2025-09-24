from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

import orjson


class AtomicJSONLWriter:
    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self.tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    def write(self, records: Iterable[Dict]) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.tmp_path.open("wb") as f:
            batch: List[bytes] = []
            for rec in records:
                batch.append(orjson.dumps(rec) + b"\n")
            if batch:
                f.writelines(batch)
            else:
                f.write(b"")
        os.replace(self.tmp_path, self.out_path)

