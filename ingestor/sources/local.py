from __future__ import annotations

import glob as pyglob
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..config import LocalOverride, glob_match, lookup
from ..constants import STRUCTURED_EXTENSIONS, TEXT_EXTENSIONS
from ..logging_utils import log_warning
from ..schema import infer_split_from_path
from .files import file_items

DATA_EXTS = STRUCTURED_EXTENSIONS | TEXT_EXTENSIONS


def expand_local_globs(
    globs: Sequence[str], exclude: Sequence[Path] = ()
) -> List[Tuple[str, List[Path]]]:
    """Sorted data files per glob; a file matched by several globs belongs to the first.

    Files in `exclude` (e.g. the run's own output) are never read.
    """
    seen = {Path(p).resolve() for p in exclude}
    expanded = []
    for pattern in globs:
        files = []
        for p in sorted(pyglob.glob(pattern, recursive=True)):
            path = Path(p)
            key = path.resolve()
            if path.is_file() and path.suffix.lower() in DATA_EXTS and key not in seen:
                seen.add(key)
                files.append(path)
        expanded.append((pattern, files))
    return expanded


def iter_local_files(
    pattern: str,
    files: Sequence[Path],
    overrides: Optional[Dict[str, LocalOverride]] = None,
) -> Iterator[Dict]:
    """Items from the files matched by one configured glob."""
    for path in files:
        override = lookup(overrides or {}, [pattern], str(path))
        if override and override.include_globs and not any(
            glob_match(str(path), g) for g in override.include_globs
        ):
            continue
        split = infer_split_from_path(str(path))
        meta = {"path": str(path), "dataset": pattern, **({"split": split} if split else {})}
        try:
            yield from file_items(
                path,
                source="local",
                id_prefix=str(path),
                meta=meta,
                spec=pattern,
                text_column=override.text_column if override else None,
                label_column=override.label_column if override else None,
            )
        except Exception as e:
            log_warning(f"[local] skipped rest of {path}: {e}")


def iter_local(
    globs: Sequence[str], overrides: Optional[Dict[str, LocalOverride]] = None
) -> Iterator[Dict]:
    for pattern, files in expand_local_globs(globs):
        yield from iter_local_files(pattern, files, overrides)
