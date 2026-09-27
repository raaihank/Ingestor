from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterator

from git import Repo

from ..logging_utils import log_warning
from .files import detect_license_file, file_items, is_crawlable_data_file


def iter_git_repo(repo_url: str) -> Iterator[Dict]:
    """Items from the structured data files of a (shallow-cloned) git repository."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ingest_git_"))
    try:
        Repo.clone_from(repo_url, tmpdir, depth=1)
        license_id = detect_license_file(tmpdir)
        for path in sorted(p for p in tmpdir.rglob("*") if p.is_file()):
            rel = path.relative_to(tmpdir)
            if not is_crawlable_data_file(rel):
                continue
            meta = {"path": str(rel), "dataset": repo_url}
            if license_id:
                meta["license"] = license_id
            try:
                yield from file_items(
                    path, source=f"git:{repo_url}", id_prefix=str(rel), meta=meta, spec=repo_url
                )
            except Exception as e:
                log_warning(f"[git] skipped rest of {rel} in {repo_url}: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
