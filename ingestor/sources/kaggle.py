from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ..config import KaggleOverride, glob_match
from ..constants import STRUCTURED_EXTENSIONS, TEXT_EXTENSIONS
from ..logging_utils import log_warning
from .files import file_items, is_crawlable_data_file


def _unzip_all(root: Path) -> None:
    for zf in root.glob("*.zip"):
        try:
            with zipfile.ZipFile(zf) as z:
                z.extractall(root)
        except zipfile.BadZipFile:
            continue


def _run_kaggle(args: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["kaggle", *args], capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise RuntimeError("kaggle CLI not found; install the `kaggle` package") from e


def _get_kaggle_license(dataset_spec: str, workdir: Path) -> Optional[str]:
    """License name from the dataset's metadata (`kaggle datasets metadata`)."""
    res = _run_kaggle(["datasets", "metadata", dataset_spec, "-p", str(workdir)])
    meta_file = workdir / "dataset-metadata.json"
    if res.returncode != 0 or not meta_file.exists():
        log_warning(f"[kaggle] license lookup failed for {dataset_spec}: {(res.stderr or res.stdout).strip()[:200]}")
        return None
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
    except ValueError:
        return None
    for lic in data.get("licenses") or []:
        if isinstance(lic, dict) and lic.get("name"):
            return str(lic["name"])
    return None


def iter_kaggle_dir(
    root: Path,
    dataset_spec: str,
    license_id: Optional[str],
    override: Optional[KaggleOverride] = None,
) -> Iterator[Dict]:
    """Items from a downloaded (and unzipped) Kaggle dataset directory."""
    include = override.include_globs if override else []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if include:
            wanted = any(glob_match(rel.as_posix(), g) for g in include)
            if not wanted or path.suffix.lower() not in STRUCTURED_EXTENSIONS | TEXT_EXTENSIONS:
                continue
        elif not is_crawlable_data_file(rel):
            continue
        meta = {"path": str(rel), "license": license_id or "UNKNOWN", "dataset": dataset_spec}
        try:
            yield from file_items(
                path,
                source=f"kaggle:{dataset_spec}",
                id_prefix=str(rel),
                meta=meta,
                spec=dataset_spec,
                text_column=override.text_column if override else None,
                label_column=override.label_column if override else None,
            )
        except Exception as e:
            log_warning(f"[kaggle] skipped rest of {rel} in {dataset_spec}: {e}")


def iter_kaggle(dataset_spec: str, override: Optional[KaggleOverride] = None) -> Iterator[Dict]:
    # Requires KAGGLE_USERNAME / KAGGLE_KEY env vars (or ~/.kaggle/kaggle.json) and the kaggle CLI
    tmpdir = Path(tempfile.mkdtemp(prefix="ingest_kaggle_"))
    try:
        data_dir = tmpdir / "data"
        data_dir.mkdir()
        res = _run_kaggle(["datasets", "download", dataset_spec, "-p", str(data_dir), "-q"])
        if res.returncode != 0:
            raise RuntimeError(
                f"kaggle download failed for {dataset_spec}: {(res.stderr or res.stdout).strip()[:300]}"
            )
        _unzip_all(data_dir)
        license_id = override.license if override and override.license else None
        if license_id is None:
            license_id = _get_kaggle_license(dataset_spec, tmpdir)
        yield from iter_kaggle_dir(data_dir, dataset_spec, license_id, override)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
