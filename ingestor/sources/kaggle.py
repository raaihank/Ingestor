from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Generator, Optional

import pandas as pd  # type: ignore
import pyarrow.ipc as pa_ipc  # type: ignore

from ..schema import extract_text_and_label

TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
STRUCTURED_EXTENSIONS = {".jsonl", ".ndjson", ".json", ".csv", ".parquet"}


def _unzip_all(root: Path) -> None:
    for zf in root.glob("*.zip"):
        try:
            with zipfile.ZipFile(zf) as z:
                z.extractall(root)
        except zipfile.BadZipFile:
            continue


def _get_kaggle_license(dataset_spec: str) -> Optional[str]:
    try:
        res = subprocess.run(
            ["kaggle", "datasets", "view", "-d", dataset_spec, "-v"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = res.stdout or ""
        for line in out.splitlines():
            if line.strip().startswith("License(s):"):
                return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def _yield_from_text_file(path: Path, dataset_spec: str, license_id: Optional[str], root: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    rel = path.relative_to(root)
    yield {
        "source": f"kaggle:{dataset_spec}",
        "source_id": str(rel),
        "raw": text,
        "label": None,
        "meta": {"path": str(rel), "license": license_id or "UNKNOWN", "dataset": dataset_spec},
    }


def _yield_from_jsonl(path: Path, dataset_spec: str, license_id: Optional[str], root: Path):
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = {"text": line}
                text = (
                    obj.get("text")
                    or obj.get("prompt")
                    or obj.get("content")
                    or json.dumps(obj, ensure_ascii=False)
                )
                label = obj.get("label")
                yield {
                    "source": f"kaggle:{dataset_spec}",
                    "source_id": f"{path.relative_to(root)}:{idx}",
                    "raw": str(text),
                    "label": label,
                    "meta": {"path": str(path.relative_to(root)), "license": license_id or "UNKNOWN", "dataset": dataset_spec},
                }
    except Exception:
        return


def _yield_from_json(path: Path, dataset_spec: str, license_id: Optional[str], root: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return
    if isinstance(data, list):
        for idx, obj in enumerate(data):
            if isinstance(obj, dict):
                text, label = extract_text_and_label(obj)
                if text is None:
                    text = json.dumps(obj, ensure_ascii=False)
            else:
                text = str(obj)
                label = None
            yield {
                "source": f"kaggle:{dataset_spec}",
                "source_id": f"{path.relative_to(root)}:{idx}",
                "raw": str(text),
                "label": label,
                "meta": {"path": str(path.relative_to(root)), "license": license_id or "UNKNOWN", "dataset": dataset_spec},
            }
    elif isinstance(data, dict):
        text, label = extract_text_and_label(data)
        if text is None:
            text = json.dumps(data, ensure_ascii=False)
        yield {
            "source": f"kaggle:{dataset_spec}",
            "source_id": str(path.relative_to(root)),
            "raw": str(text),
            "label": label,
            "meta": {"path": str(path.relative_to(root)), "license": license_id or "UNKNOWN", "dataset": dataset_spec},
        }


def _yield_from_csv(path: Path, dataset_spec: str, license_id: Optional[str], root: Path):
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            is_dict = True
    except Exception:
        is_dict = False

    if is_dict:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    text, label = extract_text_and_label(row)
                    if text is None:
                        text = " ".join(str(v) for v in row.values())
                    yield {
                        "source": f"kaggle:{dataset_spec}",
                        "source_id": f"{path.relative_to(root)}:{idx}",
                        "raw": str(text),
                        "label": label,
                        "meta": {"path": str(path.relative_to(root)), "license": license_id or "UNKNOWN", "dataset": dataset_spec},
                    }
        except Exception:
            return
    else:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                reader2 = csv.reader(f)
                for idx, row_vals in enumerate(reader2):
                    text = " ".join(str(v) for v in row_vals)
                    yield {
                        "source": f"kaggle:{dataset_spec}",
                        "source_id": f"{path.relative_to(root)}:{idx}",
                        "raw": str(text),
                        "label": None,
                        "meta": {"path": str(path.relative_to(root)), "license": license_id or "UNKNOWN", "dataset": dataset_spec},
                    }
        except Exception:
            return


def _yield_from_parquet(path: Path, dataset_spec: str, license_id: Optional[str], root: Path):
    try:
        df = pd.read_parquet(path)
    except Exception:
        return
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        text = row_dict.get("text") or row_dict.get("prompt") or row_dict.get("content")
        if not text:
            text = " ".join(str(v) for v in row_dict.values())
        yield {
            "source": f"kaggle:{dataset_spec}",
            "source_id": f"{path.relative_to(root)}:{idx}",
            "raw": str(text),
            "label": row_dict.get("label"),
            "meta": {"path": str(path.relative_to(root)), "license": license_id or "UNKNOWN", "dataset": dataset_spec},
        }


def _yield_from_arrow(path: Path, dataset_spec: str, license_id: Optional[str], root: Path):
    try:
        with path.open("rb") as f:
            try:
                reader = pa_ipc.open_file(f)
            except Exception:
                f.seek(0)
                reader = pa_ipc.open_stream(f)
            table = reader.read_all()
        df = table.to_pandas()
    except Exception:
        return
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        text = row_dict.get("text") or row_dict.get("prompt") or row_dict.get("content")
        if not text:
            text = " ".join(str(v) for v in row_dict.values())
        yield {
            "source": f"kaggle:{dataset_spec}",
            "source_id": f"{path.relative_to(root)}:{idx}",
            "raw": str(text),
            "label": row_dict.get("label"),
            "meta": {"path": str(path.relative_to(root)), "license": license_id or "UNKNOWN", "dataset": dataset_spec},
        }


def iter_kaggle(dataset_spec: str) -> Generator[Dict, None, None]:
    # Requires KAGGLE_USERNAME / KAGGLE_KEY env vars and kaggle CLI installed
    tmpdir = Path(tempfile.mkdtemp(prefix="ingest_kaggle_"))
    try:
        # Download dataset (zip)
        res = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_spec, "-p", str(tmpdir), "-q"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return

        # Unzip all archives
        _unzip_all(tmpdir)

        license_id = _get_kaggle_license(dataset_spec)

        # Iterate files
        for path in tmpdir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in TEXT_EXTENSIONS and path.stat().st_size < 5_000_000:
                yield from _yield_from_text_file(path, dataset_spec, license_id, tmpdir)
            elif path.suffix.lower() in {".jsonl", ".ndjson"}:
                yield from _yield_from_jsonl(path, dataset_spec, license_id, tmpdir)
            elif path.suffix.lower() == ".json":
                yield from _yield_from_json(path, dataset_spec, license_id, tmpdir)
            elif path.suffix.lower() == ".csv":
                yield from _yield_from_csv(path, dataset_spec, license_id, tmpdir)
            elif path.suffix.lower() == ".tsv":
                # Reuse CSV handler with tab-delimited rows
                try:
                    with path.open("r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f, delimiter="\t")
                        for idx, row in enumerate(reader):
                            text = row.get("text") or row.get("prompt") or row.get("content")
                            if not text:
                                text = " ".join(str(v) for v in row.values())
                            yield {
                                "source": f"kaggle:{dataset_spec}",
                                "source_id": f"{path.relative_to(tmpdir)}:{idx}",
                                "raw": str(text),
                                "label": row.get("label"),
                                "meta": {"path": str(path.relative_to(tmpdir)), "license": license_id or "UNKNOWN", "dataset": dataset_spec},
                            }
                except Exception:
                    pass
            elif path.suffix.lower() == ".parquet":
                yield from _yield_from_parquet(path, dataset_spec, license_id, tmpdir)
            elif path.suffix.lower() == ".arrow":
                yield from _yield_from_arrow(path, dataset_spec, license_id, tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

