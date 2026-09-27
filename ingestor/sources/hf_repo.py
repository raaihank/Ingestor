from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from huggingface_hub import HfApi, hf_hub_download

from ..logging_utils import log_warning
from ..schema import infer_split_from_path
from .files import file_items, is_crawlable_data_file


def iter_hf_repo(
    dataset_id: str,
    spec: Optional[str] = None,
    license_id: Any = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> Iterator[Dict]:
    """Items from the data files of a HF dataset repo (fallback when `datasets` can't load it)."""
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    repo_files = HfApi().list_repo_files(dataset_id, repo_type="dataset", revision=revision, token=token)
    files = sorted(f for f in repo_files if is_crawlable_data_file(Path(f)))
    if not files:
        raise RuntimeError(f"no data files found in https://huggingface.co/datasets/{dataset_id}")

    for rel_path in files:
        try:
            local_path = hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                filename=rel_path,
                token=token,
                revision=revision,
            )
        except Exception as e:
            raise RuntimeError(
                f"cannot download {rel_path} from https://huggingface.co/datasets/{dataset_id} "
                f"(gated? accept the terms, then re-run): {e}"
            ) from e

        split = infer_split_from_path(rel_path)
        meta: Dict[str, Any] = {"path": rel_path, "dataset": dataset_id}
        if split:
            meta["split"] = split
        if license_id:
            meta["license"] = license_id
        try:
            yield from file_items(
                Path(local_path),
                source=f"hf:{dataset_id}",
                id_prefix=rel_path,
                meta=meta,
                spec=spec or dataset_id,
                text_column=text_column,
                label_column=label_column,
            )
        except Exception as e:
            log_warning(f"[hf] skipped rest of {rel_path} in {dataset_id}: {e}")
