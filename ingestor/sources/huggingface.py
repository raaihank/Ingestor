from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Generator

from datasets import load_dataset

if TYPE_CHECKING:
    from ..config import IngestConfig


def _parse_hf_spec(spec: str) -> tuple[str, str | None, str | None]:
    # Supports formats like: "owner/name", "owner/name:config",
    # "owner/name@rev", "owner/name:config@rev"
    revision = None
    path = spec
    name = None
    if "@" in path:
        path, revision = path.split("@", 1)
    if ":" in path:
        path, name = path.split(":", 1)
    return path, name, revision


def _iter_rows(
    ds,
    dataset_name: str,
    license_id: str | None,
    text_col: str | None = None,
    label_col: str | None = None,
):
    for idx, row in enumerate(ds):
        # Use override columns if specified, otherwise fallback to defaults
        if text_col:
            text = row.get(text_col, "")
        else:
            text = (
                row.get("text")
                or row.get("prompt")
                or row.get("content")
                or ""
            )

        if label_col:
            label = row.get(label_col)
        else:
            label = row.get("label")

        # Exclude the override columns from meta to avoid duplication
        exclude_keys = {"text", "prompt", "label", "content"}
        if text_col:
            exclude_keys.add(text_col)
        if label_col:
            exclude_keys.add(label_col)

        meta = {k: v for k, v in row.items() if k not in exclude_keys}
        if license_id and "license" not in meta:
            meta["license"] = license_id
        # Track dataset id for overrides
        meta.setdefault("dataset", dataset_name)
        yield {
            "source": f"hf:{dataset_name}",
            "source_id": str(idx),
            "raw": str(text),
            "label": label,
            "meta": meta,
        }


def iter_huggingface(
    dataset_name: str, config: "IngestConfig | None" = None
) -> Generator[Dict, None, None]:
    # datasets respects HF_TOKEN/HUGGINGFACEHUB_API_TOKEN env vars
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    path, name, revision = _parse_hf_spec(dataset_name)

    # Get overrides for this dataset
    text_col = None
    label_col = None
    split_override = None

    if config and dataset_name in config.hf_overrides:
        override = config.hf_overrides[dataset_name]
        text_col = override.text_column
        label_col = override.label_column
        split_override = override.split

    license_id = None
    # Try train split first; fallback to all splits
    try:
        split_to_use = split_override or "train"
        ds = load_dataset(
            path, name=name, split=split_to_use, token=token, revision=revision
        )
        info = getattr(ds, "info", None)
        if info is not None:
            lic = getattr(info, "license", None)
            if lic is not None:
                license_id = str(lic)
        yield from _iter_rows(
            ds, dataset_name, license_id, text_col, label_col
        )
        return
    except Exception:
        pass

    try:
        ds_dict = load_dataset(path, name=name, token=token, revision=revision)
    except Exception:
        return

    # Capture license from builder info when possible
    try:
        info = getattr(ds_dict, "info", None)
        if info is not None:
            lic = getattr(info, "license", None)
            if lic is not None:
                license_id = str(lic)
    except Exception:
        license_id = None

    for split_name in getattr(ds_dict, "keys", lambda: [])():
        split_ds = ds_dict[split_name]
        yield from _iter_rows(
            split_ds,
            f"{dataset_name}:{split_name}",
            license_id,
            text_col,
            label_col,
        )
