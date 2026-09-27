from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional

from datasets import load_dataset
from huggingface_hub import HfApi

from ..logging_utils import log_debug, log_warning
from .hf_repo import iter_hf_repo

if TYPE_CHECKING:
    from ..config import HfOverride, IngestConfig


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


def _card_license(path: str, token: str | None, revision: str | None) -> Any:
    """License declared in the dataset card (a string or a list), if the Hub reports one."""
    try:
        info = HfApi().dataset_info(path, revision=revision, token=token)
    except Exception as e:
        log_debug(f"[hf] could not read dataset card license for {path}: {e}")
        return None
    card = getattr(info, "card_data", None)
    return getattr(card, "license", None) if card is not None else None


def _info_license(ds: Any) -> str | None:
    """License from a loaded split's builder info (often empty for Hub datasets)."""
    lic = getattr(getattr(ds, "info", None), "license", None)
    return str(lic) if lic else None


def _iter_rows(
    ds,
    dataset_name: str,
    license_id: Any,
    text_col: str | None = None,
    label_col: str | None = None,
    spec: str | None = None,
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
        # Track dataset id for overrides; a row's own "dataset" column must not replace it
        if "dataset" in meta:
            meta["row_dataset"] = meta.pop("dataset")
        meta["dataset"] = dataset_name
        yield {
            "source": f"hf:{dataset_name}",
            "source_id": str(idx),
            "raw": "" if text is None else str(text),
            "label": label,
            "meta": meta,
            "spec": spec or dataset_name,
        }


def iter_huggingface(
    dataset_name: str, config: "IngestConfig | None" = None
) -> Generator[Dict, None, None]:
    # The config's hf_token wins over the HF_TOKEN/HUGGINGFACEHUB_API_TOKEN env vars
    token = (
        (config.hf_token if config else None)
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    path, name, revision = _parse_hf_spec(dataset_name)

    def override(split: Optional[str] = None) -> Optional[HfOverride]:
        # `name:split` entries refine the `name` entry
        return config.hf_override(dataset_name, split) if config else None

    base = override()
    split_override = base.split if base else None
    card_license = (base.license if base else None) or _card_license(path, token, revision)

    def rows(ds: Any, split: str) -> Generator[Dict, None, None]:
        ov = override(split)
        license_id = (ov.license if ov else None) or card_license or _info_license(ds)
        yield from _iter_rows(
            ds,
            f"{dataset_name}:{split}",
            license_id,
            ov.text_column if ov else None,
            ov.label_column if ov else None,
            dataset_name,
        )

    # By default use ALL splits; an explicit split uses only that split
    if split_override:
        try:
            ds = load_dataset(
                path, name=name, split=split_override, token=token, revision=revision
            )
        except Exception as e:
            raise RuntimeError(f"could not load split {split_override!r} of {dataset_name}: {e}") from e
        yield from rows(ds, split_override)
        return

    try:
        ds_dict = load_dataset(path, name=name, token=token, revision=revision)
    except Exception as e:
        log_warning(f"[hf] load_dataset failed for {dataset_name} ({e}); crawling repository files instead")
        yield from iter_hf_repo(
            path,
            spec=dataset_name,
            license_id=card_license,
            text_column=base.text_column if base else None,
            label_column=base.label_column if base else None,
            token=token,
            revision=revision,
        )
        return

    for split_name in ds_dict.keys():
        yield from rows(ds_dict[split_name], split_name)
