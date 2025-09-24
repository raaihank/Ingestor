#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover
    orjson = None  # type: ignore


@dataclass
class LengthStats:
    count: int = 0
    total: int = 0
    min_len: int = 10 ** 12
    max_len: int = 0

    def add(self, n: int) -> None:
        self.count += 1
        self.total += n
        if n < self.min_len:
            self.min_len = n
        if n > self.max_len:
            self.max_len = n

    @property
    def mean(self) -> float:
        return (self.total / self.count) if self.count else 0.0


def load_line(line: str) -> Dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)  # type: ignore
    return json.loads(line)


def extract_text(obj: Dict[str, Any], override_field: Optional[str]) -> str:
    if override_field:
        v = obj.get(override_field)
        if v is not None:
            return str(v)
    return (
        str(obj.get("normalized_text")
            or obj.get("text")
            or obj.get("raw")
            or "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute quick insights for a JSONL dataset")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--top", type=int, default=10, help="Top-N items to display per category")
    parser.add_argument("--text-field", default=None, help="Override text field (default: auto)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument("--hash-stats", action="store_true", help="Compute prompt_hash duplicate stats")
    parser.add_argument("--max-hashes", type=int, default=500000, help="Max hashes to track (memory bound)")
    args = parser.parse_args()

    path = args.input
    if not os.path.isfile(path):
        print(f"Input file does not exist: {path}")
        sys.exit(1)

    total = 0
    text_field_used = args.text_field or "auto"
    length_stats = LengthStats()
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    label_field_used: Optional[str] = None
    label_fields_seen: set[str] = set()
    labeled_count = 0
    unlabeled_count = 0

    unique_hashes: Optional[set[str]] = set() if args.hash_stats else None
    dup_count = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = load_line(line)
            except Exception:
                continue
            total += 1

            # text field
            text = extract_text(obj, args.text_field)
            if args.text_field and text_field_used == "auto":
                text_field_used = args.text_field
            length_stats.add(len(text))

            # label detection (single field heuristic)
            label_value: Optional[Any] = None
            candidates = (
                ["label", "labels", "target", "class", "category", "is_malicious", "malicious", "y"]
            )
            if "label" in obj and obj["label"] is not None:
                label_value = obj["label"]
                label_fields_seen.add("label")
                if label_field_used is None:
                    label_field_used = "label"
            else:
                for k in candidates:
                    if k in obj and obj[k] is not None:
                        label_value = obj[k]
                        label_fields_seen.add(k)
                        if label_field_used is None:
                            label_field_used = k
                        break
            if label_value is not None:
                labeled_count += 1
                label_counts[str(label_value)] += 1
            else:
                unlabeled_count += 1

            # source
            if "source" in obj and obj["source"]:
                source_counts[str(obj["source"]) ] += 1

            # meta
            meta = obj.get("meta") or {}
            if isinstance(meta, dict):
                ds = meta.get("dataset")
                if ds:
                    dataset_counts[str(ds)] += 1
                cat = meta.get("category")
                if cat:
                    category_counts[str(cat)] += 1
                lic = meta.get("license")
                if lic:
                    license_counts[str(lic)] += 1

            # prompt_hash duplicate tracking (optional)
            if unique_hashes is not None:
                h = obj.get("prompt_hash")
                if isinstance(h, str):
                    if len(unique_hashes) < args.max_hashes:
                        if h in unique_hashes:
                            dup_count += 1
                        else:
                            unique_hashes.add(h)
                    else:
                        # capacity reached; skip further tracking
                        unique_hashes = None

    def top_n(counter: Counter[str]) -> list[tuple[str, int]]:
        return counter.most_common(args.top)

    report = {
        "file": path,
        "total": total,
        "text_field_used": text_field_used,
        "label_field_used": label_field_used or "<none>",
        "length": {
            "mean": round(length_stats.mean, 2),
            "min": (0 if length_stats.count == 0 else length_stats.min_len),
            "max": length_stats.max_len,
        },
        "labels_top": top_n(label_counts),
        "labels": {
            "total_labeled": labeled_count,
            "total_unlabeled": unlabeled_count,
            "distribution": [
                {
                    "value": k,
                    "count": v,
                    "pct_total": round((v / total) * 100.0, 2) if total else 0.0,
                    "pct_labeled": round((v / labeled_count) * 100.0, 2) if labeled_count else 0.0,
                }
                for k, v in sorted(label_counts.items(), key=lambda x: -x[1])
            ],
        },
        "sources_top": top_n(source_counts),
        "datasets_top": top_n(dataset_counts),
        "categories_top": top_n(category_counts),
        "licenses_top": top_n(license_counts),
    }
    if unique_hashes is not None:
        unique_count = len(unique_hashes)
        dup_ratio = (dup_count / total) if total else 0.0
        report["prompt_hash"] = {
            "unique": unique_count,
            "duplicates": dup_count,
            "dup_ratio": round(dup_ratio, 4),
            "tracking_capped": False,
        }
    else:
        report["prompt_hash"] = {"tracking_capped": True}

    if args.as_json:
        print(json.dumps(report, ensure_ascii=False))
        return

    # Human-readable output
    print(f"File: {report['file']}")
    print(f"Total records: {report['total']}")
    print(f"Text field used: {report['text_field_used']}")
    print(f"Text length: mean={report['length']['mean']} min={report['length']['min']} max={report['length']['max']}")
    print(f"Label field used: {report['label_field_used']}")
    # Full label breakdown
    labels_info = report["labels"]
    print(
        f"Labels: total_labeled={labels_info['total_labeled']} "
        f"({round((labels_info['total_labeled']/total)*100.0,2) if total else 0.0}%)  "
        f"total_unlabeled={labels_info['total_unlabeled']}"
    )
    if report["labels_top"]:
        print("\nTop labels:")
        for k, v in report["labels_top"]:
            pct = (v / total * 100.0) if total else 0.0
            print(f"  {k}: {v} ({pct:.2f}%)")
    if report["sources_top"]:
        print("\nTop sources:")
        for k, v in report["sources_top"]:
            print(f"  {k}: {v}")
    if report["datasets_top"]:
        print("\nTop datasets:")
        for k, v in report["datasets_top"]:
            print(f"  {k}: {v}")
    if report["categories_top"]:
        print("\nTop categories:")
        for k, v in report["categories_top"]:
            print(f"  {k}: {v}")
    if report["licenses_top"]:
        print("\nTop licenses:")
        for k, v in report["licenses_top"]:
            print(f"  {k}: {v}")
    ph = report.get("prompt_hash", {})
    if ph:
        print("\nPrompt hash:")
        if ph.get("tracking_capped"):
            print("  tracking_capped: true (set --hash-stats to compute, may use memory)")
        else:
            print(f"  unique: {ph.get('unique')} duplicates: {ph.get('duplicates')} dup_ratio: {ph.get('dup_ratio')}")


if __name__ == "__main__":
    main()


