from __future__ import annotations

from ingestor.normalization import normalize_text
from ingestor.schema import extract_text_and_label, infer_split_from_path


def test_normalize_text_basic():
    raw = "“Hello”—world\n\t  test\u00A0"
    normalized = normalize_text(raw)
    # Quotes/hyphen mapped, unicode to ascii, whitespace collapsed/trimmed
    assert normalized == '"Hello"-world test'


def test_extract_text_and_label_prefers_columns():
    row = {"prompt": "Do X", "label": 1}
    text, label = extract_text_and_label(row)
    assert text == "Do X"
    assert label == 1


def test_extract_text_and_label_fallback_join():
    row = {"a": 1, "b": "two", "c": [3]}
    text, label = extract_text_and_label(row)
    assert text == "1 two"
    assert label is None


def test_infer_split_from_path():
    assert infer_split_from_path("/data/train/file.jsonl") == "train"
    assert infer_split_from_path("some/VAL/text.csv") == "validation"
    assert infer_split_from_path("a/b/test.txt") == "test"
    assert infer_split_from_path("eval/results.json") == "eval"
    assert infer_split_from_path("misc/file.json") is None
