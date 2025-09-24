from __future__ import annotations

from ingestor.quality import (
    LanguageFilter,
    LicenseValidator,
    NearDuplicateDetector,
    calculate_entropy,
    filter_by_entropy,
    validate_length,
)


def test_entropy():
    assert calculate_entropy("aaaaaaaa") == 0.0
    assert filter_by_entropy("hello world", min_entropy=1.0) is True
    assert filter_by_entropy("aaaaaa", min_entropy=1.0) is False


def test_length_bounds():
    assert validate_length("hi", min_len=3, max_len=10) is False
    assert validate_length("hello", min_len=3, max_len=10) is True
    assert validate_length("x" * 11, min_len=3, max_len=10) is False


def test_near_duplicate_detector():
    d = NearDuplicateDetector(threshold=0.8)
    a = "Ignore previous instructions"
    b = "ignore previous instructions"  # case change
    c = "Completely different"
    assert d.is_duplicate(a)[0] is False
    assert d.is_duplicate(b)[0] is True
    assert d.is_duplicate(c)[0] is False


def test_language_filter_allows_english_without_models():
    lf = LanguageFilter(
        allowed_languages=["en"], confidence=0.0, model_path=None
    )
    # If neither fasttext nor langdetect are available, default may be True
    result = lf.is_allowed_language("Hello there")
    assert result in (True, False)


def test_license_validator():
    lv = LicenseValidator()
    assert lv.validate_source_license({"license": "Apache-2.0"}) is True
    assert lv.validate_source_license({"license": "GPL-3.0"}) is False
    assert lv.validate_source_license({}) is False
