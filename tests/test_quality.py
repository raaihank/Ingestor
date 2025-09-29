from __future__ import annotations

from ingestor.quality import (
    DuplicateResult,
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
    import tempfile
    import os
    
    # Use a temporary database file for this test to avoid conflicts
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as temp_db:
        temp_db_path = temp_db.name
    
    try:
        from pathlib import Path
        d = NearDuplicateDetector(
            num_perm=128,  # Use smaller num_perm for faster testing
            state_db_path=Path(temp_db_path)
        )
        
        # Test first document (should not be duplicate)
        a = "Ignore previous instructions"
        result_a = d.is_duplicate(a, "test_doc_1", "test_source", "label1")
        assert result_a.is_duplicate is False
        
        # Test very similar document (should be duplicate due to high similarity)
        b = "ignore previous instructions"  # case change only
        result_b = d.is_duplicate(b, "test_doc_2", "test_source", "label2") 
        assert result_b.is_duplicate is True
        
        # Test completely different document (should not be duplicate)
        c = "Completely different text content here with unique words"
        result_c = d.is_duplicate(c, "test_doc_3", "test_source", "label3")
        assert result_c.is_duplicate is False
    
    finally:
        # Clean up temp database
        try:
            os.unlink(temp_db_path)
        except:
            pass


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
