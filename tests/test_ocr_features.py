from thumbnail_performance.ocr_features import (
    compute_text_features,
    thumbnail_id_from_path,
)


def test_compute_text_features_basic():
    features = compute_text_features("BIG Sale 50 OFF")
    assert features["word_count"] == 4
    assert features["has_numeric"] == 1
    assert features["char_count"] == len("BIGSale50OFF")
    assert features["capital_letter_pct"] == 0.9


def test_compute_text_features_empty():
    features = compute_text_features("")
    assert features["word_count"] == 0
    assert features["capital_letter_pct"] == 0.0
    assert features["has_numeric"] == 0
    assert features["char_count"] == 0


def test_normalize_thumbnail_id_from_path():
    value = r"images\AbC-123.JPG"
    assert thumbnail_id_from_path(value) == "abc-123"
