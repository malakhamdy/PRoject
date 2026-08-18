"""OCI Normalization Module - Arabic text and digit normalization."""

from app.normalization.arabic_normalizer import (
    normalize_arabic_digits,
    normalize_unicode,
    remove_diacritics,
    normalize_whitespace,
    normalize_arabic_text,
    normalize_numeric_candidate,
    normalize_gender_text,
    is_mostly_arabic,
    is_mostly_numeric,
)

__all__ = [
    "normalize_arabic_digits",
    "normalize_unicode",
    "remove_diacritics",
    "normalize_whitespace",
    "normalize_arabic_text",
    "normalize_numeric_candidate",
    "normalize_gender_text",
    "is_mostly_arabic",
    "is_mostly_numeric",
]
