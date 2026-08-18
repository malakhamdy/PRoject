"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Arabic Text Normalization Module

Handles Arabic-specific text normalization:
- Arabic-Indic digit normalization (٠١٢٣٤٥٦٧٨٩ → 0123456789)
- Unicode normalization
- Whitespace normalization
- OCR noise removal
- Character variant normalization
"""

import re
import unicodedata
from typing import Optional


# Arabic-Indic digits mapping
ARABIC_INDIC_DIGITS = {
    '\u0660': '0',  # ٠
    '\u0661': '1',  # ١
    '\u0662': '2',  # ٢
    '\u0663': '3',  # ٣
    '\u0664': '4',  # ٤
    '\u0665': '5',  # ٥
    '\u0666': '6',  # ٦
    '\u0667': '7',  # ٧
    '\u0668': '8',  # ٨
    '\u0669': '9',  # ٩
    '\u06F0': '0',  # ۰
    '\u06F1': '1',  # ۱
    '\u06F2': '2',  # ۲
    '\u06F3': '3',  # ۳
    '\u06F4': '4',  # ۴
    '\u06F5': '5',  # ۵
    '\u06F6': '6',  # ۶
    '\u06F7': '7',  # ۷
    '\u06F8': '8',  # ۸
    '\u06F9': '9',  # ۹
}

# Common Arabic character confusions in OCR
ARABIC_OCR_CONFUSIONS = {
    '\u0622': '\u0627',  # آ → ا
    '\u0623': '\u0627',  # أ → ا
    '\u0625': '\u0627',  # إ → ا
    '\u0649': '\u064A',  # ى → ي
    '\u0640': '',  # ـ (remove)
}

# Diacritics to remove
ARABIC_DIACRITICS = [
    '\u064B', '\u064C', '\u064D',  # Tanween
    '\u064E', '\u064F', '\u0650',  # Harakat
    '\u0651',  # Shadda
    '\u0652',  # Sukun
]

WHITESPACE_PATTERN = re.compile(r'[\s\u00A0\u200B\u200C\u200D]+')


def normalize_arabic_digits(text: str) -> str:
    """Normalize Arabic-Indic digits to Western digits."""
    if not text:
        return text
    return ''.join(ARABIC_INDIC_DIGITS.get(c, c) for c in text)


def normalize_unicode(text: str) -> str:
    """Unicode NFKC normalization."""
    return unicodedata.normalize('NFKC', text) if text else text


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics."""
    if not text:
        return text
    for d in ARABIC_DIACRITICS:
        text = text.replace(d, '')
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace."""
    if not text:
        return text
    text = WHITESPACE_PATTERN.sub(' ', text)
    return text.strip()


def normalize_arabic_text(
    text: str,
    normalize_digits: bool = True,
    remove_diacritics_flag: bool = True,
    normalize_ws: bool = True,
) -> str:
    """Comprehensive Arabic text normalization."""
    if not text or not isinstance(text, str):
        return text or ""
    
    text = normalize_unicode(text)
    if remove_diacritics_flag:
        text = remove_diacritics(text)
    for confusing, correct in ARABIC_OCR_CONFUSIONS.items():
        text = text.replace(confusing, correct)
    if normalize_digits:
        text = normalize_arabic_digits(text)
    if normalize_ws:
        text = normalize_whitespace(text)
    
    return text


def normalize_numeric_candidate(text: str) -> str:
    """Normalize numeric candidate (digits only)."""
    if not text:
        return text or ""
    text = normalize_arabic_digits(text)
    return ''.join(c for c in text if c.isdigit())


def normalize_gender_text(text: str) -> Optional[str]:
    """Normalize gender text to 'male' or 'female'."""
    if not text:
        return None
    
    text = normalize_arabic_text(text).lower()
    
    if any(m in text for m in ['ذكر', 'دكر', 'male']):
        return 'male'
    if any(f in text for f in ['انثى', 'أنثى', 'انثي', 'female']):
        return 'female'
    
    return None


def is_mostly_arabic(text: str) -> bool:
    """Check if text is mostly Arabic."""
    if not text:
        return False
    arabic_range = range(0x0600, 0x06FF + 1)
    arabic_count = sum(1 for c in text if ord(c) in arabic_range)
    return arabic_count > len(text) * 0.5


def is_mostly_numeric(text: str) -> bool:
    """Check if text is mostly numeric."""
    if not text:
        return False
    digits = sum(1 for c in text if c.isdigit())
    non_ws = sum(1 for c in text if not c.isspace())
    return non_ws > 0 and digits / non_ws > 0.7
