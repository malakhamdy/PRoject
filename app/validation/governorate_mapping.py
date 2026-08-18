"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Egyptian Governorate Mapping Module

Provides centralized mapping of Egyptian governorate codes to names.
Covers all 27 governorates of Egypt.

Source: Official Egyptian Central Agency for Public Mobilization and Statistics (CAPMAS)
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GovernorateInfo:
    """Information about an Egyptian governorate."""
    code: str
    name_arabic: str
    name_english: str
    region: str  # Upper Egypt, Lower Egypt, Canal, Frontier, Greater Cairo, Alexandria


# Complete mapping of Egyptian governorate codes (2 digits) to names
# Codes are based on the Egyptian National ID encoding system
GOVERNORATE_CODES: Dict[str, GovernorateInfo] = {
    # Greater Cairo (القاهرة الكبرى)
    '01': GovernorateInfo('01', 'القاهرة', 'Cairo', 'Greater Cairo'),
    '02': GovernorateInfo('02', 'الإسكندرية', 'Alexandria', 'Alexandria'),
    '03': GovernorateInfo('03', 'بورسعيد', 'Port Said', 'Canal'),
    '04': GovernorateInfo('04', 'السويس', 'Suez', 'Canal'),
    '05': GovernorateInfo('05', 'دمياط', 'Damietta', 'Lower Egypt'),
    
    # Lower Egypt (وجه بحري)
    '06': GovernorateInfo('06', 'الدقهلية', 'Dakahlia', 'Lower Egypt'),
    '07': GovernorateInfo('07', 'الشرقية', 'Sharqia', 'Lower Egypt'),
    '08': GovernorateInfo('08', 'القليوبية', 'Qalyubia', 'Lower Egypt'),
    '09': GovernorateInfo('09', 'كفر الشيخ', 'Kafr El Sheikh', 'Lower Egypt'),
    '10': GovernorateInfo('10', 'الغربية', 'Gharbia', 'Lower Egypt'),
    '11': GovernorateInfo('11', 'المنوفية', 'Menoufia', 'Lower Egypt'),
    '12': GovernorateInfo('12', 'البحيرة', 'Beheira', 'Lower Egypt'),
    '13': GovernorateInfo('13', 'الإسماعيلية', 'Ismailia', 'Canal'),
    
    # Greater Cairo (continued)
    '14': GovernorateInfo('14', 'الجيزة', 'Giza', 'Greater Cairo'),
    '15': GovernorateInfo('15', 'بني سويف', 'Beni Suef', 'Upper Egypt'),
    '16': GovernorateInfo('16', 'الفيوم', 'Fayoum', 'Upper Egypt'),
    
    # Upper Egypt (وجه قبلي)
    '17': GovernorateInfo('17', 'المنيا', 'Minya', 'Upper Egypt'),
    '18': GovernorateInfo('18', 'أسيوط', 'Assiut', 'Upper Egypt'),
    '19': GovernorateInfo('19', 'سوهاج', 'Sohag', 'Upper Egypt'),
    '20': GovernorateInfo('20', 'قنا', 'Qena', 'Upper Egypt'),
    '21': GovernorateInfo('21', 'الأقصر', 'Luxor', 'Upper Egypt'),
    '22': GovernorateInfo('22', 'أسوان', 'Aswan', 'Upper Egypt'),
    '23': GovernorateInfo('23', 'الوادي الجديد', 'New Valley', 'Frontier'),
    
    # Canal and Frontier (قناة السويس والحدود)
    '24': GovernorateInfo('24', 'البحر الأحمر', 'Red Sea', 'Frontier'),
    '25': GovernorateInfo('25', 'مطروح', 'Matrouh', 'Frontier'),
    '26': GovernorateInfo('26', 'شمال سيناء', 'North Sinai', 'Frontier'),
    '27': GovernorateInfo('27', 'جنوب سيناء', 'South Sinai', 'Frontier'),
}

# Reverse mapping: Arabic name → code
ARABIC_NAME_TO_CODE: Dict[str, str] = {
    info.name_arabic: code for code, info in GOVERNORATE_CODES.items()
}

# English name → code
ENGLISH_NAME_TO_CODE: Dict[str, str] = {
    info.name_english: code for code, info in GOVERNORATE_CODES.items()
}

# Common variations and OCR noise patterns
GOVERNORATE_VARIATIONS: Dict[str, str] = {
    # Cairo variations
    'القاهره': '01',  # Without hamza
    'القاهرة الكبري': '01',
    'cairo': '01',
    
    # Alexandria variations
    'الاسكندرية': '02',  # Without hamza
    'الاسكندريه': '02',
    'alexandria': '02',
    'alex': '02',
    
    # Port Said variations
    'بور سعيد': '03',  # With space
    'port said': '03',
    'portsaid': '03',
    
    # Suez variations
    'السويس': '04',
    'suez': '04',
    
    # Giza variations
    'الجيزه': '14',  # Without hamza
    'giza': '14',
    'gizeh': '14',
    
    # Luxor variations
    'الاقصر': '21',  # Without hamza
    'luxor': '21',
    
    # Aswan variations
    'اسوان': '22',  # Without alif
    'aswan': '22',
}


def get_governorate_by_code(code: str) -> Optional[GovernorateInfo]:
    """
    Get governorate information by its 2-digit code.
    
    Args:
        code: 2-digit governorate code (e.g., '01', '14')
        
    Returns:
        GovernorateInfo if valid code, None otherwise
        
    Example:
        >>> get_governorate_by_code('01')
        GovernorateInfo(code='01', name_arabic='القاهرة', name_english='Cairo', ...)
    """
    if not code or len(code) != 2:
        return None
    
    # Normalize: remove leading/trailing whitespace
    code = code.strip()
    
    return GOVERNORATE_CODES.get(code)


def get_code_by_arabic_name(name: str) -> Optional[str]:
    """
    Get governorate code by Arabic name.
    
    Args:
        name: Arabic governorate name
        
    Returns:
        2-digit code if found, None otherwise
    """
    if not name:
        return None
    
    # Normalize Arabic text
    normalized = _normalize_arabic(name)
    
    # Direct lookup
    if normalized in ARABIC_NAME_TO_CODE:
        return ARABIC_NAME_TO_CODE[normalized]
    
    # Check variations
    if normalized in GOVERNORATE_VARIATIONS:
        return GOVERNORATE_VARIATIONS[normalized]
    
    return None


def get_code_by_english_name(name: str) -> Optional[str]:
    """
    Get governorate code by English name.
    
    Args:
        name: English governorate name
        
    Returns:
        2-digit code if found, None otherwise
    """
    if not name:
        return None
    
    normalized = name.strip().lower()
    
    # Direct lookup
    if normalized in ENGLISH_NAME_TO_CODE:
        return ENGLISH_NAME_TO_CODE[normalized]
    
    # Check variations
    if normalized in GOVERNORATE_VARIATIONS:
        return GOVERNORATE_VARIATIONS[normalized]
    
    return None


def is_valid_code(code: str) -> bool:
    """
    Check if a governorate code is valid.
    
    Args:
        code: 2-digit code to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not code or len(code) != 2:
        return False
    
    return code.strip() in GOVERNORATE_CODES


def get_all_codes() -> list:
    """Return list of all valid governorate codes."""
    return sorted(GOVERNORATE_CODES.keys())


def get_all_names(language: str = 'arabic') -> list:
    """
    Return list of all governorate names.
    
    Args:
        language: 'arabic' or 'english'
        
    Returns:
        List of governorate names
    """
    if language.lower() == 'english':
        return [info.name_english for info in GOVERNORATE_CODES.values()]
    else:
        return [info.name_arabic for info in GOVERNORATE_CODES.values()]


def get_governorates_by_region(region: str) -> list:
    """
    Get all governorates in a specific region.
    
    Args:
        region: Region name ('Greater Cairo', 'Lower Egypt', 'Upper Egypt', 
                'Canal', 'Frontier', 'Alexandria')
                
    Returns:
        List of GovernorateInfo objects
    """
    return [
        info for info in GOVERNORATE_CODES.values()
        if info.region.lower() == region.lower()
    ]


def _normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for matching.
    
    Handles:
    - Alef variations (ا, أ, إ, آ)
    - Yeh variations (ي, ى)
    - Hamza variations
    - Whitespace
    """
    if not text:
        return ""
    
    # Remove whitespace
    text = text.strip()
    
    # Normalize alef variations
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    
    # Normalize yeh variations
    text = text.replace('ى', 'ي')
    
    # Remove hamza
    text = text.replace('ء', '')
    
    return text


def validate_nid_governorate_code(code: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate a governorate code extracted from NID.
    
    Args:
        code: 2-digit code from NID positions 9-10
        
    Returns:
        Tuple of (is_valid, governorate_name_arabic, error_message)
    """
    if not code:
        return False, None, "Empty governorate code"
    
    if len(code) != 2:
        return False, None, f"Invalid code length: expected 2, got {len(code)}"
    
    if not code.isdigit():
        return False, None, f"Code must be numeric: got '{code}'"
    
    info = get_governorate_by_code(code)
    
    if info is None:
        return False, None, f"Unknown governorate code: {code}"
    
    return True, info.name_arabic, None


# Export public API
__all__ = [
    'GovernorateInfo',
    'GOVERNORATE_CODES',
    'ARABIC_NAME_TO_CODE',
    'ENGLISH_NAME_TO_CODE',
    'GOVERNORATE_VARIATIONS',
    'get_governorate_by_code',
    'get_code_by_arabic_name',
    'get_code_by_english_name',
    'is_valid_code',
    'get_all_codes',
    'get_all_names',
    'get_governorates_by_region',
    'validate_nid_governorate_code',
]
