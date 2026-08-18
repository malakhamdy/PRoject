"""OCI Validation Module - Egyptian National ID validation utilities."""

from app.validation.governorate_mapping import (
    GovernorateInfo,
    GOVERNORATE_CODES,
    get_governorate_by_code,
    get_code_by_arabic_name,
    get_code_by_english_name,
    is_valid_code,
    validate_nid_governorate_code,
)

from app.validation.nid_validator import (
    NIDValidator,
    NIDValidationResult,
)

__all__ = [
    # Governorate mapping
    'GovernorateInfo',
    'GOVERNORATE_CODES',
    'get_governorate_by_code',
    'get_code_by_arabic_name',
    'get_code_by_english_name',
    'is_valid_code',
    'validate_nid_governorate_code',
    
    # NID validator
    'NIDValidator',
    'NIDValidationResult',
]
