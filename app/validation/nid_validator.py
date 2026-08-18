"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Egyptian National ID (NID) Structural Validator Module

Implements comprehensive validation of Egyptian National ID numbers:
- 14-digit format validation
- Century code extraction and validation (digit 1)
- Birth date derivation from digits 3-8
- Governorate code validation (digits 9-10)
- Gender derivation from digit 11 (odd=male, even=female)
- Check digit validation (digit 14)

Egyptian NID Structure:
Position 1: Century code (2=1900s, 3=2000s)
Position 2: Unused/reserved
Positions 3-8: Birth date (YYMMDD)
Positions 9-10: Governorate code
Position 11: Gender indicator (odd=male, even=female)
Positions 12-13: Sequence number
Position 14: Check digit
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date
import re

from app.validation.governorate_mapping import (
    get_governorate_by_code,
    validate_nid_governorate_code,
    GOVERNORATE_CODES,
)


@dataclass
class NIDValidationResult:
    """Result of NID structural validation."""
    is_valid: bool
    nid: str
    normalized_nid: str
    
    # Derived information
    century: Optional[int] = None  # 1900 or 2000
    birth_year: Optional[int] = None
    birth_month: Optional[int] = None
    birth_day: Optional[int] = None
    date_of_birth: Optional[str] = None  # YYYY-MM-DD format
    
    governorate_code: Optional[str] = None
    governorate_name_arabic: Optional[str] = None
    governorate_name_english: Optional[str] = None
    
    gender: Optional[str] = None  # 'male' or 'female'
    
    # Validation details
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Confidence scores
    format_valid: bool = False
    date_valid: bool = False
    governorate_valid: bool = False
    check_digit_valid: Optional[bool] = None
    
    # Status categories
    status: str = "unknown"  # valid, invalid, malformed, unknown


class NIDValidator:
    """
    Egyptian National ID Structural Validator.
    
    Validates the complete structure of Egyptian NID numbers
    and derives embedded information.
    """
    
    # Century codes
    CENTURY_1900 = '2'
    CENTURY_2000 = '3'
    
    # Valid century mapping
    CENTURY_MAP = {
        CENTURY_1900: 1900,
        CENTURY_2000: 2000,
    }
    
    def __init__(self):
        """Initialize the NID validator."""
        pass
    
    def validate(self, nid: str) -> NIDValidationResult:
        """
        Validate an Egyptian National ID number.
        
        Args:
            nid: Raw NID string (may contain Arabic-Indic digits, separators, etc.)
            
        Returns:
            NIDValidationResult with validation details and derived information
        """
        # Normalize input
        normalized = self._normalize_nid(nid)
        
        result = NIDValidationResult(
            is_valid=False,
            nid=nid,
            normalized_nid=normalized,
        )
        
        # Check if we have anything to validate
        if not normalized:
            result.errors.append("Empty or invalid NID")
            result.status = "malformed"
            return result
        
        # Step 1: Format validation (14 digits)
        if not self._validate_format(normalized, result):
            result.status = "malformed"
            return result
        
        result.format_valid = True
        
        # Step 2: Extract and validate century
        self._extract_century(normalized, result)
        
        # Step 3: Extract and validate birth date
        self._extract_and_validate_date(normalized, result)
        
        # Step 4: Extract and validate governorate code
        self._extract_and_validate_governorate(normalized, result)
        
        # Step 5: Extract gender
        self._extract_gender(normalized, result)
        
        # Step 6: Validate check digit (if algorithm is known/available)
        # Note: Egyptian NID check digit algorithm is not publicly documented
        # We mark it as unknown rather than making assumptions
        result.check_digit_valid = None
        result.warnings.append("Check digit validation not available (algorithm not public)")
        
        # Determine overall validity
        self._determine_validity(result)
        
        return result
    
    def _normalize_nid(self, nid: str) -> str:
        """
        Normalize NID input to pure digits.
        
        Handles:
        - Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩)
        - Western digits (0123456789)
        - Separators (-, spaces, dots)
        - Whitespace
        """
        if not nid:
            return ""
        
        # Convert to string and strip whitespace
        nid_str = str(nid).strip()
        
        # Remove common separators
        separators = ['-', ' ', '.', '/', '_', ',']
        for sep in separators:
            nid_str = nid_str.replace(sep, '')
        
        # Convert Arabic-Indic digits to Western digits
        arabic_indic = '٠١٢٣٤٥٦٧٨٩'
        western = '0123456789'
        
        translated = []
        for char in nid_str:
            if char in arabic_indic:
                translated.append(western[arabic_indic.index(char)])
            elif char.isdigit():
                translated.append(char)
            # Skip non-digit characters
        
        return ''.join(translated)
    
    def _validate_format(self, nid: str, result: NIDValidationResult) -> bool:
        """
        Validate NID format (exactly 14 digits).
        
        Returns:
            True if format is valid, False otherwise
        """
        # Check length
        if len(nid) != 14:
            result.errors.append(f"Invalid NID length: expected 14 digits, got {len(nid)}")
            return False
        
        # Check all characters are digits
        if not nid.isdigit():
            non_digits = [c for c in nid if not c.isdigit()]
            result.errors.append(f"NID contains non-digit characters: {non_digits}")
            return False
        
        # Check first digit is valid century code (2 or 3)
        if nid[0] not in (self.CENTURY_1900, self.CENTURY_2000):
            result.errors.append(f"Invalid century code: '{nid[0]}' (expected '2' or '3')")
            result.warnings.append(f"First digit should be '{self.CENTURY_1900}' (1900s) or '{self.CENTURY_2000}' (2000s)")
            # Don't fail completely, just warn
        
        return True
    
    def _extract_century(self, nid: str, result: NIDValidationResult) -> None:
        """Extract century from first digit."""
        century_code = nid[0]
        
        if century_code in self.CENTURY_MAP:
            result.century = self.CENTURY_MAP[century_code]
        else:
            result.century = None
            result.errors.append(f"Unknown century code: {century_code}")
    
    def _extract_and_validate_date(self, nid: str, result: NIDValidationResult) -> None:
        """
        Extract and validate birth date from positions 2-7 (1-indexed).
        
        Egyptian NID structure (1-indexed positions):
        Position 1: Century code
        Positions 2-3: Birth year YY
        Positions 4-5: Birth month MM
        Positions 6-7: Birth day DD
        Positions 8-9: Governorate code
        Position 10: Gender indicator
        Positions 11-14: Sequence and check digit
        
        In 0-indexed Python string:
        nid[0] = century
        nid[1:3] = YY (year)
        nid[3:5] = MM (month)
        nid[5:7] = DD (day)
        nid[7:9] = governorate code
        nid[9] = gender digit
        """
        try:
            # Extract date components using correct indices
            yy = int(nid[1:3])  # Year at positions 2-3 (0-indexed: 1:3)
            mm = int(nid[3:5])  # Month at positions 4-5 (0-indexed: 3:5)
            dd = int(nid[5:7])  # Day at positions 6-7 (0-indexed: 5:7)
            
            result.birth_year = yy
            result.birth_month = mm
            result.birth_day = dd
            
            # Determine full year based on century
            if result.century:
                full_year = result.century + yy
            else:
                # Default to 2000s if century unclear
                full_year = 2000 + yy
            
            # Validate month
            if mm < 1 or mm > 12:
                result.errors.append(f"Invalid month: {mm} (must be 01-12)")
                result.date_valid = False
                return
            
            # Validate day
            if dd < 1 or dd > 31:
                result.errors.append(f"Invalid day: {dd} (must be 01-31)")
                result.date_valid = False
                return
            
            # Try to create a date object for full validation
            try:
                birth_date = date(full_year, mm, dd)
                result.date_of_birth = birth_date.strftime("%Y-%m-%d")
                result.date_valid = True
                
                # Additional sanity checks
                today = date.today()
                if birth_date > today:
                    result.errors.append(f"Birth date {result.date_of_birth} is in the future")
                    result.date_valid = False
                    result.warnings.append("Date appears invalid (future date)")
                
                # Reasonable age check (optional warning)
                age_years = (today - birth_date).days / 365.25
                if age_years > 120:
                    result.warnings.append(f"Unusual age: {age_years:.0f} years (possible OCR error)")
                elif age_years < 0:
                    result.warnings.append("Calculated age is negative (date validation failed)")
                    
            except ValueError as e:
                result.errors.append(f"Invalid date: {yy}-{mm:02d}-{dd:02d} ({str(e)})")
                result.date_valid = False
                
        except (ValueError, IndexError) as e:
            result.errors.append(f"Failed to extract date from NID: {str(e)}")
            result.date_valid = False
    
    def _extract_and_validate_governorate(self, nid: str, result: NIDValidationResult) -> None:
        """
        Extract and validate governorate code from positions 9-10.
        """
        gov_code = nid[8:10]
        result.governorate_code = gov_code
        
        is_valid, gov_name, error = validate_nid_governorate_code(gov_code)
        
        if is_valid:
            result.governorate_name_arabic = gov_name
            result.governorate_valid = True
            
            # Get English name
            gov_info = get_governorate_by_code(gov_code)
            if gov_info:
                result.governorate_name_english = gov_info.name_english
        else:
            result.errors.append(f"Invalid governorate code: {gov_code} ({error})")
            result.governorate_valid = False
            
            # Check if it's a common OCR error
            common_errors = {
                '00': 'Possible OCR error: detected as 00',
                '99': 'Possible OCR error: detected as 99',
                '28': 'Code 28 is not assigned (max is 27)',
                '29': 'Code 29 is not assigned (max is 27)',
            }
            if gov_code in common_errors:
                result.warnings.append(common_errors[gov_code])
    
    def _extract_gender(self, nid: str, result: NIDValidationResult) -> None:
        """
        Extract gender from position 11.
        Odd = Male, Even = Female
        """
        try:
            gender_digit = int(nid[10])
            
            if gender_digit % 2 == 1:  # Odd
                result.gender = 'male'
            else:  # Even
                result.gender = 'female'
                
        except (ValueError, IndexError) as e:
            result.errors.append(f"Failed to extract gender digit: {str(e)}")
            result.gender = None
    
    def _determine_validity(self, result: NIDValidationResult) -> None:
        """
        Determine overall validity status based on component validations.
        """
        # Critical requirements for validity
        critical_valid = (
            result.format_valid and
            result.date_valid and
            result.governorate_valid
        )
        
        if critical_valid:
            result.is_valid = True
            result.status = "valid"
        else:
            result.is_valid = False
            
            # Categorize the type of invalidity
            if not result.format_valid:
                result.status = "malformed"
            elif not result.date_valid and not result.governorate_valid:
                result.status = "invalid"
            elif not result.date_valid:
                result.status = "invalid_date"
            elif not result.governorate_valid:
                result.status = "invalid_governorate"
            else:
                result.status = "invalid"
    
    def derive_dob_from_nid(self, nid: str) -> Optional[str]:
        """
        Convenience method to extract only the date of birth from NID.
        
        Args:
            nid: Raw NID string
            
        Returns:
            Date of birth in YYYY-MM-DD format, or None if extraction fails
        """
        result = self.validate(nid)
        return result.date_of_birth
    
    def derive_gender_from_nid(self, nid: str) -> Optional[str]:
        """
        Convenience method to extract only the gender from NID.
        
        Args:
            nid: Raw NID string
            
        Returns:
            'male' or 'female', or None if extraction fails
        """
        result = self.validate(nid)
        return result.gender
    
    def derive_governorate_from_nid(self, nid: str) -> Optional[Dict[str, str]]:
        """
        Convenience method to extract governorate information from NID.
        
        Args:
            nid: Raw NID string
            
        Returns:
            Dictionary with code, arabic_name, english_name, or None if extraction fails
        """
        result = self.validate(nid)
        
        if not result.governorate_code:
            return None
        
        return {
            'code': result.governorate_code,
            'arabic_name': result.governorate_name_arabic,
            'english_name': result.governorate_name_english,
            'is_valid': result.governorate_valid,
        }
    
    def compare_dob(self, nid: str, ocr_dob: str) -> Dict[str, Any]:
        """
        Compare NID-derived DOB with OCR-extracted DOB.
        
        Args:
            nid: NID string
            ocr_dob: DOB extracted via OCR (various formats supported)
            
        Returns:
            Comparison result dictionary
        """
        nid_result = self.validate(nid)
        
        if not nid_result.date_of_birth:
            return {
                'status': 'unknown',
                'reason': 'Could not derive DOB from NID',
                'nid_dob': None,
                'ocr_dob': ocr_dob,
            }
        
        # Normalize OCR DOB
        normalized_ocr = self._normalize_date(ocr_dob)
        
        if normalized_ocr is None:
            return {
                'status': 'unknown',
                'reason': 'Could not normalize OCR DOB',
                'nid_dob': nid_result.date_of_birth,
                'ocr_dob': ocr_dob,
                'ocr_dob_normalized': None,
            }
        
        # Compare
        if nid_result.date_of_birth == normalized_ocr:
            return {
                'status': 'consistent',
                'nid_dob': nid_result.date_of_birth,
                'ocr_dob': ocr_dob,
                'ocr_dob_normalized': normalized_ocr,
            }
        else:
            return {
                'status': 'conflict',
                'nid_dob': nid_result.date_of_birth,
                'ocr_dob': ocr_dob,
                'ocr_dob_normalized': normalized_ocr,
                'difference': f"NID: {nid_result.date_of_birth} vs OCR: {normalized_ocr}",
            }
    
    def compare_gender(self, nid: str, ocr_gender: str) -> Dict[str, Any]:
        """
        Compare NID-derived gender with OCR-extracted gender.
        
        Args:
            nid: NID string
            ocr_gender: Gender extracted via OCR
            
        Returns:
            Comparison result dictionary
        """
        nid_result = self.validate(nid)
        
        if not nid_result.gender:
            return {
                'status': 'unknown',
                'reason': 'Could not derive gender from NID',
                'nid_gender': None,
                'ocr_gender': ocr_gender,
            }
        
        # Normalize OCR gender
        normalized_ocr = self._normalize_gender(ocr_gender)
        
        if normalized_ocr is None:
            return {
                'status': 'unknown',
                'reason': 'Could not normalize OCR gender',
                'nid_gender': nid_result.gender,
                'ocr_gender': ocr_gender,
            }
        
        # Compare
        if nid_result.gender == normalized_ocr:
            return {
                'status': 'consistent',
                'nid_gender': nid_result.gender,
                'ocr_gender': ocr_gender,
                'ocr_gender_normalized': normalized_ocr,
            }
        else:
            return {
                'status': 'conflict',
                'nid_gender': nid_result.gender,
                'ocr_gender': ocr_gender,
                'ocr_gender_normalized': normalized_ocr,
                'difference': f"NID: {nid_result.gender} vs OCR: {normalized_ocr}",
            }
    
    def compare_governorate(self, nid: str, ocr_governorate: str) -> Dict[str, Any]:
        """
        Compare NID-derived governorate with OCR-extracted governorate.
        
        Args:
            nid: NID string
            ocr_governorate: Governorate extracted via OCR
            
        Returns:
            Comparison result dictionary
        """
        nid_result = self.validate(nid)
        
        if not nid_result.governorate_code:
            return {
                'status': 'unknown',
                'reason': 'Could not derive governorate from NID',
                'nid_governorate': None,
                'ocr_governorate': ocr_governorate,
            }
        
        # Try to match OCR text to governorate
        from app.validation.governorate_mapping import (
            get_code_by_arabic_name,
            get_code_by_english_name,
        )
        
        ocr_code = get_code_by_arabic_name(ocr_governorate) or get_code_by_english_name(ocr_governorate)
        
        if ocr_code is None:
            return {
                'status': 'unknown',
                'reason': 'Could not match OCR text to governorate',
                'nid_governorate': nid_result.governorate_name_arabic,
                'ocr_governorate': ocr_governorate,
            }
        
        # Compare codes
        if nid_result.governorate_code == ocr_code:
            return {
                'status': 'consistent',
                'nid_governorate_code': nid_result.governorate_code,
                'nid_governorate_name': nid_result.governorate_name_arabic,
                'ocr_governorate': ocr_governorate,
                'ocr_governorate_code': ocr_code,
            }
        else:
            ocr_info = get_governorate_by_code(ocr_code)
            return {
                'status': 'conflict',
                'nid_governorate_code': nid_result.governorate_code,
                'nid_governorate_name': nid_result.governorate_name_arabic,
                'ocr_governorate': ocr_governorate,
                'ocr_governorate_code': ocr_code,
                'ocr_governorate_name': ocr_info.name_arabic if ocr_info else None,
                'difference': f"NID: {nid_result.governorate_name_arabic} vs OCR: {ocr_governorate}",
            }
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """
        Normalize various date formats to YYYY-MM-DD.
        
        Supported formats:
        - YYYY-MM-DD
        - DD/MM/YYYY
        - DD-MM-YYYY
        - YYYY/MM/DD
        - Arabic-Indic digits
        """
        if not date_str:
            return None
        
        # Convert Arabic-Indic digits
        arabic_indic = '٠١٢٣٤٥٦٧٨٩'
        western = '0123456789'
        for i, char in enumerate(arabic_indic):
            date_str = date_str.replace(char, western[i])
        
        # Try various patterns
        patterns = [
            # YYYY-MM-DD or YYYY/MM/DD
            (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
            # DD/MM/YYYY or DD-MM-YYYY
            (r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', lambda m: f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"),
            # YY-MM-DD
            (r'(\d{2})[-/](\d{1,2})[-/](\d{1,2})', lambda m: self._yy_to_yyyy(m)),
        ]
        
        import re
        for pattern, formatter in patterns:
            match = re.match(pattern, date_str.strip())
            if match:
                try:
                    return formatter(match)
                except (ValueError, AttributeError):
                    continue
        
        return None
    
    def _yy_to_yyyy(self, match) -> str:
        """Convert 2-digit year to 4-digit year."""
        yy = int(match.group(1))
        mm = int(match.group(2))
        dd = int(match.group(3))
        
        # Assume 2000s for years 00-25, 1900s for 26-99
        if yy <= 25:
            yyyy = 2000 + yy
        else:
            yyyy = 1900 + yy
        
        return f"{yyyy}-{mm:02d}-{dd:02d}"
    
    def _normalize_gender(self, gender_str: str) -> Optional[str]:
        """
        Normalize gender strings to 'male' or 'female'.
        
        Supports Arabic and English variants.
        """
        if not gender_str:
            return None
        
        normalized = gender_str.strip().lower()
        
        # Arabic variants
        arabic_male = ['ذكر', 'رجل', 'م', 'male']
        arabic_female = ['أنثى', 'امرأة', 'ف', 'female']
        
        # Check Arabic
        if any(var in normalized for var in arabic_male):
            return 'male'
        if any(var in normalized for var in arabic_female):
            return 'female'
        
        # Check English
        if normalized in ['male', 'm', 'man', 'boy']:
            return 'male'
        if normalized in ['female', 'f', 'woman', 'girl']:
            return 'female'
        
        return None


# Export public API
__all__ = [
    'NIDValidator',
    'NIDValidationResult',
]
