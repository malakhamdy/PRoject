"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Cross-Field Consistency Engine Module

Implements cross-field validation comparing:
- NID-derived DOB vs OCR-extracted DOB
- NID-derived Gender vs OCR-extracted Gender
- NID-derived Governorate vs OCR-extracted Governorate
- Normalized NID vs OCR-extracted NID

Distinguishes between:
- CONSISTENT: Fields agree
- CONFLICT: Fields disagree
- UNKNOWN: Cannot determine (missing/invalid data)
- MISSING: Required field not available
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.validation.nid_validator import NIDValidator, NIDValidationResult


class ConsistencyStatus(str, Enum):
    """Possible consistency check results."""
    CONSISTENT = "consistent"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"
    MISSING = "missing"


@dataclass
class ConsistencyCheck:
    """Result of a single consistency check."""
    check_name: str
    status: ConsistencyStatus
    details: str
    field1_name: str
    field1_value: Optional[str]
    field2_name: str
    field2_value: Optional[str]
    confidence: float = 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyResult:
    """Overall consistency assessment result."""
    all_checks: List[ConsistencyCheck] = field(default_factory=list)
    
    # Summary statuses
    nid_dob_status: ConsistencyStatus = ConsistencyStatus.UNKNOWN
    nid_gender_status: ConsistencyStatus = ConsistencyStatus.UNKNOWN
    nid_governorate_status: ConsistencyStatus = ConsistencyStatus.UNKNOWN
    
    # Overall assessment
    overall_status: ConsistencyStatus = ConsistencyStatus.UNKNOWN
    has_conflicts: bool = False
    conflict_count: int = 0
    consistent_count: int = 0
    unknown_count: int = 0
    
    # Detailed results
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)


class ConsistencyEngine:
    """
    Cross-Field Consistency Engine for Egyptian National ID.
    
    Compares independently extracted fields to detect conflicts
    and validate data integrity.
    """
    
    def __init__(self, nid_validator: Optional[NIDValidator] = None):
        """
        Initialize the consistency engine.
        
        Args:
            nid_validator: NID validator instance (creates default if not provided)
        """
        self.nid_validator = nid_validator or NIDValidator()
    
    def check_all(
        self,
        nid_value: Optional[str],
        dob_value: Optional[str],
        gender_value: Optional[str],
        governorate_value: Optional[str],
    ) -> ConsistencyResult:
        """
        Run all consistency checks.
        
        Args:
            nid_value: Extracted/normalized NID
            dob_value: OCR-extracted DOB
            gender_value: OCR-extracted gender
            governorate_value: OCR-extracted governorate
            
        Returns:
            ConsistencyResult with all check results
        """
        result = ConsistencyResult()
        
        # Check 1: NID ↔ DOB
        nid_dob_check = self.check_nid_dob(nid_value, dob_value)
        result.all_checks.append(nid_dob_check)
        result.nid_dob_status = nid_dob_check.status
        
        # Check 2: NID ↔ Gender
        nid_gender_check = self.check_nid_gender(nid_value, gender_value)
        result.all_checks.append(nid_gender_check)
        result.nid_gender_status = nid_gender_check.status
        
        # Check 3: NID ↔ Governorate
        nid_gov_check = self.check_nid_governorate(nid_value, governorate_value)
        result.all_checks.append(nid_gov_check)
        result.nid_governorate_status = nid_gov_check.status
        
        # Calculate summary statistics
        self._calculate_summary(result)
        
        # Determine overall status
        self._determine_overall_status(result)
        
        # Generate recommendations
        self._generate_recommendations(result)
        
        return result
    
    def check_nid_dob(
        self,
        nid_value: Optional[str],
        dob_value: Optional[str],
    ) -> ConsistencyCheck:
        """
        Compare NID-derived DOB with OCR-extracted DOB.
        
        Args:
            nid_value: NID string
            dob_value: OCR-extracted DOB
            
        Returns:
            ConsistencyCheck result
        """
        # Handle missing values
        if not nid_value:
            return ConsistencyCheck(
                check_name="nid_dob",
                status=ConsistencyStatus.MISSING,
                details="NID value not provided",
                field1_name="nid",
                field1_value=None,
                field2_name="dob",
                field2_value=dob_value,
            )
        
        if not dob_value:
            return ConsistencyCheck(
                check_name="nid_dob",
                status=ConsistencyStatus.MISSING,
                details="DOB value not provided",
                field1_name="nid",
                field1_value=nid_value,
                field2_name="dob",
                field2_value=None,
            )
        
        # Use NID validator to compare
        comparison = self.nid_validator.compare_dob(nid_value, dob_value)
        
        status_map = {
            'consistent': ConsistencyStatus.CONSISTENT,
            'conflict': ConsistencyStatus.CONFLICT,
            'unknown': ConsistencyStatus.UNKNOWN,
        }
        
        status = status_map.get(comparison.get('status', 'unknown'), ConsistencyStatus.UNKNOWN)
        
        return ConsistencyCheck(
            check_name="nid_dob",
            status=status,
            details=comparison.get('difference', comparison.get('reason', 'Dates match')),
            field1_name="nid_dob",
            field1_value=comparison.get('nid_dob'),
            field2_name="ocr_dob",
            field2_value=comparison.get('ocr_dob'),
            confidence=1.0 if status == ConsistencyStatus.CONSISTENT else 0.5,
            evidence=comparison,
        )
    
    def check_nid_gender(
        self,
        nid_value: Optional[str],
        gender_value: Optional[str],
    ) -> ConsistencyCheck:
        """
        Compare NID-derived gender with OCR-extracted gender.
        
        Args:
            nid_value: NID string
            gender_value: OCR-extracted gender
            
        Returns:
            ConsistencyCheck result
        """
        if not nid_value:
            return ConsistencyCheck(
                check_name="nid_gender",
                status=ConsistencyStatus.MISSING,
                details="NID value not provided",
                field1_name="nid",
                field1_value=None,
                field2_name="gender",
                field2_value=gender_value,
            )
        
        if not gender_value:
            return ConsistencyCheck(
                check_name="nid_gender",
                status=ConsistencyStatus.MISSING,
                details="Gender value not provided",
                field1_name="nid",
                field1_value=nid_value,
                field2_name="gender",
                field2_value=None,
            )
        
        # Use NID validator to compare
        comparison = self.nid_validator.compare_gender(nid_value, gender_value)
        
        status_map = {
            'consistent': ConsistencyStatus.CONSISTENT,
            'conflict': ConsistencyStatus.CONFLICT,
            'unknown': ConsistencyStatus.UNKNOWN,
        }
        
        status = status_map.get(comparison.get('status', 'unknown'), ConsistencyStatus.UNKNOWN)
        
        return ConsistencyCheck(
            check_name="nid_gender",
            status=status,
            details=comparison.get('difference', comparison.get('reason', 'Genders match')),
            field1_name="nid_gender",
            field1_value=comparison.get('nid_gender'),
            field2_name="ocr_gender",
            field2_value=comparison.get('ocr_gender'),
            confidence=1.0 if status == ConsistencyStatus.CONSISTENT else 0.5,
            evidence=comparison,
        )
    
    def check_nid_governorate(
        self,
        nid_value: Optional[str],
        governorate_value: Optional[str],
    ) -> ConsistencyCheck:
        """
        Compare NID-derived governorate with OCR-extracted governorate.
        
        Args:
            nid_value: NID string
            governorate_value: OCR-extracted governorate
            
        Returns:
            ConsistencyCheck result
        """
        if not nid_value:
            return ConsistencyCheck(
                check_name="nid_governorate",
                status=ConsistencyStatus.MISSING,
                details="NID value not provided",
                field1_name="nid",
                field1_value=None,
                field2_name="governorate",
                field2_value=governorate_value,
            )
        
        if not governorate_value:
            return ConsistencyCheck(
                check_name="nid_governorate",
                status=ConsistencyStatus.MISSING,
                details="Governorate value not provided",
                field1_name="nid",
                field1_value=nid_value,
                field2_name="governorate",
                field2_value=None,
            )
        
        # Use NID validator to compare
        comparison = self.nid_validator.compare_governorate(nid_value, governorate_value)
        
        status_map = {
            'consistent': ConsistencyStatus.CONSISTENT,
            'conflict': ConsistencyStatus.CONFLICT,
            'unknown': ConsistencyStatus.UNKNOWN,
        }
        
        status = status_map.get(comparison.get('status', 'unknown'), ConsistencyStatus.UNKNOWN)
        
        return ConsistencyCheck(
            check_name="nid_governorate",
            status=status,
            details=comparison.get('difference', comparison.get('reason', 'Governorates match')),
            field1_name="nid_governorate",
            field1_value=comparison.get('nid_governorate_name'),
            field2_name="ocr_governorate",
            field2_value=comparison.get('ocr_governorate'),
            confidence=1.0 if status == ConsistencyStatus.CONSISTENT else 0.5,
            evidence=comparison,
        )
    
    def _calculate_summary(self, result: ConsistencyResult) -> None:
        """Calculate summary statistics from all checks."""
        result.conflict_count = sum(
            1 for check in result.all_checks
            if check.status == ConsistencyStatus.CONFLICT
        )
        
        result.consistent_count = sum(
            1 for check in result.all_checks
            if check.status == ConsistencyStatus.CONSISTENT
        )
        
        result.unknown_count = sum(
            1 for check in result.all_checks
            if check.status in (ConsistencyStatus.UNKNOWN, ConsistencyStatus.MISSING)
        )
        
        result.has_conflicts = result.conflict_count > 0
    
    def _determine_overall_status(self, result: ConsistencyResult) -> None:
        """Determine overall consistency status."""
        if result.has_conflicts:
            result.overall_status = ConsistencyStatus.CONFLICT
        elif result.consistent_count > 0 and result.unknown_count == 0:
            result.overall_status = ConsistencyStatus.CONSISTENT
        elif result.consistent_count > 0:
            result.overall_status = ConsistencyStatus.CONSISTENT
        else:
            result.overall_status = ConsistencyStatus.UNKNOWN
    
    def _generate_recommendations(self, result: ConsistencyResult) -> None:
        """Generate recommendations based on consistency results."""
        recommendations = []
        
        # Build summary
        summary_parts = []
        
        if result.consistent_count > 0:
            summary_parts.append(f"{result.consistent_count} checks consistent")
        
        if result.conflict_count > 0:
            summary_parts.append(f"{result.conflict_count} conflicts detected")
            
            # Specific conflict recommendations
            if result.nid_dob_status == ConsistencyStatus.CONFLICT:
                recommendations.append(
                    "DOB conflict: Verify birth date against NID-encoded date. "
                    "OCR may have misread digits."
                )
            
            if result.nid_gender_status == ConsistencyStatus.CONFLICT:
                recommendations.append(
                    "Gender conflict: Verify gender field. NID digit 11 determines gender "
                    "(odd=male, even=female). OCR may have misread Arabic text."
                )
            
            if result.nid_governorate_status == ConsistencyStatus.CONFLICT:
                recommendations.append(
                    "Governorate conflict: Verify governorate name. NID digits 9-10 encode "
                    "governorate code. OCR may have misread Arabic place name."
                )
        
        if result.unknown_count > 0:
            summary_parts.append(f"{result.unknown_count} checks inconclusive")
            recommendations.append(
                "Some consistency checks could not be completed due to missing or invalid data."
            )
        
        result.summary = "; ".join(summary_parts) if summary_parts else "No consistency checks completed"
        result.recommendations = recommendations
    
    def get_conflicting_fields(self, result: ConsistencyResult) -> List[str]:
        """
        Get list of field names involved in conflicts.
        
        Args:
            result: ConsistencyResult
            
        Returns:
            List of field names with conflicts
        """
        conflicting = set()
        
        for check in result.all_checks:
            if check.status == ConsistencyStatus.CONFLICT:
                conflicting.add(check.field1_name.split('_')[0])  # e.g., 'nid' from 'nid_dob'
                conflicting.add(check.field2_name.split('_')[0])  # e.g., 'ocr' from 'ocr_dob'
        
        return list(conflicting)
    
    def is_fully_consistent(self, result: ConsistencyResult) -> bool:
        """
        Check if all completed consistency checks passed.
        
        Args:
            result: ConsistencyResult
            
        Returns:
            True if no conflicts detected, False otherwise
        """
        return not result.has_conflicts
    
    def get_confidence_score(self, result: ConsistencyResult) -> float:
        """
        Calculate overall confidence score based on consistency results.
        
        Args:
            result: ConsistencyResult
            
        Returns:
            Confidence score 0.0-1.0
        """
        if not result.all_checks:
            return 0.0
        
        # Weight consistent checks positively
        # Weight conflicts negatively
        # Unknown checks are neutral
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for check in result.all_checks:
            weight = check.confidence
            
            if check.status == ConsistencyStatus.CONSISTENT:
                weighted_score += weight * 1.0
            elif check.status == ConsistencyStatus.CONFLICT:
                weighted_score += weight * 0.0  # Full penalty
            else:
                # Unknown/missing: partial credit
                weighted_score += weight * 0.5
            
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_score / total_weight


# Export public API
__all__ = [
    'ConsistencyEngine',
    'ConsistencyResult',
    'ConsistencyCheck',
    'ConsistencyStatus',
]
