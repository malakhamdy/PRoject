"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Data Models and Schemas

Strongly typed schemas for all pipeline stages.
Distinguishes between localization confidence, OCR confidence, and validation confidence.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import time


class FieldStatus(Enum):
    """Status of a field in the pipeline."""
    NOT_PROCESSED = "not_processed"
    LOCALIZED = "localized"
    FIELD_LOCALIZATION_UNCERTAIN = "field_localization_uncertain"
    FIELD_LOCALIZATION_FAILED = "field_localization_failed"
    OCR_PROCESSED = "ocr_processed"
    OCR_FAILED = "ocr_failed"
    EXTRACTED = "extracted"
    NORMALIZED = "normalized"
    VALIDATED = "validated"
    INVALID = "invalid"


class CardDetectionStatus(Enum):
    """Card detection status."""
    DETECTED = "detected"
    CARD_NOT_DETECTED = "card_not_detected"
    CARD_GEOMETRY_INVALID = "card_geometry_invalid"
    CARD_TOO_SMALL = "card_too_small"
    CARD_TOO_CROPPED = "card_too_cropped"
    CARD_BORDER_UNCERTAIN = "card_border_uncertain"
    FULL_FRAME_CARD_FALLBACK = "full_frame_card_fallback"


class RectificationStatus(Enum):
    """Rectification status."""
    SUCCESS = "success"
    RECTIFICATION_FAILED = "rectification_failed"
    RECTIFICATION_LOW_CONFIDENCE = "rectification_low_confidence"


@dataclass
class BoundingBox:
    """
    Represents a bounding box with coordinates.
    
    Coordinates are in the format [x1, y1, x2, y2] where:
    - x1, y1: top-left corner
    - x2, y2: bottom-right corner
    
    Can be relative to original image or canonical card.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    
    def validate(self, max_x: int, max_y: int) -> bool:
        """Validate the bounding box is within bounds and well-formed."""
        if self.x1 >= self.x2:
            return False
        if self.y1 >= self.y2:
            return False
        if self.x1 < 0 or self.y1 < 0:
            return False
        if self.x2 > max_x or self.y2 > max_y:
            return False
        return True
    
    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height)."""
        if self.height == 0:
            return 0.0
        return self.width / self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_list(self) -> List[int]:
        """Convert to list format."""
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, coords: List[int]) -> "BoundingBox":
        """Create from list format."""
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])


@dataclass
class Corner:
    """Represents a single corner point."""
    x: float
    y: float
    quality: float = 1.0  # Corner detection quality score


@dataclass
class CardCorners:
    """
    Four ordered corners of the detected card.
    
    Order: TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT
    """
    top_left: Corner
    top_right: Corner
    bottom_right: Corner
    bottom_left: Corner
    
    def to_list(self) -> List[List[float]]:
        """Convert corners to list format for perspective transform."""
        return [
            [self.top_left.x, self.top_left.y],
            [self.top_right.x, self.top_right.y],
            [self.bottom_right.x, self.bottom_right.y],
            [self.bottom_left.x, self.bottom_left.y],
        ]
    
    def validate_convexity(self) -> bool:
        """Validate that corners form a convex quadrilateral."""
        # Cross product method for convexity check
        points = [
            (self.top_left.x, self.top_left.y),
            (self.top_right.x, self.top_right.y),
            (self.bottom_right.x, self.bottom_right.y),
            (self.bottom_left.x, self.bottom_left.y),
        ]
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        signs = []
        for i in range(4):
            o = points[i]
            a = points[(i + 1) % 4]
            b = points[(i + 2) % 4]
            signs.append(cross(o, a, b))
        
        # All cross products should have the same sign for convex polygon
        return all(s > 0 for s in signs) or all(s < 0 for s in signs)
    
    def validate_no_intersection(self) -> bool:
        """Validate that edges do not self-intersect."""
        # For a properly ordered quadrilateral, this should always be true
        # if convexity is valid
        return self.validate_convexity()


@dataclass
class CardDetectionResult:
    """Result of card detection stage."""
    status: CardDetectionStatus
    bbox: Optional[BoundingBox] = None
    corners: Optional[CardCorners] = None
    confidence: float = 0.0
    failure_reason: Optional[str] = None
    
    # Detection metrics
    candidate_count: int = 0
    best_candidate_score: float = 0.0
    
    # Timing
    detection_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "status": self.status.value,
            "confidence": self.confidence,
            "detection_time_ms": self.detection_time_ms,
            "candidate_count": self.candidate_count,
        }
        
        if self.bbox:
            result["bbox"] = self.bbox.to_list()
        
        if self.corners:
            result["corners"] = self.corners.to_list()
        
        if self.failure_reason:
            result["failure_reason"] = self.failure_reason
        
        return result


@dataclass
class RectificationResult:
    """Result of card rectification stage."""
    status: RectificationStatus
    canonical_image: Optional[Any] = None  # numpy array
    transformation_matrix: Optional[Any] = None  # 3x3 matrix
    rectification_confidence: float = 0.0
    
    # Geometry quality metrics
    geometry_validity: float = 0.0
    corner_quality: float = 0.0
    area_coverage: float = 0.0
    perspective_quality: float = 0.0
    
    # Dimensions
    output_width: int = 0
    output_height: int = 0
    
    # Timing
    rectification_time_ms: float = 0.0
    
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "status": self.status.value,
            "rectification_confidence": self.rectification_confidence,
            "geometry_validity": self.geometry_validity,
            "corner_quality": self.corner_quality,
            "area_coverage": self.area_coverage,
            "perspective_quality": self.perspective_quality,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "rectification_time_ms": self.rectification_time_ms,
        }
        
        if self.failure_reason:
            result["failure_reason"] = self.failure_reason
        
        return result


@dataclass
class OCRCandidate:
    """
    OCR candidate result for a field.
    
    Contains raw OCR output before normalization/validation.
    """
    text: str
    confidence: float
    language: str = "ara"  # Arabic by default for Egyptian ID
    
    # Alternative readings
    alternatives: List[str] = field(default_factory=list)
    
    # Character-level details (if available)
    char_confidences: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "alternatives": self.alternatives,
            "char_confidences": self.char_confidences,
        }


@dataclass
class LocalizationResult:
    """
    Result of field localization for a single field.
    
    Contains bounding box and localization confidence.
    OCR results are stored separately.
    """
    field_name: str
    bbox: Optional[BoundingBox] = None
    localization_confidence: float = 0.0
    status: FieldStatus = FieldStatus.NOT_PROCESSED
    
    # Alternative bounding box candidates
    alternatives: List["LocalizationResult"] = field(default_factory=list)
    
    # Content analysis metrics
    content_density: float = 0.0
    expected_character_density: float = 0.0
    aspect_ratio_match: float = 0.0
    clipping_score: float = 0.0  # How much content touches borders
    
    # Refinement history
    refinement_steps: int = 0
    refinement_method: Optional[str] = None
    
    # Failure information
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "field_name": self.field_name,
            "status": self.status.value,
            "localization_confidence": self.localization_confidence,
            "content_density": self.content_density,
            "aspect_ratio_match": self.aspect_ratio_match,
            "refinement_steps": self.refinement_steps,
        }
        
        if self.bbox:
            result["bbox"] = self.bbox.to_list()
        
        if self.alternatives:
            result["alternatives"] = [alt.to_dict() for alt in self.alternatives]
        
        if self.failure_reason:
            result["failure_reason"] = self.failure_reason
        
        return result


@dataclass
class FieldResult:
    """
    Complete result for a single field.
    
    Combines localization, OCR, extraction, normalization, and validation.
    """
    field_name: str
    
    # Localization
    bbox: Optional[BoundingBox] = None
    localization_confidence: float = 0.0
    
    # OCR
    ocr_result: Optional[OCRCandidate] = None
    ocr_confidence: float = 0.0
    
    # Extracted value
    value: Optional[str] = None
    
    # Normalized value
    normalized_value: Optional[str] = None
    
    # Validation
    validation_status: FieldStatus = FieldStatus.NOT_PROCESSED
    validation_confidence: float = 0.0
    validation_errors: List[str] = field(default_factory=list)
    
    # Source tracking
    source: str = "localization"  # localization, ocr, extraction, etc.
    
    # Alternatives
    alternatives: List[str] = field(default_factory=list)
    
    # Failure reason
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "field_name": self.field_name,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "validation_status": self.validation_status.value,
            "validation_confidence": self.validation_confidence,
            "source": self.source,
        }
        
        if self.bbox:
            result["bbox"] = self.bbox.to_list()
            result["localization_confidence"] = self.localization_confidence
        
        if self.ocr_result:
            result["ocr_confidence"] = self.ocr_result.confidence
            result["ocr_text"] = self.ocr_result.text
        
        if self.validation_errors:
            result["validation_errors"] = self.validation_errors
        
        if self.alternatives:
            result["alternatives"] = self.alternatives
        
        if self.failure_reason:
            result["failure_reason"] = self.failure_reason
        
        return result


@dataclass
class ConsistencyResult:
    """
    Result of cross-field consistency checking.
    
    Validates relationships between fields (e.g., NID contains governorate code).
    """
    is_consistent: bool = True
    confidence: float = 1.0
    
    # Individual checks
    nid_governorate_match: Optional[bool] = None
    nid_dob_consistency: Optional[bool] = None
    gender_format_valid: Optional[bool] = None
    
    # Issues found
    inconsistencies: List[str] = field(default_factory=list)
    
    # Warnings (non-blocking issues)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_consistent": self.is_consistent,
            "confidence": self.confidence,
            "nid_governorate_match": self.nid_governorate_match,
            "nid_dob_consistency": self.nid_dob_consistency,
            "gender_format_valid": self.gender_format_valid,
            "inconsistencies": self.inconsistencies,
            "warnings": self.warnings,
        }


@dataclass
class PerformanceMetrics:
    """Timing and performance metrics for the pipeline."""
    # Stage timings (milliseconds)
    quality_time_ms: float = 0.0
    card_detection_time_ms: float = 0.0
    rectification_time_ms: float = 0.0
    localization_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    extraction_time_ms: float = 0.0
    normalization_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Memory usage (optional)
    peak_memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "quality_time_ms": self.quality_time_ms,
            "card_detection_time_ms": self.card_detection_time_ms,
            "rectification_time_ms": self.rectification_time_ms,
            "localization_time_ms": self.localization_time_ms,
            "ocr_time_ms": self.ocr_time_ms,
            "extraction_time_ms": self.extraction_time_ms,
            "normalization_time_ms": self.normalization_time_ms,
            "validation_time_ms": self.validation_time_ms,
            "total_time_ms": self.total_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
        }


@dataclass
class PipelineResult:
    """
    Complete result from the OCI pipeline.
    
    Aggregates results from all stages.
    """
    # Overall status
    success: bool = False
    status: str = "not_started"
    message: Optional[str] = None
    
    # Card detection
    card_detection: Optional[CardDetectionResult] = None
    
    # Rectification
    rectification: Optional[RectificationResult] = None
    
    # Field results (keyed by field name)
    fields: Dict[str, FieldResult] = field(default_factory=dict)
    
    # Consistency check
    consistency: Optional[ConsistencyResult] = None
    
    # Performance metrics
    metrics: Optional[PerformanceMetrics] = None
    
    # Original image info
    original_width: int = 0
    original_height: int = 0
    
    # Debug artifacts (only in debug mode)
    debug_artifacts: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "success": self.success,
            "status": self.status,
            "original_dimensions": {
                "width": self.original_width,
                "height": self.original_height,
            },
        }
        
        if self.message:
            result["message"] = self.message
        
        if self.card_detection:
            result["card"] = self.card_detection.to_dict()
        
        if self.rectification:
            result["rectification"] = self.rectification.to_dict()
        
        if self.fields:
            result["fields"] = {
                name: field.to_dict() 
                for name, field in self.fields.items()
            }
        
        if self.consistency:
            result["consistency"] = self.consistency.to_dict()
        
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
        
        if self.debug_artifacts:
            result["debug_artifacts"] = self.debug_artifacts
        
        return result


@dataclass
class ImageQualityResult:
    """Result of image quality assessment."""
    is_acceptable: bool = True
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    noise_level: float = 0.0
    
    # Quality issues
    issues: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_acceptable": self.is_acceptable,
            "sharpness": self.sharpness,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "noise_level": self.noise_level,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }
