"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Field Localizer Module

Implements dynamic field localization for Egyptian National ID cards:
- Level 1: Canonical card input
- Level 2: Field region proposals
- Level 3: Visual/text anchors
- Level 4: Content-aware refinement

Supports these target fields:
1. National ID Number (NID)
2. Full Arabic Name
3. Date of Birth
4. Gender
5. Governorate
6. Address
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time

from app.config import get_config, LocalizationConfig, QualityConfig
from app.schemas.models import (
    LocalizationResult,
    FieldStatus,
    BoundingBox,
    RectificationResult,
    RectificationStatus,
)


class FieldLocalizer:
    """
    Egyptian National ID Field Localizer.
    
    Dynamically localizes each target field in the canonical card image.
    Does NOT use fixed pixel coordinates - re-localizes for every image.
    """
    
    # Target field names
    FIELD_NID = "nid"
    FIELD_NAME = "name"
    FIELD_DOB = "dob"
    FIELD_GENDER = "gender"
    FIELD_GOVERNORATE = "governorate"
    FIELD_ADDRESS = "address"
    
    ALL_FIELDS = [FIELD_NID, FIELD_NAME, FIELD_DOB, FIELD_GENDER, 
                  FIELD_GOVERNORATE, FIELD_ADDRESS]
    
    def __init__(self, config: Optional[LocalizationConfig] = None):
        """
        Initialize the field localizer.
        
        Args:
            config: Localization configuration (uses global config if not provided)
        """
        self.config = config or get_config().localization
        self.quality_config = get_config().quality
    
    def localize_all_fields(
        self,
        canonical_image: np.ndarray,
        rectification_result: RectificationResult
    ) -> Dict[str, LocalizationResult]:
        """
        Localize all target fields in the canonical card image.
        
        Args:
            canonical_image: Rectified canonical card image
            rectification_result: Result from rectification stage
            
        Returns:
            Dictionary mapping field names to LocalizationResult
        """
        start_time = time.time()
        
        results = {}
        
        for field_name in self.ALL_FIELDS:
            result = self._localize_field(field_name, canonical_image, rectification_result)
            results[field_name] = result
        
        return results
    
    def _localize_field(
        self,
        field_name: str,
        canonical_image: np.ndarray,
        rectification_result: RectificationResult
    ) -> LocalizationResult:
        """
        Localize a single field using field-specific logic.
        
        Args:
            field_name: Name of the field to localize
            canonical_image: Canonical card image
            rectification_result: Rectification result
            
        Returns:
            LocalizationResult for the field
        """
        if canonical_image is None or canonical_image.size == 0:
            return LocalizationResult(
                field_name=field_name,
                status=FieldStatus.FIELD_LOCALIZATION_FAILED,
                failure_reason="Empty canonical image",
            )
        
        # Get initial region proposal based on normalized layout
        initial_bbox = self._get_initial_proposal(field_name, canonical_image)
        
        if initial_bbox is None:
            return LocalizationResult(
                field_name=field_name,
                status=FieldStatus.FIELD_LOCALIZATION_FAILED,
                failure_reason=f"No initial proposal for field {field_name}",
            )
        
        # Apply field-specific localization strategy
        if field_name == self.FIELD_NID:
            return self._localize_nid(canonical_image, initial_bbox)
        elif field_name == self.FIELD_NAME:
            return self._localize_name(canonical_image, initial_bbox)
        elif field_name == self.FIELD_DOB:
            return self._localize_dob(canonical_image, initial_bbox)
        elif field_name == self.FIELD_GENDER:
            return self._localize_gender(canonical_image, initial_bbox)
        elif field_name == self.FIELD_GOVERNORATE:
            return self._localize_governorate(canonical_image, initial_bbox)
        elif field_name == self.FIELD_ADDRESS:
            return self._localize_address(canonical_image, initial_bbox)
        else:
            return LocalizationResult(
                field_name=field_name,
                status=FieldStatus.FIELD_LOCALIZATION_FAILED,
                failure_reason=f"Unknown field: {field_name}",
            )
    
    def _get_initial_proposal(
        self,
        field_name: str,
        canonical_image: np.ndarray
    ) -> Optional[BoundingBox]:
        """
        Get initial bounding box proposal based on normalized Egyptian ID layout.
        
        These are INITIAL PROPOSALS only, not final bboxes.
        The final bbox will be refined through visual analysis.
        """
        height, width = canonical_image.shape[:2]
        
        # Get normalized region from config
        if field_name == self.FIELD_NID:
            norm_region = self.config.nid_region
        elif field_name == self.FIELD_NAME:
            norm_region = self.config.name_region
        elif field_name == self.FIELD_DOB:
            norm_region = self.config.dob_region
        elif field_name == self.FIELD_GENDER:
            norm_region = self.config.gender_region
        elif field_name == self.FIELD_GOVERNORATE:
            norm_region = self.config.governorate_region
        elif field_name == self.FIELD_ADDRESS:
            norm_region = self.config.address_region
        else:
            return None
        
        x1_norm, y1_norm, x2_norm, y2_norm = norm_region
        
        # Convert to pixel coordinates
        x1 = int(x1_norm * width)
        y1 = int(y1_norm * height)
        x2 = int(x2_norm * width)
        y2 = int(y2_norm * height)
        
        # Validate bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    
    def _localize_nid(
        self,
        canonical_image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> LocalizationResult:
        """
        Localize National ID Number field.
        
        Expected characteristics:
        - Digit-heavy region
        - Approximately horizontal numeric text
        - 14-digit candidate structure
        - High digit density
        """
        height, width = canonical_image.shape[:2]
        
        # Start with initial proposal
        current_bbox = initial_bbox
        refinement_steps = 0
        
        # Extract region for analysis
        region = self._extract_region(canonical_image, current_bbox)
        
        # Analyze content density
        content_density = self._calculate_content_density(region)
        
        # NID should have high digit-like content
        expected_digit_density = self.config.nid_min_digit_density
        
        # Content-aware refinement
        best_bbox, best_confidence = self._refine_bbox_for_digits(
            canonical_image, current_bbox, expected_digit_density
        )
        
        if best_bbox is None:
            best_bbox = current_bbox
            best_confidence = 0.5
        
        # Calculate localization confidence
        localization_confidence = self._calculate_nid_confidence(
            canonical_image, best_bbox, content_density
        )
        
        # Generate alternatives if confidence is low
        alternatives = []
        if localization_confidence < self.config.low_confidence_threshold:
            alternatives = self._generate_alternatives(
                canonical_image, best_bbox, self.FIELD_NID
            )
        
        status = FieldStatus.LOCALIZED if localization_confidence >= self.config.min_localization_confidence else FieldStatus.FIELD_LOCALIZATION_UNCERTAIN
        
        return LocalizationResult(
            field_name=self.FIELD_NID,
            bbox=best_bbox,
            localization_confidence=localization_confidence,
            status=status,
            content_density=content_density,
            expected_character_density=expected_digit_density,
            aspect_ratio_match=self._calculate_aspect_ratio_match(best_bbox, 4.0),  # NID is typically wide
            clipping_score=self._calculate_clipping_score(canonical_image, best_bbox),
            refinement_steps=refinement_steps,
            refinement_method="digit_density_optimization",
            alternatives=alternatives,
        )
    
    def _localize_name(
        self,
        canonical_image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> LocalizationResult:
        """
        Localize Full Arabic Name field.
        
        Expected characteristics:
        - Arabic text
        - Arabic character density
        - Name-line structure
        - Relatively long text region
        """
        height, width = canonical_image.shape[:2]
        
        current_bbox = initial_bbox
        refinement_steps = 0
        
        region = self._extract_region(canonical_image, current_bbox)
        content_density = self._calculate_content_density(region)
        
        # Name should have moderate-to-high Arabic text density
        expected_arabic_density = self.config.name_min_arabic_density
        
        # Refine based on text line detection
        best_bbox, best_confidence = self._refine_bbox_for_text_lines(
            canonical_image, current_bbox
        )
        
        if best_bbox is None:
            best_bbox = current_bbox
            best_confidence = 0.5
        
        localization_confidence = self._calculate_name_confidence(
            canonical_image, best_bbox, content_density
        )
        
        alternatives = []
        if localization_confidence < self.config.low_confidence_threshold:
            alternatives = self._generate_alternatives(
                canonical_image, best_bbox, self.FIELD_NAME
            )
        
        status = FieldStatus.LOCALIZED if localization_confidence >= self.config.min_localization_confidence else FieldStatus.FIELD_LOCALIZATION_UNCERTAIN
        
        return LocalizationResult(
            field_name=self.FIELD_NAME,
            bbox=best_bbox,
            localization_confidence=localization_confidence,
            status=status,
            content_density=content_density,
            expected_character_density=expected_arabic_density,
            aspect_ratio_match=self._calculate_aspect_ratio_match(best_bbox, 3.0),
            clipping_score=self._calculate_clipping_score(canonical_image, best_bbox),
            refinement_steps=refinement_steps,
            refinement_method="text_line_detection",
            alternatives=alternatives,
        )
    
    def _localize_dob(
        self,
        canonical_image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> LocalizationResult:
        """
        Localize Date of Birth field.
        
        Expected characteristics:
        - Digit-heavy
        - Date-like structure
        - Short text region
        """
        height, width = canonical_image.shape[:2]
        
        current_bbox = initial_bbox
        refinement_steps = 0
        
        region = self._extract_region(canonical_image, current_bbox)
        content_density = self._calculate_content_density(region)
        
        # DOB should be relatively short and digit-heavy
        expected_width_ratio = self.config.dob_expected_width_ratio
        
        # Refine based on digit patterns
        best_bbox, best_confidence = self._refine_bbox_for_short_digits(
            canonical_image, current_bbox
        )
        
        if best_bbox is None:
            best_bbox = current_bbox
            best_confidence = 0.5
        
        localization_confidence = self._calculate_dob_confidence(
            canonical_image, best_bbox, content_density
        )
        
        alternatives = []
        if localization_confidence < self.config.low_confidence_threshold:
            alternatives = self._generate_alternatives(
                canonical_image, best_bbox, self.FIELD_DOB
            )
        
        status = FieldStatus.LOCALIZED if localization_confidence >= self.config.min_localization_confidence else FieldStatus.FIELD_LOCALIZATION_UNCERTAIN
        
        return LocalizationResult(
            field_name=self.FIELD_DOB,
            bbox=best_bbox,
            localization_confidence=localization_confidence,
            status=status,
            content_density=content_density,
            expected_character_density=0.6,
            aspect_ratio_match=self._calculate_aspect_ratio_match(best_bbox, 2.0),
            clipping_score=self._calculate_clipping_score(canonical_image, best_bbox),
            refinement_steps=refinement_steps,
            refinement_method="short_digit_pattern",
            alternatives=alternatives,
        )
    
    def _localize_gender(
        self,
        canonical_image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> LocalizationResult:
        """
        Localize Gender field.
        
        Expected characteristics:
        - Short categorical Arabic text
        - Small text region
        - Relationship to expected gender region
        """
        height, width = canonical_image.shape[:2]
        
        current_bbox = initial_bbox
        refinement_steps = 0
        
        region = self._extract_region(canonical_image, current_bbox)
        content_density = self._calculate_content_density(region)
        
        # Gender field is typically small
        max_width_ratio = self.config.gender_max_width_ratio
        
        # Refine based on compact text region
        best_bbox, best_confidence = self._refine_bbox_for_compact_text(
            canonical_image, current_bbox
        )
        
        if best_bbox is None:
            best_bbox = current_bbox
            best_confidence = 0.5
        
        localization_confidence = self._calculate_gender_confidence(
            canonical_image, best_bbox, content_density
        )
        
        alternatives = []
        if localization_confidence < self.config.low_confidence_threshold:
            alternatives = self._generate_alternatives(
                canonical_image, best_bbox, self.FIELD_GENDER
            )
        
        status = FieldStatus.LOCALIZED if localization_confidence >= self.config.min_localization_confidence else FieldStatus.FIELD_LOCALIZATION_UNCERTAIN
        
        return LocalizationResult(
            field_name=self.FIELD_GENDER,
            bbox=best_bbox,
            localization_confidence=localization_confidence,
            status=status,
            content_density=content_density,
            expected_character_density=0.4,
            aspect_ratio_match=self._calculate_aspect_ratio_match(best_bbox, 1.5),
            clipping_score=self._calculate_clipping_score(canonical_image, best_bbox),
            refinement_steps=refinement_steps,
            refinement_method="compact_text_region",
            alternatives=alternatives,
        )
    
    def _localize_governorate(
        self,
        canonical_image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> LocalizationResult:
        """
        Localize Governorate field.
        
        Expected characteristics:
        - Arabic text
        - Relatively short categorical/location text
        - Expected governorate region
        """
        height, width = canonical_image.shape[:2]
        
        current_bbox = initial_bbox
        refinement_steps = 0
        
        region = self._extract_region(canonical_image, current_bbox)
        content_density = self._calculate_content_density(region)
        
        # Refine based on text region
        best_bbox, best_confidence = self._refine_bbox_for_text_region(
            canonical_image, current_bbox
        )
        
        if best_bbox is None:
            best_bbox = current_bbox
            best_confidence = 0.5
        
        localization_confidence = self._calculate_governorate_confidence(
            canonical_image, best_bbox, content_density
        )
        
        alternatives = []
        if localization_confidence < self.config.low_confidence_threshold:
            alternatives = self._generate_alternatives(
                canonical_image, best_bbox, self.FIELD_GOVERNORATE
            )
        
        status = FieldStatus.LOCALIZED if localization_confidence >= self.config.min_localization_confidence else FieldStatus.FIELD_LOCALIZATION_UNCERTAIN
        
        return LocalizationResult(
            field_name=self.FIELD_GOVERNORATE,
            bbox=best_bbox,
            localization_confidence=localization_confidence,
            status=status,
            content_density=content_density,
            expected_character_density=0.5,
            aspect_ratio_match=self._calculate_aspect_ratio_match(best_bbox, 2.0),
            clipping_score=self._calculate_clipping_score(canonical_image, best_bbox),
            refinement_steps=refinement_steps,
            refinement_method="text_region_analysis",
            alternatives=alternatives,
        )
    
    def _localize_address(
        self,
        canonical_image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> LocalizationResult:
        """
        Localize Address field.
        
        Expected characteristics:
        - Larger Arabic text block
        - Multiple words
        - Possibly multiple lines
        - Mixed Arabic and digits may appear
        """
        height, width = canonical_image.shape[:2]
        
        current_bbox = initial_bbox
        refinement_steps = 0
        
        region = self._extract_region(canonical_image, current_bbox)
        content_density = self._calculate_content_density(region)
        
        # Address is typically larger than other fields
        best_bbox, best_confidence = self._refine_bbox_for_large_text_block(
            canonical_image, current_bbox
        )
        
        if best_bbox is None:
            best_bbox = current_bbox
            best_confidence = 0.5
        
        localization_confidence = self._calculate_address_confidence(
            canonical_image, best_bbox, content_density
        )
        
        alternatives = []
        if localization_confidence < self.config.low_confidence_threshold:
            alternatives = self._generate_alternatives(
                canonical_image, best_bbox, self.FIELD_ADDRESS
            )
        
        status = FieldStatus.LOCALIZED if localization_confidence >= self.config.min_localization_confidence else FieldStatus.FIELD_LOCALIZATION_UNCERTAIN
        
        return LocalizationResult(
            field_name=self.FIELD_ADDRESS,
            bbox=best_bbox,
            localization_confidence=localization_confidence,
            status=status,
            content_density=content_density,
            expected_character_density=0.6,
            aspect_ratio_match=self._calculate_aspect_ratio_match(best_bbox, 3.5),
            clipping_score=self._calculate_clipping_score(canonical_image, best_bbox),
            refinement_steps=refinement_steps,
            refinement_method="large_text_block_analysis",
            alternatives=alternatives,
        )
    
    def _extract_region(
        self,
        image: np.ndarray,
        bbox: BoundingBox
    ) -> np.ndarray:
        """Extract region from image based on bounding box."""
        return image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    
    def _calculate_content_density(self, region: np.ndarray) -> float:
        """
        Calculate content density in a region.
        
        Higher density indicates more text/content.
        """
        if region is None or region.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Threshold to get foreground content
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate ratio of non-background pixels
        total_pixels = binary.size
        foreground_pixels = cv2.countNonZero(binary)
        
        return foreground_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def _refine_bbox_for_digits(
        self,
        image: np.ndarray,
        initial_bbox: BoundingBox,
        expected_density: float
    ) -> Tuple[Optional[BoundingBox], float]:
        """Refine bbox to optimize for digit-heavy content."""
        return self._generic_refinement(image, initial_bbox, "digits")
    
    def _refine_bbox_for_text_lines(
        self,
        image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> Tuple[Optional[BoundingBox], float]:
        """Refine bbox based on text line detection."""
        return self._generic_refinement(image, initial_bbox, "text_lines")
    
    def _refine_bbox_for_short_digits(
        self,
        image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> Tuple[Optional[BoundingBox], float]:
        """Refine bbox for short digit sequences."""
        return self._generic_refinement(image, initial_bbox, "short_digits")
    
    def _refine_bbox_for_compact_text(
        self,
        image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> Tuple[Optional[BoundingBox], float]:
        """Refine bbox for compact text regions."""
        return self._generic_refinement(image, initial_bbox, "compact")
    
    def _refine_bbox_for_text_region(
        self,
        image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> Tuple[Optional[BoundingBox], float]:
        """Refine bbox for general text regions."""
        return self._generic_refinement(image, initial_bbox, "text")
    
    def _refine_bbox_for_large_text_block(
        self,
        image: np.ndarray,
        initial_bbox: BoundingBox
    ) -> Tuple[Optional[BoundingBox], float]:
        """Refine bbox for large text blocks."""
        return self._generic_refinement(image, initial_bbox, "large_block")
    
    def _generic_refinement(
        self,
        image: np.ndarray,
        initial_bbox: BoundingBox,
        refinement_type: str
    ) -> Tuple[Optional[BoundingBox], float]:
        """
        Generic bbox refinement with expansion/contraction.
        
        Tries multiple variations and selects the best one.
        """
        height, width = image.shape[:2]
        best_bbox = initial_bbox
        best_score = 0.0
        
        # Try original
        score = self._score_bbox(image, initial_bbox, refinement_type)
        if score > best_score:
            best_score = score
        
        # Try expansions
        for step in range(1, self.config.max_expansion_steps + 1):
            expanded = self._expand_bbox(initial_bbox, step * self.config.expansion_step, width, height)
            if expanded and expanded.validate(width, height):
                score = self._score_bbox(image, expanded, refinement_type)
                if score > best_score:
                    best_score = score
                    best_bbox = expanded
        
        # Try contractions
        for step in range(1, self.config.max_contraction_steps + 1):
            contracted = self._contract_bbox(initial_bbox, step * self.config.contraction_step, width, height)
            if contracted and contracted.validate(width, height):
                score = self._score_bbox(image, contracted, refinement_type)
                if score > best_score:
                    best_score = score
                    best_bbox = contracted
        
        return best_bbox, best_score
    
    def _expand_bbox(
        self,
        bbox: BoundingBox,
        pixels: int,
        max_width: int,
        max_height: int
    ) -> Optional[BoundingBox]:
        """Expand bounding box by given pixels."""
        x1 = max(0, bbox.x1 - pixels)
        y1 = max(0, bbox.y1 - pixels)
        x2 = min(max_width, bbox.x2 + pixels)
        y2 = min(max_height, bbox.y2 + pixels)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    
    def _contract_bbox(
        self,
        bbox: BoundingBox,
        pixels: int,
        max_width: int,
        max_height: int
    ) -> Optional[BoundingBox]:
        """Contract bounding box by given pixels."""
        x1 = min(bbox.x1 + pixels, max_width)
        y1 = min(bbox.y1 + pixels, max_height)
        x2 = max(0, bbox.x2 - pixels)
        y2 = max(0, bbox.y2 - pixels)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    
    def _score_bbox(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        refinement_type: str
    ) -> float:
        """Score a bounding box based on refinement type."""
        region = self._extract_region(image, bbox)
        
        if region.size == 0:
            return 0.0
        
        content_density = self._calculate_content_density(region)
        
        # Different scoring for different types
        if refinement_type == "digits":
            # Prefer moderate-to-high density with good aspect ratio
            ar_score = 1.0 - abs(bbox.aspect_ratio - 4.0) / 4.0
            return content_density * max(0, ar_score)
        
        elif refinement_type == "text_lines":
            # Prefer regions with horizontal structure
            return content_density * (1.0 + 0.2 * min(bbox.width / bbox.height, 5.0))
        
        elif refinement_type == "short_digits":
            # Prefer compact, dense regions
            size_penalty = min(1.0, 100.0 / bbox.area) if bbox.area > 0 else 0
            return content_density * size_penalty
        
        elif refinement_type == "compact":
            # Prefer small, dense regions
            return content_density * min(1.0, 5000.0 / bbox.area) if bbox.area > 0 else 0
        
        elif refinement_type == "text":
            return content_density
        
        elif refinement_type == "large_block":
            # Prefer larger regions with good content
            size_bonus = min(1.0, bbox.area / 20000.0)
            return content_density * (0.5 + 0.5 * size_bonus)
        
        return content_density
    
    def _calculate_nid_confidence(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        content_density: float
    ) -> float:
        """Calculate localization confidence for NID field."""
        base_confidence = min(1.0, content_density * 1.5)
        
        # Adjust for aspect ratio (NID should be wide)
        ar_factor = 1.0 - min(abs(bbox.aspect_ratio - 4.0) / 4.0, 0.3)
        
        # Adjust for position (should be in upper portion)
        height = image.shape[0]
        vertical_position = (bbox.y1 + bbox.y2) / (2 * height)
        position_factor = 1.0 if vertical_position < 0.4 else 0.8
        
        return base_confidence * ar_factor * position_factor
    
    def _calculate_name_confidence(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        content_density: float
    ) -> float:
        """Calculate localization confidence for Name field."""
        base_confidence = min(1.0, content_density * 1.3)
        
        # Names should have reasonable width
        ar_factor = 1.0 - min(abs(bbox.aspect_ratio - 3.0) / 3.0, 0.3)
        
        return base_confidence * ar_factor
    
    def _calculate_dob_confidence(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        content_density: float
    ) -> float:
        """Calculate localization confidence for DOB field."""
        base_confidence = min(1.0, content_density * 1.4)
        
        # DOB should be compact
        size_factor = min(1.0, 8000.0 / bbox.area) if bbox.area > 0 else 0
        
        return base_confidence * size_factor
    
    def _calculate_gender_confidence(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        content_density: float
    ) -> float:
        """Calculate localization confidence for Gender field."""
        base_confidence = min(1.0, content_density * 1.2)
        
        # Gender field is very compact
        size_factor = min(1.0, 5000.0 / bbox.area) if bbox.area > 0 else 0
        
        return base_confidence * size_factor
    
    def _calculate_governorate_confidence(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        content_density: float
    ) -> float:
        """Calculate localization confidence for Governorate field."""
        base_confidence = min(1.0, content_density * 1.3)
        return base_confidence
    
    def _calculate_address_confidence(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        content_density: float
    ) -> float:
        """Calculate localization confidence for Address field."""
        base_confidence = min(1.0, content_density * 1.2)
        
        # Address should be larger
        size_factor = min(1.0, bbox.area / 15000.0) if bbox.area > 0 else 0.5
        
        return base_confidence * (0.5 + 0.5 * size_factor)
    
    def _calculate_aspect_ratio_match(self, bbox: BoundingBox, expected_ar: float) -> float:
        """Calculate how well bbox aspect ratio matches expected."""
        if bbox.height == 0:
            return 0.0
        deviation = abs(bbox.aspect_ratio - expected_ar) / expected_ar
        return max(0, 1.0 - deviation)
    
    def _calculate_clipping_score(
        self,
        image: np.ndarray,
        bbox: BoundingBox
    ) -> float:
        """
        Calculate how much content appears to touch the bbox boundaries.
        
        Lower score is better (less clipping).
        """
        height, width = image.shape[:2]
        
        # Check if bbox touches image borders
        touches_border = (
            bbox.x1 <= 2 or
            bbox.y1 <= 2 or
            bbox.x2 >= width - 2 or
            bbox.y2 >= height - 2
        )
        
        if touches_border:
            return 0.7  # High clipping score
        
        # Check content near edges of bbox
        margin = 3
        region = self._extract_region(image, bbox)
        
        if region.size == 0:
            return 0.0
        
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Check edge regions
        top_edge = gray[:margin, :].mean()
        bottom_edge = gray[-margin:, :].mean()
        left_edge = gray[:, :margin].mean()
        right_edge = gray[:, -margin:].mean()
        
        avg_edge_intensity = (top_edge + bottom_edge + left_edge + right_edge) / 4
        
        # If edges are very bright/dark, content might be clipped
        if avg_edge_intensity > 240 or avg_edge_intensity < 15:
            return 0.5
        
        return 0.2  # Low clipping score
    
    def _generate_alternatives(
        self,
        image: np.ndarray,
        primary_bbox: BoundingBox,
        field_name: str
    ) -> List[LocalizationResult]:
        """Generate alternative bounding box candidates."""
        alternatives = []
        height, width = image.shape[:2]
        
        # Generate limited number of alternatives
        shifts = [
            (0, -self.config.expansion_step),  # Up
            (0, self.config.expansion_step),   # Down
            (-self.config.expansion_step, 0),  # Left
            (self.config.expansion_step, 0),   # Right
        ]
        
        for dx, dy in shifts:
            if len(alternatives) >= self.config.max_alternatives:
                break
            
            new_x1 = max(0, primary_bbox.x1 + dx)
            new_y1 = max(0, primary_bbox.y1 + dy)
            new_x2 = min(width, primary_bbox.x2 + dx)
            new_y2 = min(height, primary_bbox.y2 + dy)
            
            if new_x1 < new_x2 and new_y1 < new_y2:
                alt_bbox = BoundingBox(x1=new_x1, y1=new_y1, x2=new_x2, y2=new_y2)
                
                # Calculate confidence for alternative
                region = self._extract_region(image, alt_bbox)
                content_density = self._calculate_content_density(region)
                
                alt = LocalizationResult(
                    field_name=field_name,
                    bbox=alt_bbox,
                    localization_confidence=content_density * 0.8,  # Slightly lower than primary
                    status=FieldStatus.LOCALIZED,
                    content_density=content_density,
                )
                alternatives.append(alt)
        
        return alternatives
