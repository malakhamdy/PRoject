"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Card Rectifier Module

Implements perspective rectification:
- Orientation detection
- Four-point perspective transformation
- Canonical card generation
- Rectification confidence calculation
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import time

from app.config import get_config, RectificationConfig, CardConfig
from app.schemas.models import (
    RectificationResult,
    RectificationStatus,
    CardCorners,
    CardDetectionResult,
    CardDetectionStatus,
)
from app.utils.geometry import (
    apply_perspective_transform,
    calculate_corner_angles,
    calculate_quadrilateral_area,
    calculate_rectangularity,
)


class CardRectifier:
    """
    Egyptian National ID Card Rectifier.
    
    Performs perspective correction to transform the detected card
    into a canonical rectangular form.
    """
    
    def __init__(self, config: Optional[RectificationConfig] = None):
        """
        Initialize the card rectifier.
        
        Args:
            config: Rectification configuration (uses global config if not provided)
        """
        self.config = config or get_config().rectification
        self.card_config = get_config().card
    
    def rectify(
        self,
        image: np.ndarray,
        detection_result: CardDetectionResult
    ) -> RectificationResult:
        """
        Rectify the detected card to canonical form.
        
        Args:
            image: Original input image
            detection_result: Result from card detection
            
        Returns:
            RectificationResult with canonical image and metadata
        """
        start_time = time.time()
        
        # Validate input
        if image is None or image.size == 0:
            return RectificationResult(
                status=RectificationStatus.RECTIFICATION_FAILED,
                failure_reason="Empty or invalid image",
                rectification_time_ms=(time.time() - start_time) * 1000,
            )
        
        if detection_result.status not in [
            CardDetectionStatus.DETECTED,
            CardDetectionStatus.FULL_FRAME_CARD_FALLBACK,
        ]:
            return RectificationResult(
                status=RectificationStatus.RECTIFICATION_FAILED,
                failure_reason=f"Cannot rectify: card detection status is {detection_result.status.value}",
                rectification_time_ms=(time.time() - start_time) * 1000,
            )
        
        if detection_result.corners is None:
            return RectificationResult(
                status=RectificationStatus.RECTIFICATION_FAILED,
                failure_reason="No corners available for rectification",
                rectification_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Get canonical dimensions
        canonical_width = self.card_config.canonical_width
        canonical_height = self.card_config.canonical_height
        
        # Apply perspective transformation
        try:
            rectified_image, transform_matrix = apply_perspective_transform(
                image,
                detection_result.corners,
                canonical_width,
                canonical_height,
                self.config.interpolation_method,
            )
        except Exception as e:
            return RectificationResult(
                status=RectificationStatus.RECTIFICATION_FAILED,
                failure_reason=f"Perspective transform failed: {str(e)}",
                rectification_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Calculate rectification confidence
        rectification_confidence = self._calculate_rectification_confidence(
            detection_result,
            detection_result.corners,
            image.shape[1],
            image.shape[0],
            rectified_image,
        )
        
        # Determine status based on confidence
        if rectification_confidence < self.config.critical_confidence_threshold:
            status = RectificationStatus.RECTIFICATION_FAILED
            failure_reason = "Rectification confidence critically low"
        elif rectification_confidence < self.config.low_confidence_threshold:
            status = RectificationStatus.RECTIFICATION_LOW_CONFIDENCE
            failure_reason = "Rectification confidence below threshold"
        else:
            status = RectificationStatus.SUCCESS
            failure_reason = None
        
        rectification_time_ms = (time.time() - start_time) * 1000
        
        return RectificationResult(
            status=status,
            canonical_image=rectified_image,
            transformation_matrix=transform_matrix,
            rectification_confidence=rectification_confidence,
            geometry_validity=self._calculate_geometry_validity(detection_result.corners),
            corner_quality=self._calculate_corner_quality(detection_result.corners),
            area_coverage=self._calculate_area_coverage(
                detection_result.corners, image.shape[1], image.shape[0]
            ),
            perspective_quality=calculate_rectangularity(detection_result.corners),
            output_width=canonical_width,
            output_height=canonical_height,
            rectification_time_ms=rectification_time_ms,
            failure_reason=failure_reason,
        )
    
    def _calculate_rectification_confidence(
        self,
        detection_result: CardDetectionResult,
        corners: CardCorners,
        image_width: int,
        image_height: int,
        rectified_image: np.ndarray,
    ) -> float:
        """
        Calculate overall rectification confidence.
        
        Based on:
        - Card detection confidence
        - Corner quality
        - Geometry validity
        - Area coverage
        - Aspect ratio
        - Perspective plausibility
        - Border quality
        - Output quality
        """
        # Start with detection confidence
        confidence = detection_result.confidence
        
        # Adjust based on corner quality
        corner_quality = self._calculate_corner_quality(corners)
        confidence *= (0.7 + 0.3 * corner_quality)
        
        # Adjust based on geometry validity
        geometry_validity = self._calculate_geometry_validity(corners)
        confidence *= (0.7 + 0.3 * geometry_validity)
        
        # Adjust based on area coverage
        area_coverage = self._calculate_area_coverage(corners, image_width, image_height)
        if area_coverage < self.config.min_area_coverage:
            confidence *= 0.8
        
        # Check aspect ratio of rectified output
        expected_ar = self.card_config.expected_aspect_ratio
        actual_ar = rectified_image.shape[1] / rectified_image.shape[0] if rectified_image.shape[0] > 0 else 0
        ar_deviation = abs(actual_ar - expected_ar) / expected_ar
        if ar_deviation > 0.2:
            confidence *= 0.9
        
        # Check border quality (should have some content, not all black/white)
        border_quality = self._check_border_quality(rectified_image)
        confidence *= (0.8 + 0.2 * border_quality)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_corner_quality(self, corners: CardCorners) -> float:
        """Calculate quality score for corners."""
        quality = 1.0
        
        # Check angle deviation from 90 degrees
        angles = calculate_corner_angles(corners)
        for angle in angles:
            deviation = abs(angle - 90.0)
            if deviation > 30.0:
                quality -= 0.1
            if deviation > 45.0:
                quality -= 0.15
        
        # Check corner distances
        points = corners.to_list()
        for i in range(4):
            for j in range(i + 1, 4):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance < self.card_config.min_corner_distance:
                    quality -= 0.2
        
        return max(quality, 0.0)
    
    def _calculate_geometry_validity(self, corners: CardCorners) -> float:
        """Calculate geometry validity score."""
        validity = 1.0
        
        # Convexity check
        if not corners.validate_convexity():
            validity -= 0.5
        
        # Self-intersection check
        if not corners.validate_no_intersection():
            validity -= 0.5
        
        # Rectangularity
        rectangularity = calculate_rectangularity(corners)
        if rectangularity < 0.7:
            validity -= 0.2
        
        return max(validity, 0.0)
    
    def _calculate_area_coverage(
        self,
        corners: CardCorners,
        image_width: int,
        image_height: int
    ) -> float:
        """Calculate how much of the image the card covers."""
        image_area = image_width * image_height
        card_area = calculate_quadrilateral_area(corners)
        
        if image_area == 0:
            return 0.0
        
        return min(card_area / image_area, 1.0)
    
    def _check_border_quality(self, image: np.ndarray) -> float:
        """
        Check the quality of borders in the rectified image.
        
        Good rectification should not have extreme borders (all black or all white).
        """
        if image is None or image.size == 0:
            return 0.0
        
        height, width = image.shape[:2]
        border_size = max(5, int(min(width, height) * 0.05))
        
        # Extract border regions
        top_border = image[:border_size, :]
        bottom_border = image[-border_size:, :]
        left_border = image[:, :border_size]
        right_border = image[:, -border_size:]
        
        # Calculate variance in borders (should have some variation)
        borders = [top_border, bottom_border, left_border, right_border]
        variances = []
        
        for border in borders:
            if border.size > 0:
                variance = np.var(border)
                variances.append(variance)
        
        if len(variances) == 0:
            return 0.0
        
        avg_variance = np.mean(variances)
        
        # Normalize variance score (higher variance = better, up to a point)
        # Typical good images have variance > 1000
        quality = min(avg_variance / 5000.0, 1.0)
        
        return quality
