"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Card Detector Module

Implements hierarchical card detection:
- Level 1: Image preparation
- Level 2: Candidate generation
- Level 3: Candidate scoring
- Level 4: Candidate validation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time

from app.config import get_config, DetectionConfig, CardConfig
from app.schemas.models import (
    CardDetectionResult,
    CardDetectionStatus,
    BoundingBox,
    CardCorners,
    Corner,
)
from app.utils.geometry import (
    order_corners,
    validate_corners,
    calculate_rectangularity,
    calculate_convexity,
    calculate_aspect_ratio,
    normalize_coordinates_for_detection,
    scale_bbox,
)


class CardDetector:
    """
    Egyptian National ID Card Detector.
    
    Detects the card in an input image using computer vision techniques.
    Does NOT assume fixed pixel coordinates - detects dynamically for each image.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize the card detector.
        
        Args:
            config: Detection configuration (uses global config if not provided)
        """
        self.config = config or get_config().detection
        self.card_config = get_config().card
    
    def detect(self, image: np.ndarray) -> CardDetectionResult:
        """
        Detect the Egyptian National ID card in an image.
        
        Args:
            image: Input image (BGR format, original resolution)
            
        Returns:
            CardDetectionResult with detection status, bbox, corners, and confidence
        """
        start_time = time.time()
        
        # Validate input
        if image is None or image.size == 0:
            return CardDetectionResult(
                status=CardDetectionStatus.CARD_NOT_DETECTED,
                failure_reason="Empty or invalid image",
                detection_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Get image dimensions
        original_height, original_width = image.shape[:2]
        
        # Level 1: Image preparation
        prepared_image, scale_x, scale_y = self._prepare_image(image)
        
        if prepared_image is None:
            return CardDetectionResult(
                status=CardDetectionStatus.CARD_NOT_DETECTED,
                failure_reason="Failed to prepare image for detection",
                detection_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Level 2: Candidate generation
        candidates = self._generate_candidates(prepared_image)
        
        if len(candidates) == 0:
            # Try fallback: check if entire frame might be the card
            return self._try_full_frame_fallback(
                image, original_width, original_height, start_time
            )
        
        # Level 3: Candidate scoring
        scored_candidates = self._score_candidates(candidates, prepared_image)
        
        if len(scored_candidates) == 0:
            return self._try_full_frame_fallback(
                image, original_width, original_height, start_time
            )
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = scored_candidates[0]
        
        # Level 4: Candidate validation
        is_valid, validation_issues = self._validate_candidate(
            best_candidate, prepared_image.shape[1], prepared_image.shape[0]
        )
        
        if not is_valid:
            # Try next best candidate
            for candidate in scored_candidates[1:]:
                is_valid, validation_issues = self._validate_candidate(
                    candidate, prepared_image.shape[1], prepared_image.shape[0]
                )
                if is_valid:
                    best_candidate = candidate
                    break
        
        if not is_valid:
            # Best candidate failed validation
            status = CardDetectionStatus.CARD_GEOMETRY_INVALID
            if "area" in str(validation_issues).lower():
                status = CardDetectionStatus.CARD_TOO_SMALL
            
            return CardDetectionResult(
                status=status,
                confidence=best_candidate['score'],
                failure_reason=f"Validation failed: {validation_issues}",
                candidate_count=len(candidates),
                best_candidate_score=best_candidate['score'],
                detection_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Extract corners and scale back to original image
        scaled_corners = self._scale_corners_to_original(
            best_candidate['corners'], scale_x, scale_y
        )
        
        # Validate corners in original image space
        corners_valid, corner_issues = validate_corners(
            scaled_corners, original_width, original_height
        )
        
        if not corners_valid:
            return CardDetectionResult(
                status=CardDetectionStatus.CARD_GEOMETRY_INVALID,
                confidence=best_candidate['score'],
                failure_reason=f"Corner validation failed: {corner_issues}",
                candidate_count=len(candidates),
                best_candidate_score=best_candidate['score'],
                detection_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Create bounding box from corners
        bbox = self._corners_to_bbox(scaled_corners)
        
        # Calculate final confidence
        confidence = self._calculate_final_confidence(
            best_candidate, scaled_corners, original_width, original_height
        )
        
        detection_time_ms = (time.time() - start_time) * 1000
        
        return CardDetectionResult(
            status=CardDetectionStatus.DETECTED,
            bbox=bbox,
            corners=scaled_corners,
            confidence=confidence,
            candidate_count=len(candidates),
            best_candidate_score=best_candidate['score'],
            detection_time_ms=detection_time_ms,
        )
    
    def _prepare_image(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Level 1: Prepare image for detection.
        
        - Create detection-scale copy
        - Convert to grayscale
        - Apply controlled denoising
        - Enhance edges
        
        Returns:
            Tuple of (prepared_image, scale_x, scale_y)
        """
        original_height, original_width = image.shape[:2]
        
        # Calculate scale for detection
        scale_x, scale_y, det_width, det_height = normalize_coordinates_for_detection(
            original_width,
            original_height,
            get_config().image.detection_max_dim,
        )
        
        # Resize if needed
        if scale_x != 1.0 or scale_y != 1.0:
            detection_image = cv2.resize(
                image,
                (det_width, det_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detection_image = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(detection_image, cv2.COLOR_BGR2GRAY)
        
        # Apply mild denoising
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=10,  # Filter strength
            templateWindowSize=7,
            searchWindowSize=21,
        )
        
        # Apply adaptive thresholding for edge enhancement
        enhanced = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2,   # C value
        )
        
        return enhanced, scale_x, scale_y
    
    def _generate_candidates(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Level 2: Generate card candidates.
        
        Uses edge detection, morphology, contours, and quadrilateral approximation.
        
        Returns:
            List of candidate dictionaries with contour and preliminary info
        """
        candidates = []
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            image,
            self.config.canny_threshold1,
            self.config.canny_threshold2,
        )
        
        # Apply morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.config.kernel_size, self.config.kernel_size))
        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config.morphology_iterations,
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.config.min_contour_area:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for quadrilaterals (4 vertices)
            if len(approx) == 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                
                candidates.append({
                    'contour': contour,
                    'approx': approx,
                    'area': area,
                    'bbox': (x, y, w, h),
                })
            elif len(approx) > 4:
                # Try to find a quadrilateral within the contour
                # This handles cases where the card has rounded corners
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                box_area = cv2.contourArea(box)
                if box_area >= self.config.min_contour_area:
                    candidates.append({
                        'contour': contour,
                        'approx': box,
                        'area': box_area,
                        'bbox': cv2.boundingRect(box),
                        'min_area_rect': rect,
                    })
        
        return candidates
    
    def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Level 3: Score candidates based on multiple criteria.
        
        Scoring factors:
        - Area ratio
        - Aspect ratio
        - Rectangularity
        - Convexity
        - Edge strength
        
        Returns:
            List of candidates with added 'score' field
        """
        image_area = image.shape[0] * image.shape[1]
        expected_ar = self.card_config.expected_aspect_ratio
        ar_tolerance = self.card_config.aspect_ratio_tolerance
        
        scored = []
        
        for candidate in candidates:
            approx = candidate['approx']
            area = candidate['area']
            
            # Normalize corner points
            corners_array = approx.reshape(-1, 2)
            
            try:
                corners = order_corners(corners_array)
            except (ValueError, IndexError):
                continue
            
            # Calculate metrics
            area_ratio = area / image_area
            
            aspect_ratio = calculate_aspect_ratio(corners)
            ar_score = 1.0 - min(abs(aspect_ratio - expected_ar) / ar_tolerance, 1.0)
            
            rectangularity = calculate_rectangularity(corners)
            
            convexity = calculate_convexity(candidate['contour'])
            
            # Edge strength (average gradient magnitude in the region)
            edge_strength = self._calculate_edge_strength(
                image, corners, candidate['bbox']
            )
            
            # Calculate weighted score
            score = (
                self.config.weight_area * min(area_ratio * 10, 1.0) +
                self.config.weight_aspect_ratio * ar_score +
                self.config.weight_rectangularity * rectangularity +
                self.config.weight_convexity * convexity +
                self.config.weight_edge_strength * edge_strength
            )
            
            candidate['score'] = score
            candidate['corners'] = corners
            candidate['metrics'] = {
                'area_ratio': area_ratio,
                'aspect_ratio': aspect_ratio,
                'ar_score': ar_score,
                'rectangularity': rectangularity,
                'convexity': convexity,
                'edge_strength': edge_strength,
            }
            
            scored.append(candidate)
        
        return scored
    
    def _calculate_edge_strength(
        self,
        image: np.ndarray,
        corners: CardCorners,
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate edge strength around the candidate border.
        
        Args:
            image: Preprocessed image
            corners: Card corners
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Edge strength score (0.0 to 1.0)
        """
        x, y, w, h = bbox
        
        # Extract ROI
        if w <= 0 or h <= 0:
            return 0.0
        
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0.0
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Sample points along the border
        border_width = max(5, int(min(w, h) * 0.1))
        
        # Top and bottom borders
        top_border = magnitude[:border_width, :].mean()
        bottom_border = magnitude[-border_width:, :].mean()
        
        # Left and right borders
        left_border = magnitude[:, :border_width].mean()
        right_border = magnitude[:, -border_width:].mean()
        
        avg_border_strength = (top_border + bottom_border + left_border + right_border) / 4
        
        # Normalize to 0-1 range (assuming max gradient ~255)
        return min(avg_border_strength / 255.0, 1.0)
    
    def _validate_candidate(
        self,
        candidate: Dict[str, Any],
        image_width: int,
        image_height: int
    ) -> Tuple[bool, List[str]]:
        """
        Level 4: Validate candidate geometry.
        
        Rejects candidates with:
        - Impossible geometry
        - Extremely poor aspect ratio
        - Tiny area
        - Self-intersection
        - Invalid corners
        - Unrealistic perspective
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        corners = candidate['corners']
        metrics = candidate.get('metrics', {})
        
        # Check area
        image_area = image_width * image_height
        min_area = image_area * self.card_config.min_card_area_fraction
        
        if metrics.get('area_ratio', 0) * image_area < min_area:
            issues.append(f"Area too small ({metrics.get('area_ratio', 0):.3f} of image)")
        
        # Check aspect ratio
        ar = metrics.get('aspect_ratio', 0)
        expected_ar = self.card_config.expected_aspect_ratio
        tolerance = self.card_config.aspect_ratio_tolerance
        
        if ar < (expected_ar - tolerance) or ar > (expected_ar + tolerance):
            issues.append(f"Aspect ratio {ar:.2f} outside acceptable range [{expected_ar - tolerance:.2f}, {expected_ar + tolerance:.2f}]")
        
        # Check rectangularity
        if metrics.get('rectangularity', 0) < self.config.min_rectangularity:
            issues.append(f"Rectangularity too low ({metrics.get('rectangularity', 0):.2f})")
        
        # Check convexity
        if metrics.get('convexity', 0) < self.config.min_convexity:
            issues.append(f"Convexity too low ({metrics.get('convexity', 0):.2f})")
        
        # Check corner validity
        corners_valid, corner_issues = validate_corners(corners, image_width, image_height)
        if not corners_valid:
            issues.extend(corner_issues)
        
        return len(issues) == 0, issues
    
    def _try_full_frame_fallback(
        self,
        image: np.ndarray,
        original_width: int,
        original_height: int,
        start_time: float
    ) -> CardDetectionResult:
        """
        Try to use the full frame as a fallback when no card is detected.
        
        Only used when the entire frame strongly satisfies card-like geometry.
        
        Returns:
            CardDetectionResult with FULL_FRAME_CARD_FALLBACK status or CARD_NOT_DETECTED
        """
        # Check if image dimensions are card-like
        image_ar = original_width / original_height if original_height > 0 else 0
        expected_ar = self.card_config.expected_aspect_ratio
        tolerance = self.card_config.aspect_ratio_tolerance * 2  # More lenient for fallback
        
        if abs(image_ar - expected_ar) <= tolerance:
            # Create corners for full frame
            corners = CardCorners(
                top_left=Corner(x=0, y=0),
                top_right=Corner(x=original_width, y=0),
                bottom_right=Corner(x=original_width, y=original_height),
                bottom_left=Corner(x=0, y=original_height),
            )
            
            bbox = BoundingBox(
                x1=0,
                y1=0,
                x2=original_width,
                y2=original_height,
            )
            
            return CardDetectionResult(
                status=CardDetectionStatus.FULL_FRAME_CARD_FALLBACK,
                bbox=bbox,
                corners=corners,
                confidence=0.3,  # Low confidence for fallback
                failure_reason="Using full frame as fallback - no distinct card detected",
                detection_time_ms=(time.time() - start_time) * 1000,
            )
        
        return CardDetectionResult(
            status=CardDetectionStatus.CARD_NOT_DETECTED,
            failure_reason="No card-like region detected",
            detection_time_ms=(time.time() - start_time) * 1000,
        )
    
    def _scale_corners_to_original(
        self,
        corners: CardCorners,
        scale_x: float,
        scale_y: float
    ) -> CardCorners:
        """Scale corners from detection space back to original image space."""
        return CardCorners(
            top_left=Corner(
                x=corners.top_left.x * scale_x,
                y=corners.top_left.y * scale_y,
            ),
            top_right=Corner(
                x=corners.top_right.x * scale_x,
                y=corners.top_right.y * scale_y,
            ),
            bottom_right=Corner(
                x=corners.bottom_right.x * scale_x,
                y=corners.bottom_right.y * scale_y,
            ),
            bottom_left=Corner(
                x=corners.bottom_left.x * scale_x,
                y=corners.bottom_left.y * scale_y,
            ),
        )
    
    def _corners_to_bbox(self, corners: CardCorners) -> BoundingBox:
        """Convert corners to axis-aligned bounding box."""
        points = np.array(corners.to_list())
        
        x_min = int(np.floor(points[:, 0].min()))
        y_min = int(np.floor(points[:, 1].min()))
        x_max = int(np.ceil(points[:, 0].max()))
        y_max = int(np.ceil(points[:, 1].max()))
        
        return BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max)
    
    def _calculate_final_confidence(
        self,
        candidate: Dict[str, Any],
        corners: CardCorners,
        image_width: int,
        image_height: int
    ) -> float:
        """Calculate final detection confidence."""
        base_score = candidate['score']
        
        # Adjust based on corner quality
        angles = [90.0] * 4  # Placeholder - could calculate actual angles
        
        # Penalize for extreme perspective
        rectangularity = candidate.get('metrics', {}).get('rectangularity', 1.0)
        perspective_penalty = max(0, (1.0 - rectangularity) * 0.3)
        
        final_confidence = base_score * (1.0 - perspective_penalty)
        
        return min(max(final_confidence, 0.0), 1.0)
