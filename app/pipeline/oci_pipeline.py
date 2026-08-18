"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Pipeline Module

Orchestrates all processing stages from image input through field localization.
Phases 1-3: Quality → Detection → Rectification → Localization
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
import time
import logging

from app.config import get_config, AppConfig
from app.schemas.models import (
    PipelineResult,
    CardDetectionResult,
    RectificationResult,
    FieldResult,
    PerformanceMetrics,
    ImageQualityResult,
    FieldStatus,
)
from app.detection import CardDetector
from app.rectification import CardRectifier
from app.localization import FieldLocalizer


logger = logging.getLogger(__name__)


class OCIPipeline:
    """
    Main OCI pipeline for Egyptian National ID processing.
    
    Orchestrates:
    1. Image quality assessment
    2. Card detection
    3. Card rectification
    4. Field localization
    
    Phases 1-3 stop before OCR.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the OCI pipeline.
        
        Args:
            config: Application configuration (uses global config if not provided)
        """
        self.config = config or get_config()
        
        # Initialize stage components
        self.detector = CardDetector()
        self.rectifier = CardRectifier()
        self.localizer = FieldLocalizer()
    
    def process(self, image: np.ndarray) -> PipelineResult:
        """
        Process an Egyptian National ID image through all implemented stages.
        
        Args:
            image: Input image (BGR format, numpy array)
            
        Returns:
            PipelineResult with results from all stages
        """
        start_time = time.time()
        metrics = PerformanceMetrics()
        
        result = PipelineResult(
            success=False,
            status="processing",
        )
        
        try:
            # Validate input
            if image is None or image.size == 0:
                return PipelineResult(
                    success=False,
                    status="failed",
                    message="Empty or invalid image provided",
                )
            
            # Store original dimensions
            result.original_height = image.shape[0]
            result.original_width = image.shape[1]
            
            # Stage 1: Image Quality Assessment
            quality_start = time.time()
            quality_result = self._assess_image_quality(image)
            metrics.quality_time_ms = (time.time() - quality_start) * 1000
            
            # Log quality issues but don't fail - let downstream stages handle
            if not quality_result.is_acceptable:
                logger.warning(f"Image quality issues: {quality_result.issues}")
            
            # Stage 2: Card Detection
            detection_start = time.time()
            detection_result = self.detector.detect(image)
            metrics.card_detection_time_ms = (time.time() - detection_start) * 1000
            
            result.card_detection = detection_result
            
            # Check detection status
            if detection_result.status.value.startswith("card_not") or \
               detection_result.status.value.startswith("card_geometry"):
                result.status = "card_detection_failed"
                result.message = f"Card detection failed: {detection_result.failure_reason}"
                result.metrics = metrics
                return result
            
            # Stage 3: Card Rectification
            rectification_start = time.time()
            rectification_result = self.rectifier.rectify(image, detection_result)
            metrics.rectification_time_ms = (time.time() - rectification_start) * 1000
            
            result.rectification = rectification_result
            
            # Check rectification status
            if rectification_result.status.value.startswith("rectification_failed"):
                result.status = "rectification_failed"
                result.message = f"Rectification failed: {rectification_result.failure_reason}"
                result.metrics = metrics
                return result
            
            # Stage 4: Field Localization (Phase 3)
            localization_start = time.time()
            
            if rectification_result.canonical_image is not None:
                localization_results = self.localizer.localize_all_fields(
                    rectification_result.canonical_image,
                    rectification_result
                )
                
                # Convert localization results to field results
                for field_name, loc_result in localization_results.items():
                    field_result = FieldResult(
                        field_name=field_name,
                        bbox=loc_result.bbox,
                        localization_confidence=loc_result.localization_confidence,
                        validation_status=loc_result.status,
                        source="localization",
                        failure_reason=loc_result.failure_reason,
                    )
                    result.fields[field_name] = field_result
                
                metrics.localization_time_ms = (time.time() - localization_start) * 1000
            
            # Determine overall status
            successful_localizations = sum(
                1 for f in result.fields.values()
                if f.validation_status == FieldStatus.LOCALIZED
            )
            
            if successful_localizations >= len(result.fields) * 0.5:
                result.success = True
                result.status = "localized"
                result.message = f"Successfully localized {successful_localizations}/{len(result.fields)} fields"
            else:
                result.status = "partial_localization"
                result.message = f"Only {successful_localizations}/{len(result.fields)} fields localized with confidence"
            
            # Store performance metrics
            metrics.total_time_ms = (time.time() - start_time) * 1000
            result.metrics = metrics
            
            # Add debug artifacts if enabled
            if self.config.debug.enabled:
                result.debug_artifacts = self._generate_debug_artifacts(
                    image, detection_result, rectification_result, localization_results
                )
            
            return result
            
        except Exception as e:
            logger.exception(f"Pipeline error: {str(e)}")
            result.status = "error"
            result.message = f"Pipeline error: {str(e)}"
            result.metrics = metrics
            return result
    
    def _assess_image_quality(self, image: np.ndarray) -> ImageQualityResult:
        """
        Assess image quality.
        
        Checks:
        - Sharpness (Laplacian variance)
        - Brightness
        - Contrast
        - Noise level
        """
        result = ImageQualityResult()
        
        if image is None or image.size == 0:
            result.is_acceptable = False
            result.issues.append("Empty image")
            return result
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        result.sharpness = float(np.var(laplacian))
        
        if result.sharpness < self.config.quality.min_sharpness:
            result.issues.append(f"Low sharpness ({result.sharpness:.1f})")
            result.recommendations.append("Use a sharper image")
        
        # Brightness
        result.brightness = float(np.mean(gray))
        
        if result.brightness < self.config.quality.min_brightness:
            result.issues.append(f"Image too dark ({result.brightness:.1f})")
            result.recommendations.append("Improve lighting")
        elif result.brightness > self.config.quality.max_brightness:
            result.issues.append(f"Image too bright ({result.brightness:.1f})")
            result.recommendations.append("Reduce glare/lighting")
        
        # Contrast (standard deviation)
        result.contrast = float(np.std(gray))
        
        if result.contrast < self.config.quality.min_contrast:
            result.issues.append(f"Low contrast ({result.contrast:.1f})")
            result.recommendations.append("Increase contrast")
        
        # Simple noise estimation (high-frequency content)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blur)
        result.noise_level = float(np.mean(noise))
        
        # Determine acceptability
        result.is_acceptable = (
            result.sharpness >= self.config.quality.min_sharpness * 0.5 and
            self.config.quality.min_brightness * 0.8 <= result.brightness <= self.config.quality.max_brightness * 1.2 and
            result.contrast >= self.config.quality.min_contrast * 0.7
        )
        
        return result
    
    def _generate_debug_artifacts(
        self,
        original_image: np.ndarray,
        detection_result: CardDetectionResult,
        rectification_result: RectificationResult,
        localization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate debug visualization artifacts."""
        artifacts = {}
        
        # Create annotated visualization
        if self.config.debug.save_intermediate_images:
            viz_image = self._create_debug_visualization(
                original_image,
                detection_result,
                rectification_result,
                localization_results
            )
            artifacts["visualization"] = viz_image
        
        # Store canonical card if available
        if rectification_result.canonical_image is not None:
            artifacts["canonical_card"] = rectification_result.canonical_image
        
        return artifacts
    
    def _create_debug_visualization(
        self,
        original_image: np.ndarray,
        detection_result: CardDetectionResult,
        rectification_result: RectificationResult,
        localization_results: Dict[str, Any]
    ) -> np.ndarray:
        """Create debug visualization with annotations."""
        import cv2
        
        # Start with a copy of original image
        viz = original_image.copy()
        
        # Draw card detection bounding box
        if detection_result.bbox:
            bbox = detection_result.bbox
            cv2.rectangle(
                viz,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                (0, 255, 0),  # Green
                self.config.debug.visualization_box_thickness,
            )
            
            # Add confidence label
            label = f"Card: {detection_result.confidence:.2f}"
            cv2.putText(
                viz,
                label,
                (bbox.x1, bbox.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        
        # Draw corners if available
        if detection_result.corners:
            corners = detection_result.corners.to_list()
            corner_names = ["TL", "TR", "BR", "BL"]
            corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            
            for i, (point, name) in enumerate(zip(corners, corner_names)):
                x, y = int(point[0]), int(point[1])
                color = corner_colors[i]
                
                # Draw point
                cv2.circle(viz, (x, y), 8, color, -1)
                
                # Draw label
                cv2.putText(
                    viz,
                    name,
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        
        # Add rectification info
        if rectification_result:
            info_y = 30
            cv2.putText(
                viz,
                f"Rect: {rectification_result.rectification_confidence:.2f}",
                (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
        
        # Add field localization boxes (on canonical card overlay or separate)
        if localization_results and rectification_result.canonical_image is not None:
            # Create separate visualization for canonical card with fields
            canonical_viz = rectification_result.canonical_image.copy()
            
            for field_name, loc_result in localization_results.items():
                if loc_result.bbox:
                    bbox = loc_result.bbox
                    
                    # Different colors for different confidence levels
                    if loc_result.localization_confidence >= 0.8:
                        color = (0, 255, 0)  # Green - high confidence
                    elif loc_result.localization_confidence >= 0.6:
                        color = (0, 255, 255)  # Cyan - medium confidence
                    else:
                        color = (0, 0, 255)  # Red - low confidence
                    
                    cv2.rectangle(
                        canonical_viz,
                        (bbox.x1, bbox.y1),
                        (bbox.x2, bbox.y2),
                        color,
                        self.config.debug.visualization_box_thickness,
                    )
                    
                    # Add field info
                    label_lines = [
                        field_name.upper(),
                        f"Loc: {loc_result.localization_confidence:.2f}",
                        "OCR: N/A",
                    ]
                    
                    for i, line in enumerate(label_lines):
                        cv2.putText(
                            canonical_viz,
                            line,
                            (bbox.x1, bbox.y1 + i * 20 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )
            
            artifacts = {"canonical_with_fields": canonical_viz}
        
        return viz
