"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Integration Tests for Phase 1-3

Tests the full pipeline with synthetic test fixtures.
Does NOT use real personal National ID images.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from app.config import AppConfig, get_config, set_config
from app.schemas.models import (
    CardDetectionStatus,
    RectificationStatus,
    FieldStatus,
)
from app.detection import CardDetector
from app.rectification import CardRectifier
from app.localization import FieldLocalizer
from app.pipeline import OCIPipeline


def create_synthetic_card_image(
    width: int = 800,
    height: int = 500,
    position: tuple = None,
    rotation: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Create a synthetic Egyptian-ID-like test image.
    
    Uses fake/non-sensitive content only.
    Simulates:
    - Card border
    - Arabic labels (simulated with rectangles)
    - Numeric regions
    - Text blocks
    
    Args:
        width: Image width
        height: Image height
        position: (x, y) offset for card position
        rotation: Rotation angle in degrees
        scale: Scale factor for card size
        
    Returns:
        Synthetic test image (BGR format)
    """
    # Create blank background
    image = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Calculate card dimensions
    card_width = int(min(width, height * 1.6) * scale * 0.8)
    card_height = int(card_width / 1.6)
    
    # Center card by default
    if position is None:
        card_x = (width - card_width) // 2
        card_y = (height - card_height) // 2
    else:
        card_x, card_y = position
    
    # Draw card border
    cv2.rectangle(
        image,
        (card_x, card_y),
        (card_x + card_width, card_y + card_height),
        (50, 50, 50),  # Dark gray border
        3,
    )
    
    # Fill card interior
    cv2.rectangle(
        image,
        (card_x + 3, card_y + 3),
        (card_x + card_width - 3, card_y + card_height - 3),
        (255, 255, 255),  # White interior
        -1,
    )
    
    # Simulate field regions with gray rectangles (representing text areas)
    field_y_start = card_y + int(card_height * 0.15)
    field_height = int(card_height * 0.08)
    
    # NID region (upper portion, digit-heavy)
    nid_y = field_y_start
    cv2.rectangle(
        image,
        (card_x + int(card_width * 0.1), nid_y),
        (card_x + int(card_width * 0.9), nid_y + field_height),
        (200, 200, 200),  # Light gray for text region
        -1,
    )
    
    # Name region (below NID)
    name_y = nid_y + int(field_height * 1.5)
    cv2.rectangle(
        image,
        (card_x + int(card_width * 0.1), name_y),
        (card_x + int(card_width * 0.9), name_y + field_height),
        (200, 200, 200),
        -1,
    )
    
    # DOB region (left side, below name)
    dob_y = name_y + int(field_height * 1.5)
    cv2.rectangle(
        image,
        (card_x + int(card_width * 0.1), dob_y),
        (card_x + int(card_width * 0.45), dob_y + field_height),
        (200, 200, 200),
        -1,
    )
    
    # Gender region (right side, same level as DOB)
    cv2.rectangle(
        image,
        (card_x + int(card_width * 0.55), dob_y),
        (card_x + int(card_width * 0.9), dob_y + field_height),
        (200, 200, 200),
        -1,
    )
    
    # Governorate region (below DOB)
    gov_y = dob_y + int(field_height * 1.5)
    cv2.rectangle(
        image,
        (card_x + int(card_width * 0.1), gov_y),
        (card_x + int(card_width * 0.45), gov_y + field_height),
        (200, 200, 200),
        -1,
    )
    
    # Address region (larger block at bottom)
    addr_y = gov_y + int(field_height * 1.5)
    cv2.rectangle(
        image,
        (card_x + int(card_width * 0.1), addr_y),
        (card_x + int(card_width * 0.9), addr_y + int(field_height * 2)),
        (200, 200, 200),
        -1,
    )
    
    # Add some noise/texture to make it more realistic
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


class TestCardDetector:
    """Test card detection with synthetic images."""
    
    def test_detect_centered_card(self):
        """Test detection of centered card."""
        image = create_synthetic_card_image(800, 500)
        
        detector = CardDetector()
        result = detector.detect(image)
        
        # Should detect the card
        assert result.status in [
            CardDetectionStatus.DETECTED,
            CardDetectionStatus.FULL_FRAME_CARD_FALLBACK,
        ]
        assert result.bbox is not None
        assert result.confidence > 0.0
    
    def test_detect_shifted_card(self):
        """Test detection of shifted card (Variant B)."""
        image = create_synthetic_card_image(
            800, 500,
            position=(100, 50)  # Shifted from center
        )
        
        detector = CardDetector()
        result = detector.detect(image)
        
        assert result.status in [
            CardDetectionStatus.DETECTED,
            CardDetectionStatus.FULL_FRAME_CARD_FALLBACK,
        ]
        assert result.bbox is not None
    
    def test_detect_scaled_card(self):
        """Test detection of scaled card (Variant C)."""
        image = create_synthetic_card_image(
            800, 500,
            scale=0.6  # Smaller card
        )
        
        detector = CardDetector()
        result = detector.detect(image)
        
        # Should still detect or fallback
        assert result.status in [
            CardDetectionStatus.DETECTED,
            CardDetectionStatus.FULL_FRAME_CARD_FALLBACK,
            CardDetectionStatus.CARD_TOO_SMALL,
        ]
    
    def test_detect_rotated_card(self):
        """Test detection of rotated card (Variant D)."""
        # Create centered card first
        image = create_synthetic_card_image(800, 600)
        
        # Rotate the image
        center = (400, 300)
        matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
        rotated = cv2.warpAffine(image, matrix, (800, 600))
        
        detector = CardDetector()
        result = detector.detect(rotated)
        
        # Should handle moderate rotation
        assert result.status in [
            CardDetectionStatus.DETECTED,
            CardDetectionStatus.FULL_FRAME_CARD_FALLBACK,
        ]


class TestCardRectifier:
    """Test card rectification."""
    
    def test_rectify_detected_card(self):
        """Test rectification after successful detection."""
        image = create_synthetic_card_image(800, 500)
        
        detector = CardDetector()
        detection_result = detector.detect(image)
        
        if detection_result.status == CardDetectionStatus.DETECTED:
            rectifier = CardRectifier()
            result = rectifier.rectify(image, detection_result)
            
            assert result.status == RectificationStatus.SUCCESS
            assert result.canonical_image is not None
            assert result.output_width == 1000
            assert result.output_height == 630
            assert result.rectification_confidence > 0.0


class TestFieldLocalizer:
    """Test field localization."""
    
    def test_localize_all_fields(self):
        """Test localization of all fields on canonical image."""
        # Create a canonical-sized image with field regions
        canonical = np.ones((630, 1000, 3), dtype=np.uint8) * 255
        
        # Add simulated field content
        # NID region
        cv2.rectangle(
            canonical,
            (50, 100),
            (950, 190),
            (200, 200, 200),
            -1,
        )
        
        localizer = FieldLocalizer()
        
        # Create mock rectification result
        from app.schemas.models import RectificationResult
        rect_result = RectificationResult(
            status=RectificationStatus.SUCCESS,
            canonical_image=canonical,
            rectification_confidence=0.9,
            output_width=1000,
            output_height=630,
        )
        
        results = localizer.localize_all_fields(canonical, rect_result)
        
        # Should have results for all fields
        assert len(results) == 6
        assert "nid" in results
        assert "name" in results
        assert "dob" in results
        assert "gender" in results
        assert "governorate" in results
        assert "address" in results
        
        # Each should have a bbox
        for field_name, result in results.items():
            assert result.field_name == field_name
            assert result.bbox is not None


class TestOCIPipeline:
    """Test full pipeline integration."""
    
    def test_pipeline_synthetic_image(self):
        """Test full pipeline with synthetic image."""
        image = create_synthetic_card_image(800, 500)
        
        pipeline = OCIPipeline()
        result = pipeline.process(image)
        
        # Should complete processing
        assert result.original_width == 800
        assert result.original_height == 500
        
        # Should have card detection
        assert result.card_detection is not None
        
        # Should have rectification or failure reason
        if result.card_detection.status == CardDetectionStatus.DETECTED:
            assert result.rectification is not None
            
            # Should have field localizations
            assert len(result.fields) > 0
    
    def test_pipeline_empty_image(self):
        """Test pipeline with empty/invalid image."""
        pipeline = OCIPipeline()
        
        # Test with None
        result = pipeline.process(None)
        assert result.success is False
        assert "invalid" in result.message.lower()
    
    def test_pipeline_generalization_variants(self):
        """Test pipeline generalization across image variants."""
        pipeline = OCIPipeline()
        
        variants = []
        
        # Variant A: Centered
        variants.append(("centered", create_synthetic_card_image(800, 500)))
        
        # Variant B: Shifted
        variants.append(("shifted", create_synthetic_card_image(800, 500, position=(150, 80))))
        
        # Variant C: Scaled smaller
        variants.append(("scaled", create_synthetic_card_image(800, 500, scale=0.7)))
        
        # Variant E: Perspective distortion (simulated with affine transform)
        base = create_synthetic_card_image(800, 500)
        src_pts = np.float32([[0, 0], [800, 0], [800, 500], [0, 500]])
        dst_pts = np.float32([[50, 50], [750, 30], [780, 480], [20, 500]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        perspective = cv2.warpPerspective(base, matrix, (800, 500))
        variants.append(("perspective", perspective))
        
        # Process each variant
        results = {}
        for name, image in variants:
            result = pipeline.process(image)
            results[name] = result
            
            # Each should produce some result
            assert result is not None
            assert result.original_width > 0
            assert result.original_height > 0
        
        # Verify that fields are localized in at least some variants
        localized_count = sum(
            1 for r in results.values()
            if r.fields and any(
                f.validation_status == FieldStatus.LOCALIZED
                for f in r.fields.values()
            )
        )
        
        # At least half the variants should have some localization
        assert localized_count >= len(variants) * 0.5


# Run tests with: pytest tests/integration/test_pipeline.py -v
