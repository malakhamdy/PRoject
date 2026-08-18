"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Unit Tests for Phase 1-3

Tests for:
- Configuration
- Bounding box validation
- Schema validation
- Pipeline contracts
- Corner ordering
- Card geometry validation
"""

import pytest
import numpy as np
import cv2

from app.config import AppConfig, get_config, reset_config
from app.schemas.models import (
    BoundingBox,
    Corner,
    CardCorners,
    CardDetectionResult,
    CardDetectionStatus,
    RectificationResult,
    RectificationStatus,
    LocalizationResult,
    FieldStatus,
)
from app.utils.geometry import (
    order_corners,
    validate_corners,
    calculate_quadrilateral_area,
    calculate_rectangularity,
    scale_bbox,
)


class TestBoundingBox:
    """Test BoundingBox schema."""
    
    def test_valid_bbox(self):
        """Test valid bounding box creation."""
        bbox = BoundingBox(x1=10, y1=10, x2=100, y2=100)
        
        assert bbox.width == 90
        assert bbox.height == 90
        assert bbox.area == 8100
        assert bbox.aspect_ratio == 1.0
    
    def test_bbox_validation_success(self):
        """Test bbox validation passes for valid box."""
        bbox = BoundingBox(x1=10, y1=10, x2=100, y2=100)
        assert bbox.validate(max_x=200, max_y=200) is True
    
    def test_bbox_validation_fails_inverted(self):
        """Test bbox validation fails for inverted coordinates."""
        bbox = BoundingBox(x1=100, y1=100, x2=10, y2=10)
        assert bbox.validate(max_x=200, max_y=200) is False
    
    def test_bbox_validation_fails_out_of_bounds(self):
        """Test bbox validation fails for out of bounds."""
        bbox = BoundingBox(x1=10, y1=10, x2=300, y2=300)
        assert bbox.validate(max_x=200, max_y=200) is False
    
    def test_bbox_to_list(self):
        """Test bbox conversion to list."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        assert bbox.to_list() == [10, 20, 100, 200]
    
    def test_bbox_from_list(self):
        """Test bbox creation from list."""
        bbox = BoundingBox.from_list([10, 20, 100, 200])
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 200


class TestCardCorners:
    """Test CardCorners schema."""
    
    def test_corner_ordering(self):
        """Test corner ordering function."""
        # Create points in random order
        points = np.array([
            [100, 100],  # TL
            [200, 100],  # TR
            [200, 150],  # BR
            [100, 150],  # BL
        ], dtype=np.float32)
        
        corners = order_corners(points)
        
        assert corners.top_left.x == 100
        assert corners.top_left.y == 100
        assert corners.top_right.x == 200
        assert corners.top_right.y == 100
        assert corners.bottom_right.x == 200
        assert corners.bottom_right.y == 150
        assert corners.bottom_left.x == 100
        assert corners.bottom_left.y == 150
    
    def test_convexity_validation(self):
        """Test convexity validation for valid quadrilateral."""
        corners = CardCorners(
            top_left=Corner(x=0, y=0),
            top_right=Corner(x=100, y=0),
            bottom_right=Corner(x=100, y=50),
            bottom_left=Corner(x=0, y=50),
        )
        assert corners.validate_convexity() is True
    
    def test_invalid_convexity(self):
        """Test convexity validation catches invalid shape."""
        # Create a bow-tie shape (self-intersecting)
        corners = CardCorners(
            top_left=Corner(x=0, y=0),
            top_right=Corner(x=100, y=50),
            bottom_right=Corner(x=0, y=50),
            bottom_left=Corner(x=100, y=0),
        )
        assert corners.validate_convexity() is False


class TestConfiguration:
    """Test configuration system."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AppConfig()
        
        assert config.card.canonical_width == 1000
        assert config.card.canonical_height == 630
        assert config.detection.min_detection_confidence == 0.5
    
    def test_config_reset(self):
        """Test configuration reset."""
        reset_config()
        config = get_config()
        assert config.debug.enabled == False
    
    def test_localization_regions(self):
        """Test localization region configuration."""
        config = AppConfig()
        
        # Check all field regions are defined
        assert config.localization.nid_region is not None
        assert config.localization.name_region is not None
        assert config.localization.dob_region is not None
        assert config.localization.gender_region is not None
        assert config.localization.governorate_region is not None
        assert config.localization.address_region is not None


class TestGeometryUtils:
    """Test geometry utility functions."""
    
    def test_scale_bbox(self):
        """Test bounding box scaling."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        scaled = scale_bbox(bbox, scale_x=2.0, scale_y=2.0)
        
        assert scaled.x1 == 20
        assert scaled.y1 == 40
        assert scaled.x2 == 200
        assert scaled.y2 == 400
    
    def test_quadrilateral_area(self):
        """Test quadrilateral area calculation."""
        corners = CardCorners(
            top_left=Corner(x=0, y=0),
            top_right=Corner(x=100, y=0),
            bottom_right=Corner(x=100, y=50),
            bottom_left=Corner(x=0, y=50),
        )
        
        area = calculate_quadrilateral_area(corners)
        assert area == 5000.0  # 100 * 50
    
    def test_rectangularity_perfect_rectangle(self):
        """Test rectangularity for perfect rectangle."""
        corners = CardCorners(
            top_left=Corner(x=0, y=0),
            top_right=Corner(x=100, y=0),
            bottom_right=Corner(x=100, y=50),
            bottom_left=Corner(x=0, y=50),
        )
        
        rectangularity = calculate_rectangularity(corners)
        assert rectangularity == 1.0


class TestLocalizationResult:
    """Test LocalizationResult schema."""
    
    def test_localized_field(self):
        """Test localized field result."""
        bbox = BoundingBox(x1=10, y1=10, x2=100, y2=50)
        result = LocalizationResult(
            field_name="nid",
            bbox=bbox,
            localization_confidence=0.95,
            status=FieldStatus.LOCALIZED,
        )
        
        assert result.field_name == "nid"
        assert result.localization_confidence == 0.95
        assert result.status == FieldStatus.LOCALIZED
    
    def test_failed_localization(self):
        """Test failed localization result."""
        result = LocalizationResult(
            field_name="name",
            bbox=None,
            localization_confidence=0.0,
            status=FieldStatus.FIELD_LOCALIZATION_FAILED,
            failure_reason="Empty canonical image",
        )
        
        assert result.bbox is None
        assert result.localization_confidence == 0.0
        assert result.failure_reason is not None
    
    def test_to_dict(self):
        """Test localization result dictionary conversion."""
        bbox = BoundingBox(x1=10, y1=10, x2=100, y2=50)
        result = LocalizationResult(
            field_name="dob",
            bbox=bbox,
            localization_confidence=0.85,
            status=FieldStatus.LOCALIZED,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["field_name"] == "dob"
        assert result_dict["bbox"] == [10, 10, 100, 50]
        assert result_dict["localization_confidence"] == 0.85


class TestCardDetectionResult:
    """Test CardDetectionResult schema."""
    
    def test_detected_status(self):
        """Test card detection with successful status."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        result = CardDetectionResult(
            status=CardDetectionStatus.DETECTED,
            bbox=bbox,
            confidence=0.95,
        )
        
        assert result.status == CardDetectionStatus.DETECTED
        assert result.confidence == 0.95
    
    def test_not_detected_status(self):
        """Test card detection failure."""
        result = CardDetectionResult(
            status=CardDetectionStatus.CARD_NOT_DETECTED,
            confidence=0.0,
            failure_reason="No card-like region detected",
        )
        
        assert result.status == CardDetectionStatus.CARD_NOT_DETECTED
        assert result.bbox is None
        assert result.failure_reason is not None


class TestRectificationResult:
    """Test RectificationResult schema."""
    
    def test_successful_rectification(self):
        """Test successful rectification result."""
        result = RectificationResult(
            status=RectificationStatus.SUCCESS,
            rectification_confidence=0.92,
            output_width=1000,
            output_height=630,
        )
        
        assert result.status == RectificationStatus.SUCCESS
        assert result.rectification_confidence == 0.92
    
    def test_failed_rectification(self):
        """Test failed rectification."""
        result = RectificationResult(
            status=RectificationStatus.RECTIFICATION_FAILED,
            rectification_confidence=0.0,
            failure_reason="No corners available",
        )
        
        assert result.status == RectificationStatus.RECTIFICATION_FAILED
        assert result.failure_reason is not None


# Run tests with: pytest tests/unit/test_schemas.py -v
