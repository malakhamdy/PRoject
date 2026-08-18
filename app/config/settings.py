"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Centralized Configuration Module

All thresholds, dimensions, and runtime settings are defined here.
NO magic numbers should exist elsewhere in the codebase.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import os


@dataclass
class ImageConfig:
    """Image processing limits and settings."""
    max_width: int = 4096
    max_height: int = 4096
    min_width: int = 100
    min_height: int = 100
    max_file_size_mb: int = 50
    supported_formats: Tuple[str, ...] = ("jpg", "jpeg", "png", "bmp", "tiff", "webp")
    
    # Detection resolution (downscaled for performance)
    detection_max_dim: int = 1280
    detection_min_dim: int = 300


@dataclass
class CardConfig:
    """Egyptian National ID card physical properties and tolerances."""
    # Canonical card dimensions (in pixels after rectification)
    canonical_width: int = 1000
    canonical_height: int = 630
    
    # Aspect ratio tolerance (Egyptian ID is approximately 1.587:1)
    expected_aspect_ratio: float = 1.587
    aspect_ratio_tolerance: float = 0.3
    
    # Minimum card area as fraction of image
    min_card_area_fraction: float = 0.05
    
    # Corner detection parameters
    min_corner_distance: int = 50
    max_corner_angle_deviation: float = 30.0  # degrees from 90°
    
    # Perspective plausibility
    min_perspective_quality: float = 0.3
    max_perspective_distortion: float = 0.6


@dataclass
class DetectionConfig:
    """Card detection parameters."""
    # Edge detection
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    
    # Morphology
    kernel_size: int = 5
    morphology_iterations: int = 2
    
    # Contour filtering
    min_contour_area: int = 10000
    min_rectangularity: float = 0.6  # Lowered to handle moderate rotation
    min_convexity: float = 0.75  # Slightly lowered for robustness
    
    # Candidate scoring weights
    weight_area: float = 0.25
    weight_aspect_ratio: float = 0.20
    weight_rectangularity: float = 0.25
    weight_convexity: float = 0.15
    weight_edge_strength: float = 0.15
    
    # Minimum confidence to accept detection
    min_detection_confidence: float = 0.5


@dataclass
class RectificationConfig:
    """Perspective rectification settings."""
    # Border margins (pixels in canonical space)
    border_margin_x: int = 10
    border_margin_y: int = 10
    
    # Interpolation method (OpenCV constant)
    interpolation_method: int = 1  # INTER_LINEAR
    
    # Minimum quality thresholds
    min_geometry_validity: float = 0.7
    min_corner_quality: float = 0.6
    min_area_coverage: float = 0.8
    
    # Confidence thresholds
    low_confidence_threshold: float = 0.5
    critical_confidence_threshold: float = 0.3


@dataclass
class LocalizationConfig:
    """Field localization parameters."""
    # Normalized layout coordinates (0-1 scale relative to canonical card)
    # These are INITIAL PROPOSALS only, not final bboxes
    
    # National ID Number region
    nid_region: Tuple[float, float, float, float] = (0.05, 0.15, 0.95, 0.30)
    
    # Full Arabic Name region
    name_region: Tuple[float, float, float, float] = (0.05, 0.30, 0.95, 0.45)
    
    # Date of Birth region
    dob_region: Tuple[float, float, float, float] = (0.05, 0.45, 0.50, 0.58)
    
    # Gender region
    gender_region: Tuple[float, float, float, float] = (0.55, 0.45, 0.95, 0.58)
    
    # Governorate region
    governorate_region: Tuple[float, float, float, float] = (0.05, 0.58, 0.50, 0.72)
    
    # Address region
    address_region: Tuple[float, float, float, float] = (0.05, 0.72, 0.95, 0.90)
    
    # Field-specific parameters
    nid_min_digit_density: float = 0.4
    name_min_arabic_density: float = 0.5
    dob_expected_width_ratio: float = 0.3
    gender_max_width_ratio: float = 0.15
    
    # Content-aware refinement
    expansion_step: int = 5
    max_expansion_steps: int = 4
    contraction_step: int = 3
    max_contraction_steps: int = 3
    
    # Minimum localization confidence
    min_localization_confidence: float = 0.5
    low_confidence_threshold: float = 0.6
    
    # Maximum alternative candidates per field
    max_alternatives: int = 3
    
    # Crop margins (pixels in canonical space)
    crop_margin_x: int = 5
    crop_margin_y: int = 3


@dataclass
class QualityConfig:
    """Image and field quality assessment thresholds."""
    # Sharpness (Laplacian variance)
    min_sharpness: float = 50.0
    low_sharpness_threshold: float = 100.0
    
    # Brightness
    min_brightness: float = 30
    max_brightness: float = 220
    
    # Contrast
    min_contrast: float = 20
    
    # Noise level
    max_noise_level: float = 50.0
    
    # Empty area ratio
    max_empty_ratio: float = 0.7
    min_content_ratio: float = 0.15


@dataclass
class DebugConfig:
    """Debug mode settings."""
    enabled: bool = False
    save_intermediate_images: bool = True
    save_crops: bool = True
    visualization_font_size: int = 14
    visualization_box_thickness: int = 2
    debug_output_dir: str = "debug"


@dataclass
class RuntimeConfig:
    """Runtime and performance settings."""
    max_processing_time_ms: int = 10000
    enable_timing_metrics: bool = True
    log_level: str = "INFO"
    max_workers: int = 4


@dataclass
class AppConfig:
    """Main application configuration container."""
    image: ImageConfig = field(default_factory=ImageConfig)
    card: CardConfig = field(default_factory=CardConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    rectification: RectificationConfig = field(default_factory=RectificationConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Override debug mode from environment
        debug_env = os.getenv("OCI_DEBUG", "false").lower()
        config.debug.enabled = debug_env in ("true", "1", "yes")
        
        # Override log level
        log_level = os.getenv("OCI_LOG_LEVEL", "INFO")
        config.runtime.log_level = log_level
        
        return config


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = AppConfig.from_env()
