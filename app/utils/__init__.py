"""App utils package."""
from .geometry import (
    order_corners,
    validate_corners,
    calculate_corner_angles,
    calculate_quadrilateral_area,
    calculate_rectangularity,
    calculate_convexity,
    get_perspective_transform_matrix,
    apply_perspective_transform,
    map_bbox_from_source_to_canonical,
    map_bbox_from_canonical_to_source,
    calculate_aspect_ratio,
    normalize_coordinates_for_detection,
    scale_bbox,
)

__all__ = [
    "order_corners",
    "validate_corners",
    "calculate_corner_angles",
    "calculate_quadrilateral_area",
    "calculate_rectangularity",
    "calculate_convexity",
    "get_perspective_transform_matrix",
    "apply_perspective_transform",
    "map_bbox_from_source_to_canonical",
    "map_bbox_from_canonical_to_source",
    "calculate_aspect_ratio",
    "normalize_coordinates_for_detection",
    "scale_bbox",
]
