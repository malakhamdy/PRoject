"""
OCI - Egyptian National ID Intelligent OCR, Validation and Document Analysis System
Geometry Utilities

Provides geometric operations for card detection, corner ordering,
perspective transformation, and coordinate mapping.
"""

import numpy as np
from typing import List, Tuple, Optional
from app.schemas.models import BoundingBox, Corner, CardCorners


def order_corners(points: np.ndarray) -> CardCorners:
    """
    Order four corner points into TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT.
    
    Args:
        points: numpy array of shape (4, 2) containing [x, y] coordinates
        
    Returns:
        CardCorners with properly ordered corners
    """
    if points.shape != (4, 2):
        raise ValueError(f"Expected 4 points with 2 coordinates each, got {points.shape}")
    
    # Sort by y-coordinate to separate top and bottom
    # Use a small tolerance to handle near-equal y values
    y_sorted_indices = np.argsort(points[:, 1])
    
    # Top two points (smaller y values)
    top_indices = y_sorted_indices[:2]
    top_points = points[top_indices]
    
    # Bottom two points (larger y values)
    bottom_indices = y_sorted_indices[2:]
    bottom_points = points[bottom_indices]
    
    # Among top points, left one has smaller x
    top_left_idx = top_indices[np.argmin(top_points[:, 0])]
    top_right_idx = top_indices[np.argmax(top_points[:, 0])]
    
    # Among bottom points, right one has larger x (for proper ordering)
    bottom_right_idx = bottom_indices[np.argmax(points[bottom_indices][:, 0])]
    bottom_left_idx = bottom_indices[np.argmin(points[bottom_indices][:, 0])]
    
    # Create corners with quality scores (default to 1.0)
    corners = CardCorners(
        top_left=Corner(x=float(points[top_left_idx][0]), y=float(points[top_left_idx][1])),
        top_right=Corner(x=float(points[top_right_idx][0]), y=float(points[top_right_idx][1])),
        bottom_right=Corner(x=float(points[bottom_right_idx][0]), y=float(points[bottom_right_idx][1])),
        bottom_left=Corner(x=float(points[bottom_left_idx][0]), y=float(points[bottom_left_idx][1])),
    )
    
    return corners


def validate_corners(corners: CardCorners, image_width: int, image_height: int) -> Tuple[bool, List[str]]:
    """
    Validate that corners form a valid quadrilateral within image bounds.
    
    Args:
        corners: CardCorners object
        image_width: Width of the source image
        image_height: Height of the source image
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check convexity
    if not corners.validate_convexity():
        issues.append("Corners do not form a convex quadrilateral")
    
    # Check self-intersection
    if not corners.validate_no_intersection():
        issues.append("Corner edges self-intersect")
    
    # Check bounds
    corner_list = corners.to_list()
    for i, (x, y) in enumerate(corner_list):
        if x < 0 or x > image_width:
            issues.append(f"Corner {i} x-coordinate ({x}) outside image bounds [0, {image_width}]")
        if y < 0 or y > image_height:
            issues.append(f"Corner {i} y-coordinate ({y}) outside image bounds [0, {image_height}]")
    
    # Check minimum distances between corners
    min_distance = 50  # pixels
    for i in range(4):
        for j in range(i + 1, 4):
            p1 = corner_list[i]
            p2 = corner_list[j]
            distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if distance < min_distance:
                issues.append(f"Corners {i} and {j} are too close ({distance:.1f}px < {min_distance}px)")
    
    # Check angles (should be roughly 90 degrees for a rectangular card)
    angles = calculate_corner_angles(corners)
    for i, angle in enumerate(angles):
        deviation = abs(angle - 90.0)
        if deviation > 45.0:  # Allow some perspective distortion
            issues.append(f"Corner {i} angle ({angle:.1f}°) deviates too much from 90°")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def calculate_corner_angles(corners: CardCorners) -> List[float]:
    """
    Calculate interior angles at each corner.
    
    Args:
        corners: CardCorners object
        
    Returns:
        List of 4 angles in degrees [TL, TR, BR, BL]
    """
    points = [
        np.array([corners.top_left.x, corners.top_left.y]),
        np.array([corners.top_right.x, corners.top_right.y]),
        np.array([corners.bottom_right.x, corners.bottom_right.y]),
        np.array([corners.bottom_left.x, corners.bottom_left.y]),
    ]
    
    angles = []
    for i in range(4):
        prev_point = points[(i - 1) % 4]
        curr_point = points[i]
        next_point = points[(i + 1) % 4]
        
        # Vectors from current point
        v1 = prev_point - curr_point
        v2 = next_point - curr_point
        
        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            angles.append(0.0)
        else:
            cos_angle = dot_product / (norm_v1 * norm_v2)
            # Clamp to [-1, 1] to handle numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)
    
    return angles


def calculate_quadrilateral_area(corners: CardCorners) -> float:
    """
    Calculate the area of a quadrilateral using the shoelace formula.
    
    Args:
        corners: CardCorners object
        
    Returns:
        Area in square units
    """
    points = corners.to_list()
    
    # Shoelace formula
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2.0


def calculate_rectangularity(corners: CardCorners) -> float:
    """
    Calculate how rectangular a quadrilateral is.
    
    Compares the area of the quadrilateral to its bounding box.
    
    Args:
        corners: CardCorners object
        
    Returns:
        Rectangularity score (0.0 to 1.0)
    """
    points = np.array(corners.to_list())
    
    # Quadrilateral area
    quad_area = calculate_quadrilateral_area(corners)
    
    # Bounding box area
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    bbox_area = (x_max - x_min) * (y_max - y_min)
    
    if bbox_area == 0:
        return 0.0
    
    return quad_area / bbox_area


def calculate_convexity(contour: np.ndarray) -> float:
    """
    Calculate convexity of a contour.
    
    Convexity = contour area / convex hull area
    
    Args:
        contour: OpenCV contour
        
    Returns:
        Convexity score (0.0 to 1.0)
    """
    import cv2
    
    contour_area = cv2.contourArea(contour)
    if contour_area <= 0:
        return 0.0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area <= 0:
        return 0.0
    
    return contour_area / hull_area


def get_perspective_transform_matrix(
    corners: CardCorners,
    output_width: int,
    output_height: int
) -> np.ndarray:
    """
    Calculate the perspective transformation matrix.
    
    Maps the detected card corners to a canonical rectangular output.
    
    Args:
        corners: CardCorners (ordered TL, TR, BR, BL)
        output_width: Width of the output image
        output_height: Height of the output image
        
    Returns:
        3x3 transformation matrix
    """
    import cv2
    
    # Source points (detected corners)
    src_points = np.array(corners.to_list(), dtype=np.float32)
    
    # Destination points (canonical rectangle)
    dst_points = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1],
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    return matrix


def apply_perspective_transform(
    image: np.ndarray,
    corners: CardCorners,
    output_width: int,
    output_height: int,
    interpolation: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply perspective transformation to rectify the card.
    
    Args:
        image: Input image (numpy array)
        corners: CardCorners (ordered TL, TR, BR, BL)
        output_width: Width of the output image
        output_height: Height of the output image
        interpolation: OpenCV interpolation method
        
    Returns:
        Tuple of (rectified_image, transformation_matrix)
    """
    import cv2
    
    matrix = get_perspective_transform_matrix(corners, output_width, output_height)
    
    rectified = cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height),
        flags=interpolation,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rectified, matrix


def map_bbox_from_source_to_canonical(
    bbox: BoundingBox,
    transform_matrix: np.ndarray,
    canonical_width: int,
    canonical_height: int
) -> BoundingBox:
    """
    Map a bounding box from source image coordinates to canonical coordinates.
    
    Args:
        bbox: BoundingBox in source image
        transform_matrix: 3x3 perspective transformation matrix
        canonical_width: Width of canonical output
        canonical_height: Height of canonical output
        
    Returns:
        BoundingBox in canonical coordinates
    """
    import cv2
    
    # Transform the four corners of the bbox
    src_points = np.array([
        [bbox.x1, bbox.y1],
        [bbox.x2, bbox.y1],
        [bbox.x2, bbox.y2],
        [bbox.x1, bbox.y2],
    ], dtype=np.float32)
    
    # Reshape for perspective transform
    src_points = src_points.reshape(-1, 1, 2)
    
    # Apply transform
    dst_points = cv2.perspectiveTransform(src_points, transform_matrix)
    dst_points = dst_points.reshape(-1, 2)
    
    # Get bounding box of transformed points
    x_min = int(np.floor(dst_points[:, 0].min()))
    y_min = int(np.floor(dst_points[:, 1].min()))
    x_max = int(np.ceil(dst_points[:, 0].max()))
    y_max = int(np.ceil(dst_points[:, 1].max()))
    
    # Clamp to canonical dimensions
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(canonical_width, x_max)
    y_max = min(canonical_height, y_max)
    
    return BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max)


def map_bbox_from_canonical_to_source(
    bbox: BoundingBox,
    transform_matrix: np.ndarray,
    source_width: int,
    source_height: int
) -> BoundingBox:
    """
    Map a bounding box from canonical coordinates back to source image.
    
    Uses the inverse of the transformation matrix.
    
    Args:
        bbox: BoundingBox in canonical coordinates
        transform_matrix: 3x3 perspective transformation matrix
        source_width: Width of source image
        source_height: Height of source image
        
    Returns:
        BoundingBox in source coordinates
    """
    import cv2
    
    # Compute inverse matrix
    inv_matrix = np.linalg.inv(transform_matrix)
    
    # Transform the four corners of the bbox
    src_points = np.array([
        [bbox.x1, bbox.y1],
        [bbox.x2, bbox.y1],
        [bbox.x2, bbox.y2],
        [bbox.x1, bbox.y2],
    ], dtype=np.float32)
    
    # Reshape for perspective transform
    src_points = src_points.reshape(-1, 1, 2)
    
    # Apply inverse transform
    dst_points = cv2.perspectiveTransform(src_points, inv_matrix)
    dst_points = dst_points.reshape(-1, 2)
    
    # Get bounding box of transformed points
    x_min = int(np.floor(dst_points[:, 0].min()))
    y_min = int(np.floor(dst_points[:, 1].min()))
    x_max = int(np.ceil(dst_points[:, 0].max()))
    y_max = int(np.ceil(dst_points[:, 1].max()))
    
    # Clamp to source dimensions
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(source_width, x_max)
    y_max = min(source_height, y_max)
    
    return BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max)


def calculate_aspect_ratio(corners: CardCorners) -> float:
    """
    Calculate the aspect ratio of a quadrilateral.
    
    Uses the average of top and bottom widths divided by average of left and right heights.
    
    Args:
        corners: CardCorners object
        
    Returns:
        Aspect ratio (width / height)
    """
    points = np.array(corners.to_list())
    
    # Top and bottom widths
    top_width = np.linalg.norm(points[0] - points[1])
    bottom_width = np.linalg.norm(points[3] - points[2])
    
    # Left and right heights
    left_height = np.linalg.norm(points[0] - points[3])
    right_height = np.linalg.norm(points[1] - points[2])
    
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    
    if avg_height == 0:
        return 0.0
    
    return avg_width / avg_height


def normalize_coordinates_for_detection(
    image_width: int,
    image_height: int,
    target_max_dim: int
) -> Tuple[float, float, int, int]:
    """
    Calculate scaling factors for detection-scale image.
    
    Args:
        image_width: Original image width
        image_height: Original image height
        target_max_dim: Maximum dimension for detection
        
    Returns:
        Tuple of (scale_x, scale_y, new_width, new_height)
    """
    max_dim = max(image_width, image_height)
    
    if max_dim <= target_max_dim:
        return 1.0, 1.0, image_width, image_height
    
    scale = target_max_dim / max_dim
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)
    
    return scale, scale, new_width, new_height


def scale_bbox(bbox: BoundingBox, scale_x: float, scale_y: float) -> BoundingBox:
    """
    Scale a bounding box by given factors.
    
    Args:
        bbox: Original bounding box
        scale_x: Horizontal scale factor
        scale_y: Vertical scale factor
        
    Returns:
        Scaled bounding box
    """
    return BoundingBox(
        x1=int(bbox.x1 * scale_x),
        y1=int(bbox.y1 * scale_y),
        x2=int(bbox.x2 * scale_x),
        y2=int(bbox.y2 * scale_y),
    )
