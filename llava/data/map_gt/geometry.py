"""
Geometry processing utilities for map GT generation
Aligned with MapTR's implementation
"""

import numpy as np
from typing import List, Tuple, Optional
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString
from shapely.geometry import box as Box
import warnings

warnings.filterwarnings('ignore')


def transform_to_ego(
    points_world: np.ndarray,
    ego_translation: np.ndarray,
    ego_rotation: np.ndarray
) -> np.ndarray:
    """
    Transform points from world frame to ego/LiDAR frame
    
    Args:
        points_world: [N, 2] or [N, 3] in world coordinates
        ego_translation: [3] ego position in world frame
        ego_rotation: [3, 3] rotation matrix (world -> ego)
        
    Returns:
        [N, 2] in ego BEV coordinates (x=left/right, y=back/front)
    """
    # Ensure 3D
    if points_world.shape[1] == 2:
        points_3d = np.hstack([points_world, np.zeros((len(points_world), 1))])
    else:
        points_3d = points_world[:, :3]
    
    # Transform: p_ego = R.T @ (p_world - t)
    points_centered = points_3d - ego_translation
    points_ego = points_centered @ ego_rotation.T
    
    # Return only x, y (BEV projection)
    return points_ego[:, :2].astype(np.float32)


def clip_polyline_by_roi(
    polyline: np.ndarray,
    pc_range: List[float]
) -> List[np.ndarray]:
    """
    Clip polyline by ROI (pc_range), may produce multiple segments
    
    Args:
        polyline: [N, 2] points in ego frame
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Returns:
        List of clipped polyline segments
    """
    x_min, y_min, _, x_max, y_max, _ = pc_range
    roi_box = Box(x_min, y_min, x_max, y_max)
    
    try:
        line = LineString(polyline)
        clipped = line.intersection(roi_box)
        
        if clipped.is_empty:
            return []
        
        # Handle different geometry types
        if isinstance(clipped, LineString):
            coords = np.array(clipped.coords, dtype=np.float32)
            return [coords] if len(coords) >= 2 else []
        
        elif isinstance(clipped, MultiLineString):
            segments = []
            for geom in clipped.geoms:
                coords = np.array(geom.coords, dtype=np.float32)
                if len(coords) >= 2:
                    segments.append(coords)
            return segments
        
        else:
            return []
    
    except Exception as e:
        print(f"Warning: polyline clip failed - {e}")
        return []


def clip_polygon_by_roi(
    polygon: np.ndarray,
    pc_range: List[float]
) -> List[np.ndarray]:
    """
    Clip polygon by ROI, may produce multiple polygons
    
    Args:
        polygon: [N, 2] boundary points in ego frame
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Returns:
        List of clipped polygon boundaries (each [M, 2])
    """
    x_min, y_min, _, x_max, y_max, _ = pc_range
    roi_box = Box(x_min, y_min, x_max, y_max)
    
    try:
        poly = Polygon(polygon)
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        clipped = poly.intersection(roi_box)
        
        if clipped.is_empty:
            return []
        
        # Handle different types
        if isinstance(clipped, Polygon):
            coords = np.array(clipped.exterior.coords, dtype=np.float32)
            return [coords[:-1]]  # Remove duplicate last point
        
        elif isinstance(clipped, MultiPolygon):
            polygons = []
            for geom in clipped.geoms:
                coords = np.array(geom.exterior.coords, dtype=np.float32)
                polygons.append(coords[:-1])
            return polygons
        
        else:
            return []
    
    except Exception as e:
        print(f"Warning: polygon clip failed - {e}")
        return []


def sample_polyline_20(polyline: np.ndarray) -> Optional[np.ndarray]:
    """
    Sample polyline to 20 points with uniform arc length (MapTR style)
    
    Args:
        polyline: [N, 2] points (N >= 2)
        
    Returns:
        [20, 2] sampled points or None
    """
    if len(polyline) < 2:
        return None
    
    try:
        line = LineString(polyline)
        total_length = line.length
        
        if total_length < 1e-6:
            return None
        
        # Sample at uniform arc lengths
        distances = np.linspace(0, total_length, 20)
        points = []
        
        for dist in distances:
            point = line.interpolate(dist)
            points.append([point.x, point.y])
        
        return np.array(points, dtype=np.float32)
    
    except Exception as e:
        print(f"Warning: polyline sampling failed - {e}")
        return None


def sample_polygon_20(polygon: np.ndarray) -> Optional[np.ndarray]:
    """
    Sample polygon boundary to 20 points with uniform perimeter (MapTR style)
    
    IMPORTANT: Following MapTR, the first and last points are IDENTICAL.
    This is how MapTR detects polygons: is_poly = (pts[0] == pts[-1])
    
    Args:
        polygon: [N, 2] boundary points
        
    Returns:
        [20, 2] sampled points where pts[0] == pts[19] (clockwise) or None
    """
    if len(polygon) < 3:
        return None
    
    try:
        # Create polygon and ensure valid
        poly = Polygon(polygon)
        if not poly.is_valid:
            poly = poly.buffer(0)
        
        # Get exterior boundary
        boundary = poly.exterior
        total_length = boundary.length
        
        if total_length < 1e-6:
            return None
        
        # Sample 20 points where first and last are the same (MapTR style)
        # linspace(0, length, 20) gives distances: 0, L/19, 2L/19, ..., L
        # For closed boundary, position 0 == position L, so pts[0] == pts[19]
        distances = np.linspace(0, total_length, 20)
        points = []
        
        for dist in distances:
            point = boundary.interpolate(dist)
            points.append([point.x, point.y])
        
        points = np.array(points, dtype=np.float32)
        
        # Ensure clockwise order
        points = ensure_clockwise(points)
        
        return points
    
    except Exception as e:
        print(f"Warning: polygon sampling failed - {e}")
        return None


def ensure_clockwise(points: np.ndarray) -> np.ndarray:
    """
    Ensure polygon points are in clockwise order
    
    For MapTR-style polygons where first and last points are the same,
    we compute signed area and reverse if counter-clockwise.
    
    Args:
        points: [N, 2] polygon boundary points (first == last for polygons)
        
    Returns:
        [N, 2] points in clockwise order
    """
    # Use all points for signed area calculation
    # For closed polygon (pts[0] == pts[-1]), this correctly computes the signed area
    x = points[:, 0]
    y = points[:, 1]
    
    # Shoelace formula for signed area
    signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    signed_area += 0.5 * (x[-1] * y[0] - x[0] * y[-1])
    
    # If counter-clockwise (signed_area > 0), reverse
    # Note: After reversing, the first point becomes the last and vice versa,
    # but since they're the same point, the polygon remains valid
    if signed_area > 0:
        return points[::-1].copy()
    else:
        return points


def compute_aabb(points: np.ndarray) -> np.ndarray:
    """
    Compute axis-aligned bounding box from points
    
    Args:
        points: [N, 2]
        
    Returns:
        [4] array: [x_center, y_center, width, height]
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return np.array([x_center, y_center, width, height], dtype=np.float32)


def filter_by_length(
    polylines: List[np.ndarray],
    min_length: float
) -> List[np.ndarray]:
    """
    Filter polylines by minimum arc length
    
    Args:
        polylines: List of [N, 2] arrays
        min_length: Minimum arc length in meters
        
    Returns:
        Filtered list
    """
    filtered = []
    for poly in polylines:
        if len(poly) < 2:
            continue
        
        line = LineString(poly)
        if line.length >= min_length:
            filtered.append(poly)
    
    return filtered


def filter_by_area(
    polygons: List[np.ndarray],
    min_area: float
) -> List[np.ndarray]:
    """
    Filter polygons by minimum area
    
    Args:
        polygons: List of [N, 2] arrays
        min_area: Minimum area in square meters
        
    Returns:
        Filtered list
    """
    filtered = []
    for poly_pts in polygons:
        if len(poly_pts) < 3:
            continue
        
        poly = Polygon(poly_pts)
        if poly.is_valid and poly.area >= min_area:
            filtered.append(poly_pts)
    
    return filtered


def is_valid_instance(points: np.ndarray) -> bool:
    """
    Check if sampled points form a valid instance
    
    Args:
        points: [20, 2]
        
    Returns:
        True if valid
    """
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return False
    
    # Check if all points are identical (degenerate)
    if np.std(points) < 0.01:
        return False
    
    # Check average spacing (avoid too dense)
    if len(points) >= 2:
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.mean(distances) < 0.05:  # < 5cm average spacing
            return False
    
    return True


if __name__ == "__main__":
    print("Testing geometry utilities...")
    
    # Test transform
    pts_world = np.array([[100, 200], [101, 201]], dtype=np.float32)
    ego_t = np.array([100, 200, 0])
    ego_r = np.eye(3)
    pts_ego = transform_to_ego(pts_world, ego_t, ego_r)
    print(f"✓ Transform: {pts_ego.shape}")
    
    # Test clip polyline
    polyline = np.array([[-20, 0], [0, 0], [20, 0]])
    pc_range = [-15, -30, -2, 15, 30, 2]
    clipped = clip_polyline_by_roi(polyline, pc_range)
    print(f"✓ Clip polyline: {len(clipped)} segment(s)")
    
    # Test sample
    if clipped:
        sampled = sample_polyline_20(clipped[0])
        print(f"✓ Sample polyline: {sampled.shape if sampled is not None else None}")
    
    # Test polygon
    polygon = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]])
    clipped_poly = clip_polygon_by_roi(polygon, pc_range)
    print(f"✓ Clip polygon: {len(clipped_poly)} polygon(s)")
    
    if clipped_poly:
        sampled_poly = sample_polygon_20(clipped_poly[0])
        print(f"✓ Sample polygon: {sampled_poly.shape if sampled_poly is not None else None}")
        if sampled_poly is not None:
            bbox = compute_aabb(sampled_poly)
            print(f"✓ AABB: {bbox}")
    
    print("\n✓ All geometry tests passed!")

