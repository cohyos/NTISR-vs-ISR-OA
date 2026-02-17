"""
Geometry utilities for airborne EO/IR sensor footprint calculations.

Handles slant-range geometry, angular FOV to ground footprint conversion,
and coordinate transforms between platform and ground frames.
"""

import numpy as np

NM_TO_FT = 6076.12
FT_TO_NM = 1.0 / NM_TO_FT
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
KTS_TO_NM_PER_S = 1.0 / 3600.0


def slant_range_ft(altitude_ft: float, ground_range_ft: float) -> float:
    """Compute slant range from altitude and ground range."""
    return np.sqrt(altitude_ft**2 + ground_range_ft**2)


def depression_angle_rad(altitude_ft: float, ground_range_ft: float) -> float:
    """Compute depression angle from platform to ground point."""
    return np.arctan2(altitude_ft, ground_range_ft)


def fov_to_ground_footprint_nm(fov_deg: float, slant_range_nm: float,
                                depression_angle_deg: float) -> float:
    """
    Convert angular FOV to approximate ground footprint width [nm].

    Uses the small-angle projection: footprint ≈ slant_range * fov / cos(dep).
    For larger depression angles the footprint shrinks (more overhead).
    """
    fov_rad = fov_deg * DEG_TO_RAD
    dep_rad = depression_angle_deg * DEG_TO_RAD
    cos_dep = np.cos(dep_rad)
    if cos_dep < 0.01:
        cos_dep = 0.01  # Avoid division by zero near nadir
    return slant_range_nm * fov_rad / cos_dep


def ground_footprint_at_cell(altitude_ft: float, slant_range_nm: float,
                              fov_deg: float) -> float:
    """
    Compute ground footprint width [nm] at the cell center given platform
    altitude, slant range, and sensor FOV.
    """
    alt_nm = altitude_ft * FT_TO_NM
    ground_range_nm = np.sqrt(max(slant_range_nm**2 - alt_nm**2, 0.01))
    dep_deg = np.arctan2(alt_nm, ground_range_nm) * RAD_TO_DEG
    return fov_to_ground_footprint_nm(fov_deg, slant_range_nm, dep_deg)


def gsd_ft(altitude_ft: float, slant_range_nm: float,
           fov_deg: float, sensor_pixels: int = 1024) -> float:
    """
    Estimate ground sample distance [ft] for a sensor with given FOV
    and pixel count at a given slant range.
    """
    footprint_ft = ground_footprint_at_cell(altitude_ft, slant_range_nm,
                                             fov_deg) * NM_TO_FT
    return footprint_ft / sensor_pixels


def point_in_circle(x: float, y: float, cx: float, cy: float,
                    radius: float) -> bool:
    """Check if point (x,y) is inside circle centered at (cx,cy)."""
    return (x - cx)**2 + (y - cy)**2 <= radius**2


def angle_between_points(x1: float, y1: float, x2: float, y2: float) -> float:
    """Angle in radians from point 1 to point 2, measured from North (y-axis)."""
    return np.arctan2(x2 - x1, y2 - y1)


def distance_nm(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in nm between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def footprint_corners(center_x: float, center_y: float,
                      width_nm: float, height_nm: float,
                      rotation_rad: float = 0.0) -> np.ndarray:
    """
    Return 4 corner coordinates of a rectangular footprint on the ground.
    Returns shape (4, 2).
    """
    hw, hh = width_nm / 2, height_nm / 2
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
    rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    rotated = corners @ rot.T
    rotated[:, 0] += center_x
    rotated[:, 1] += center_y
    return rotated


def point_in_rect_footprint(px: float, py: float,
                            center_x: float, center_y: float,
                            width_nm: float, height_nm: float,
                            rotation_rad: float = 0.0) -> bool:
    """
    Check if a point is inside a rotated rectangular footprint.
    Uses inverse rotation to test in the footprint's local frame.
    """
    dx = px - center_x
    dy = py - center_y
    cos_r, sin_r = np.cos(-rotation_rad), np.sin(-rotation_rad)
    local_x = dx * cos_r - dy * sin_r
    local_y = dx * sin_r + dy * cos_r
    return abs(local_x) <= width_nm / 2 and abs(local_y) <= height_nm / 2
