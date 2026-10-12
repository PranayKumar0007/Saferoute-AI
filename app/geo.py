"""Geospatial calculations from first principles.

Formulas implemented:
1. Haversine Distance:
   Calculates great-circle distance between two coordinates on a spherical Earth (radius R = 6,371,000 m).
   a = sin^2(Δlat / 2) + cos(lat1) * cos(lat2) * sin^2(Δlon / 2)
   c = 2 * atan2(sqrt(a), sqrt(1 - a))
   d = R * c

2. Instantaneous Speed:
   v = (d / Δt) * 3.6  [km/h]

3. Point-to-Segment Orthogonal Projection:
   Local planar equirectangular projection onto segment AB:
   t = clamp( (P - A) · (B - A) / |B - A|^2, 0, 1 )
   Closest Point C = A + t * (B - A)
   Perpendicular Distance = Haversine(P, C)
"""

import math
from typing import Dict, List, Tuple

EARTH_RADIUS_M = 6_371_000.0  # Mean radius of Earth in meters


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance in meters between two lat/lon coordinates.

    Args:
        lat1, lon1: First coordinate in decimal degrees.
        lat2, lon2: Second coordinate in decimal degrees.

    Returns:
        Distance in meters.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_M * c


def calculate_speed_kmh(
    lat1: float, lon1: float, t1: float,
    lat2: float, lon2: float, t2: float
) -> float:
    """Calculates instantaneous speed in km/h from two timestamped fixes.

    Speed = (Distance in meters / delta_t in seconds) * 3.6
    """
    dt = t2 - t1
    if dt <= 0.001:
        return 0.0
    dist_m = haversine_distance(lat1, lon1, lat2, lon2)
    speed_mps = dist_m / dt
    return speed_mps * 3.6


def point_to_segment_distance(
    plat: float, plon: float,
    alat: float, alon: float,
    blat: float, blon: float
) -> Tuple[float, Tuple[float, float], float]:
    """Computes minimum distance from point P to line segment AB using local projection.

    Returns:
        (distance_in_meters, (closest_lat, closest_lon), parameter_t)
    """
    # Degenerate segment case
    seg_len_sq = (blat - alat) ** 2 + (blon - alon) ** 2
    if seg_len_sq < 1e-12:
        return haversine_distance(plat, plon, alat, alon), (alat, alon), 0.0

    # Local equirectangular scaling factor for longitude
    mid_lat = math.radians((alat + blat) / 2.0)
    cos_lat = math.cos(mid_lat)

    # Vector AB and AP in projected Cartesian coordinates (degrees roughly scaled)
    dx = (blon - alon) * cos_lat
    dy = blat - alat
    px = (plon - alon) * cos_lat
    py = plat - alat

    # Scalar projection parameter t clamped to [0, 1]
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        t = 0.0
    else:
        t = (px * dx + py * dy) / denom
        t = max(0.0, min(1.0, t))

    # Nearest point coordinates on segment AB
    clat = alat + t * (blat - alat)
    clon = alon + t * (blon - alon)

    dist_m = haversine_distance(plat, plon, clat, clon)
    return dist_m, (clat, clon), t


def point_to_polyline_distance(
    plat: float, plon: float,
    polyline: List[Tuple[float, float]]
) -> Tuple[float, int, Tuple[float, float]]:
    """Calculates minimum perpendicular distance from point P to any segment in polyline.

    Returns:
        (min_distance_m, closest_segment_index, (closest_lat, closest_lon))
    """
    if not polyline:
        return 0.0, 0, (plat, plon)
    if len(polyline) == 1:
        alat, alon = polyline[0]
        return haversine_distance(plat, plon, alat, alon), 0, (alat, alon)

    min_dist = float("inf")
    best_idx = 0
    best_pt = polyline[0]

    for i in range(len(polyline) - 1):
        alat, alon = polyline[i]
        blat, blon = polyline[i + 1]
        dist_m, closest_pt, _ = point_to_segment_distance(plat, plon, alat, alon, blat, blon)
        if dist_m < min_dist:
            min_dist = dist_m
            best_idx = i
            best_pt = closest_pt

    return min_dist, best_idx, best_pt


def interpolate_route(waypoints: List[Dict], target_points: int = 300) -> List[Dict]:
    """Generates ~target_points smoothly spaced points along a sequence of waypoints.

    Computes cumulative distance and expected travel time assuming ~30 km/h nominal bus speed.
    """
    if len(waypoints) < 2:
        return waypoints

    # Calculate segment lengths
    seg_lengths = []
    total_distance_m = 0.0
    for i in range(len(waypoints) - 1):
        d = haversine_distance(
            waypoints[i]["lat"], waypoints[i]["lon"],
            waypoints[i + 1]["lat"], waypoints[i + 1]["lon"]
        )
        seg_lengths.append(d)
        total_distance_m += d

    if total_distance_m <= 0:
        return waypoints

    step_m = total_distance_m / max(1, target_points - 1)
    nominal_speed_mps = 30.0 / 3.6  # 30 km/h nominal transit speed = ~8.33 m/s

    interpolated = []
    curr_seg_idx = 0
    curr_seg_dist_walked = 0.0
    cum_dist = 0.0

    for idx in range(target_points):
        target_dist = idx * step_m

        while curr_seg_idx < len(seg_lengths) - 1 and (cum_dist + (seg_lengths[curr_seg_idx] - curr_seg_dist_walked)) < target_dist:
            cum_dist += (seg_lengths[curr_seg_idx] - curr_seg_dist_walked)
            curr_seg_idx += 1
            curr_seg_dist_walked = 0.0

        w1 = waypoints[curr_seg_idx]
        w2 = waypoints[curr_seg_idx + 1]
        seg_len = seg_lengths[curr_seg_idx]

        if seg_len > 1e-6:
            fraction = min(1.0, max(0.0, (target_dist - cum_dist) / seg_len))
        else:
            fraction = 0.0

        lat = w1["lat"] + fraction * (w2["lat"] - w1["lat"])
        lon = w1["lon"] + fraction * (w2["lon"] - w1["lon"])
        expected_time_s = target_dist / nominal_speed_mps

        interpolated.append({
            "step": idx,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "cum_dist_m": round(target_dist, 1),
            "expected_time_s": round(expected_time_s, 1),
            "nearest_landmark": w1["name"] if fraction < 0.5 else w2["name"]
        })

    return interpolated


def compute_first_principles_step(
    plat: float, plon: float,
    alat: float, alon: float,
    blat: float, blon: float
) -> Dict:
    """Returns a step-by-step breakdown of the first-principles math for live presentation."""
    # 1. Haversine step between A and B
    phi1 = math.radians(alat)
    phi2 = math.radians(blat)
    dphi = math.radians(blat - alat)
    dlam = math.radians(blon - alon)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlam / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    seg_dist_m = EARTH_RADIUS_M * c

    # 2. Projection step
    dist_m, closest_pt, t = point_to_segment_distance(plat, plon, alat, alon, blat, blon)

    return {
        "inputs": {
            "current_point": {"lat": round(plat, 6), "lon": round(plon, 6)},
            "segment_start": {"lat": round(alat, 6), "lon": round(alon, 6)},
            "segment_end": {"lat": round(blat, 6), "lon": round(blon, 6)}
        },
        "haversine_steps": {
            "delta_lat_rad": round(dphi, 6),
            "delta_lon_rad": round(dlam, 6),
            "chord_length_sq_a": round(a, 8),
            "angular_distance_rad_c": round(c, 8),
            "segment_length_m": round(seg_dist_m, 2)
        },
        "projection_steps": {
            "scalar_projection_t": round(t, 4),
            "closest_point": {"lat": round(closest_pt[0], 6), "lon": round(closest_pt[1], 6)},
            "perpendicular_deviation_m": round(dist_m, 2)
        }
    }
