"""Feature extraction module for real-time and windowed GPS data.

Calculates behavioral and geospatial features from a sliding window of fixes:
1. route_deviation_m : Min orthogonal distance from planned bus route polyline.
2. stop_duration_s   : Consecutive duration vehicle has remained stationary (<3 km/h).
3. avg_speed_kmh     : Window mean speed.
4. speed_variance    : Variance of instantaneous speeds across sliding window.
5. delay_ratio       : Elapsed travel time divided by expected schedule time at current progress.
6. is_night          : Binary indicator (1 if hour in [21..5], else 0).
7. zone_risk_score   : Proximity-weighted risk score (0.0 to 1.0) based on known risk zones.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.geo import haversine_distance, point_to_polyline_distance

DATA_DIR = Path(__file__).parent / "data"


def load_risk_zones() -> List[Dict]:
    """Loads risk zones from risk_zones.json."""
    rz_file = DATA_DIR / "risk_zones.json"
    if rz_file.exists():
        with open(rz_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def calculate_zone_risk(lat: float, lon: float, risk_zones: Optional[List[Dict]] = None) -> Tuple[float, Optional[str]]:
    """Calculates risk score based on proximity to high-risk zones.

    If distance <= radius: full severity.
    If radius < distance <= 2*radius: linear decay from severity to 0.
    Else: 0.

    Returns:
        (max_score, zone_name_if_any)
    """
    if risk_zones is None:
        risk_zones = load_risk_zones()

    max_score = 0.0
    matched_zone_name = None

    for rz in risk_zones:
        d = haversine_distance(lat, lon, rz["lat"], rz["lon"])
        radius = float(rz["radius_m"])
        severity = float(rz["severity"])

        if d <= radius:
            score = severity
        elif d <= 2.0 * radius:
            # Linear decay
            decay = 1.0 - ((d - radius) / radius)
            score = severity * decay
        else:
            score = 0.0

        if score > max_score:
            max_score = score
            matched_zone_name = rz["name"]

    return round(float(max_score), 4), matched_zone_name


class FeatureExtractor:
    """Sliding-window feature extractor for GPS point streams."""

    def __init__(self, planned_route_coords: List[Tuple[float, float]], window_size: int = 10):
        self.planned_route_coords = planned_route_coords
        self.window_size = window_size
        self.window_points: List[Dict[str, Any]] = []
        self.stationary_start_time: Optional[float] = None
        self.current_stop_duration_s: float = 0.0
        self.risk_zones = load_risk_zones()

    def add_fix(self, fix: Dict[str, Any]) -> Dict[str, Any]:
        """Ingests a GPS fix and extracts features from the updated window.

        fix dict expected keys:
        - lat: float
        - lon: float
        - timestamp: float (epoch seconds or cumulative seconds)
        - speed_kmh: float (optional, if absent computed from previous fix)
        - elapsed_time_s: float
        - expected_time_s: float
        - hour_of_day: int (0-23, optional, defaults to 14 or wall-clock)
        """
        self.window_points.append(fix)
        if len(self.window_points) > self.window_size:
            self.window_points.pop(0)

        lat = fix["lat"]
        lon = fix["lon"]
        timestamp = fix.get("timestamp", 0.0)

        # 1. Route deviation
        deviation_m, _, closest_pt = point_to_polyline_distance(lat, lon, self.planned_route_coords)

        # 2. Speed metrics in window
        speeds = []
        for p in self.window_points:
            speeds.append(float(p.get("speed_kmh", 0.0)))
        current_speed = speeds[-1] if speeds else 0.0
        avg_speed = float(np.mean(speeds)) if speeds else current_speed
        speed_variance = float(np.var(speeds)) if len(speeds) > 1 else 0.0

        # 3. Stop duration
        if current_speed < 3.0:
            if self.stationary_start_time is None:
                self.stationary_start_time = timestamp
            self.current_stop_duration_s = max(0.0, timestamp - self.stationary_start_time)
        else:
            self.stationary_start_time = None
            self.current_stop_duration_s = 0.0

        # 4. Delay ratio (elapsed / expected time)
        elapsed_s = max(1.0, float(fix.get("elapsed_time_s", 1.0)))
        expected_s = max(1.0, float(fix.get("expected_time_s", 1.0)))
        delay_ratio = elapsed_s / expected_s

        # 5. Night travel indicator
        hour = fix.get("hour_of_day")
        if hour is None:
            hour = datetime.now().hour
        is_night = 1 if (hour >= 21 or hour < 5) else 0

        # 6. Zone risk score
        zone_risk_score, matched_zone = calculate_zone_risk(lat, lon, self.risk_zones)

        features = {
            "route_deviation_m": round(float(deviation_m), 1),
            "stop_duration_s": round(float(self.current_stop_duration_s), 1),
            "avg_speed_kmh": round(float(avg_speed), 1),
            "speed_variance": round(float(speed_variance), 2),
            "delay_ratio": round(float(delay_ratio), 2),
            "is_night": int(is_night),
            "zone_risk_score": round(float(zone_risk_score), 3),
            "closest_route_pt": closest_pt,
            "matched_zone_name": matched_zone,
            "hour_of_day": int(hour)
        }
        return features

    def reset(self):
        """Resets the sliding window state."""
        self.window_points.clear()
        self.stationary_start_time = None
        self.current_stop_duration_s = 0.0


FEATURE_NAMES = [
    "route_deviation_m",
    "stop_duration_s",
    "avg_speed_kmh",
    "speed_variance",
    "delay_ratio",
    "is_night",
    "zone_risk_score"
]


def explain_risk_factors(features: Dict[str, Any], risk_class: int, probabilities: Dict[str, float]) -> List[str]:
    """Generates human-readable explanatory bullet points for why risk was assigned.

    Classes: 0: Safe, 1: Caution, 2: High-Risk
    """
    reasons = []

    dev = features.get("route_deviation_m", 0.0)
    stop = features.get("stop_duration_s", 0.0)
    avg_spd = features.get("avg_speed_kmh", 0.0)
    delay = features.get("delay_ratio", 1.0)
    night = features.get("is_night", 0)
    zone_score = features.get("zone_risk_score", 0.0)
    zone_name = features.get("matched_zone_name")

    if risk_class == 2:  # High-Risk
        if dev >= 350:
            reasons.append(f"Significant route deviation: {dev:.0f} m off planned bus corridor.")
        if stop >= 150:
            reasons.append(f"Prolonged unexpected stop: stationary for {int(stop)} seconds.")
        if zone_score >= 0.5:
            reasons.append(f"Inside high-risk buffer ({zone_score*100:.0f}% risk): {zone_name or 'unlit/isolated zone'}.")
        if night == 1:
            reasons.append(f"Late night transit window (hour: {features.get('hour_of_day', 23)}:00) with elevated vulnerability.")
        if dev >= 200 and stop >= 90:
            reasons.append("Compounded hazard: stopped stationary while deviated off route.")
        if not reasons:
            reasons.append("Multiple compound anomaly thresholds triggered simultaneously.")

    elif risk_class == 1:  # Caution
        if dev >= 120:
            reasons.append(f"Moderate detour/deviation detected ({dev:.0f} m).")
        if stop >= 60:
            reasons.append(f"Unscheduled stop duration ({int(stop)} s).")
        if delay >= 1.3:
            reasons.append(f"Schedule delay ratio {delay:.1f}x nominal travel time.")
        if zone_score > 0.2:
            reasons.append(f"Approaching boundary of caution area ({zone_name or 'monitored zone'}).")
        if night == 1 and (dev > 80 or stop > 45):
            reasons.append("Minor anomalies occurring during night transit hours.")
        if not reasons:
            reasons.append("Minor route or schedule variance requiring passenger confirmation.")

    else:  # Safe
        reasons.append("Vehicle actively following planned route within tolerance (<100m).")
        if avg_spd >= 10:
            reasons.append(f"Consistent transit motion (avg speed {avg_spd:.0f} km/h).")
        if zone_score < 0.2:
            reasons.append("Operating in standard surveillance transit corridor.")

    return reasons
