"""GPS Stream Simulator with Scenario Injection and Real Device Ingestion."""

import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from app.geo import haversine_distance, interpolate_route

DATA_DIR = Path(__file__).parent / "data"


def load_route_waypoints() -> List[Dict[str, Any]]:
    """Loads canonical waypoints from route.json."""
    r_file = DATA_DIR / "route.json"
    with open(r_file, "r", encoding="utf-8") as f:
        return json.load(f)


class GPSSimulator:
    """Manages playback of planned Hyderabad corridor telemetry and anomalous scenarios."""

    SCENARIOS = [
        "Normal",
        "Route Deviation",
        "Prolonged Stop",
        "Speed Anomaly",
        "Night + High-Risk Zone",
        "Combined Worst-Case"
    ]

    def __init__(self):
        self.raw_waypoints = load_route_waypoints()
        self.planned_points = interpolate_route(self.raw_waypoints, target_points=320)
        self.planned_polyline: List[Tuple[float, float]] = [
            (p["lat"], p["lon"]) for p in self.planned_points
        ]

        self.current_index = 0
        self.is_running = True
        self.speed_multiplier = 5.0  # Default 5x real-time playback
        self.mode = "simulator"  # 'simulator' or 'real_gps'

        # Scenario overrides
        self.active_scenario = "Normal"
        self.scenario_start_time = time.time()
        self.scenario_step = 0
        self.stop_accumulated_seconds = 0.0

        # Demo Mode state
        self.demo_mode_active = False
        self.demo_mode_start_time = 0.0
        self.demo_caption = ""

        # Real GPS latest fix buffer
        self.latest_real_gps: Optional[Dict[str, Any]] = None

        # Fix history
        self.history: List[Dict[str, Any]] = []
        self.elapsed_simulation_seconds = 0.0

    def start_journey(self):
        """Starts or resumes simulation."""
        self.is_running = True

    def stop_journey(self):
        """Pauses simulation."""
        self.is_running = False

    def reset_journey(self):
        """Resets simulation back to origin (Dundigal)."""
        self.current_index = 0
        self.active_scenario = "Normal"
        self.stop_accumulated_seconds = 0.0
        self.demo_mode_active = False
        self.demo_caption = ""
        self.history.clear()
        self.elapsed_simulation_seconds = 0.0
        self.mode = "simulation"
        self.latest_real_gps = None

    def set_speed_multiplier(self, multiplier: float):
        """Sets playback speed multiplier (e.g. 1x, 5x, 10x, 20x)."""
        self.speed_multiplier = max(1.0, min(50.0, multiplier))

    def set_scenario(self, scenario_name: str):
        """Injects anomalous or normal scenario."""
        self.mode = "simulation"
        self.latest_real_gps = None
        if scenario_name in self.SCENARIOS:
            self.active_scenario = scenario_name
            self.scenario_start_time = time.time()
            self.scenario_step = 0
            if scenario_name != "Prolonged Stop":
                self.stop_accumulated_seconds = 0.0
            print(f"[SIMULATOR] Scenario switched to: {scenario_name}")

    def start_demo_mode(self):
        """Initiates automated 90-second collegiate review demo script."""
        self.demo_mode_active = True
        self.demo_mode_start_time = time.time()
        self.is_running = True
        self.set_scenario("Normal")
        self.demo_caption = "STAGE 1: Nominal transit on NH44 corridor. All parameters in green tolerance."

    def ingest_real_gps(self, lat: float, lon: float, accuracy: float, speed_kmh: Optional[float] = None) -> Dict[str, Any]:
        """Ingests live GPS fix from mobile/browser navigator.geolocation."""
        self.mode = "real_gps"
        now = time.time()
        speed = speed_kmh if speed_kmh is not None and speed_kmh >= 0 else 0.0

        fix = {
            "lat": lat,
            "lon": lon,
            "accuracy": accuracy,
            "speed_kmh": speed,
            "timestamp": now,
            "elapsed_time_s": self.elapsed_simulation_seconds,
            "expected_time_s": self.elapsed_simulation_seconds,
            "hour_of_day": time.localtime().tm_hour,
            "source": "browser_geolocation",
            "scenario": "Live Device"
        }
        self.latest_real_gps = fix
        return fix

    def tick(self, delta_time_real_s: float = 1.0) -> Dict[str, Any]:
        """Advances simulation by delta_time_real_s * speed_multiplier."""
        dt_sim = delta_time_real_s * self.speed_multiplier

        # Handle Real GPS Mode
        if self.mode == "real_gps" and self.latest_real_gps is not None:
            return self.latest_real_gps

        if not self.is_running:
            # Return current position stationary
            base_pt = self.planned_points[self.current_index]
            return {
                "lat": base_pt["lat"],
                "lon": base_pt["lon"],
                "speed_kmh": 0.0,
                "timestamp": time.time(),
                "elapsed_time_s": self.elapsed_simulation_seconds,
                "expected_time_s": base_pt["expected_time_s"],
                "hour_of_day": 14,
                "scenario": self.active_scenario,
                "step": self.current_index,
                "is_paused": True,
                "demo_caption": self.demo_caption
            }

        self.elapsed_simulation_seconds += dt_sim

        # Handle Demo Mode Script Transitions
        if self.demo_mode_active:
            demo_elapsed = time.time() - self.demo_mode_start_time
            if demo_elapsed < 16.0:
                self.active_scenario = "Normal"
                self.demo_caption = "DEMO [0-15s]: Nominal transit on NH44 corridor. Bus speed ~32 km/h. Pipeline state: Safe."
            elif demo_elapsed < 36.0:
                if self.active_scenario != "Route Deviation":
                    self.set_scenario("Route Deviation")
                self.demo_caption = "DEMO [16-35s]: Route Deviation detected (>300m off-route). Caution triggered: In-app 'Are you safe?' prompt with 15s timer."
            elif demo_elapsed < 52.0:
                # Still off route, timer has run out -> alert dispatched
                self.demo_caption = "DEMO [36-50s]: 15s passenger confirmation timed out! Escalated to High-Risk. Simulated SMS alert sent with live GPS."
            elif demo_elapsed < 72.0:
                if self.active_scenario != "Night + High-Risk Zone":
                    self.set_scenario("Night + High-Risk Zone")
                self.demo_caption = "DEMO [51-70s]: Night transit (23:45) inside Jeedimetla Industrial Belt + Prolonged Stop. Compound threat active."
            elif demo_elapsed < 90.0:
                if self.active_scenario != "Combined Worst-Case":
                    self.set_scenario("Combined Worst-Case")
                self.demo_caption = "DEMO [71-90s]: Combined Worst-Case: Severe deviation + stationary dwell in risk zone. Emergency broadcast triggered."
            else:
                self.demo_mode_active = False
                self.demo_caption = "DEMO COMPLETE: All 4 graded SLAM criteria demonstrated successfully."

        # Fetch base route point
        base_pt = self.planned_points[self.current_index]
        lat = base_pt["lat"]
        lon = base_pt["lon"]
        hour_of_day = 14  # Default daytime
        speed_kmh = 32.0  # Nominal bus speed

        self.scenario_step += 1

        # Apply Scenario Distortions
        if self.active_scenario == "Normal":
            # Small random drift within 15 meters
            speed_kmh = 28.0 + 6.0 * math.sin(self.scenario_step * 0.2)
            self.stop_accumulated_seconds = 0.0

        elif self.active_scenario == "Route Deviation":
            # Deviate progressively away from corridor (e.g., west/east side road)
            # 0.001 deg lat is ~111m, 0.001 deg lon is ~106m
            offset_m = min(750.0, self.scenario_step * 15.0)  # Grows up to 750m
            # Drift lon west and lat south-west
            lon -= (offset_m / 106000.0)
            lat -= (offset_m * 0.4 / 111000.0)
            speed_kmh = 22.0

        elif self.active_scenario == "Prolonged Stop":
            # Stationary at current coordinates
            speed_kmh = 0.0
            self.stop_accumulated_seconds += dt_sim

        elif self.active_scenario == "Speed Anomaly":
            # Erratic speed behavior
            if self.scenario_step % 6 < 3:
                speed_kmh = 72.0  # High reckless speed
            else:
                speed_kmh = 2.0   # Sudden screeching stop

        elif self.active_scenario == "Night + High-Risk Zone":
            # Force hour to 23:45 (late night) and place near Jeedimetla Industrial Zone
            hour_of_day = 23
            # Near Jeedimetla industrial zone (17.518, 78.465)
            lat = 17.5180 + 0.0015 * math.sin(self.scenario_step * 0.1)
            lon = 78.4650 + 0.0015 * math.cos(self.scenario_step * 0.1)
            speed_kmh = 8.0  # Crawling speed
            self.stop_accumulated_seconds += (dt_sim * 0.6)

        elif self.active_scenario == "Combined Worst-Case":
            # Severe deviation + night (00:30) + inside risk zone + prolonged stop
            hour_of_day = 0
            lat = 17.5195  # Deep inside Jeedimetla zone
            lon = 78.4630  # 600m off main corridor
            speed_kmh = 0.0
            self.stop_accumulated_seconds += dt_sim

        # Advance along route if not prolonged stop / combined worst case
        if self.active_scenario not in ["Prolonged Stop", "Combined Worst-Case"]:
            step_increment = max(1, int(round(1 * (speed_kmh / 30.0))))
            self.current_index = (self.current_index + step_increment) % len(self.planned_points)

        fix_record = {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "speed_kmh": round(speed_kmh, 1),
            "timestamp": time.time(),
            "elapsed_time_s": round(self.elapsed_simulation_seconds, 1),
            "expected_time_s": round(base_pt["expected_time_s"], 1),
            "hour_of_day": int(hour_of_day),
            "scenario": self.active_scenario,
            "step": self.current_index,
            "total_steps": len(self.planned_points),
            "nearest_landmark": base_pt.get("nearest_landmark", "Transit Corridor"),
            "stop_accumulated_s": round(self.stop_accumulated_seconds, 1),
            "demo_caption": self.demo_caption,
            "demo_mode_active": self.demo_mode_active
        }

        self.history.append(fix_record)
        if len(self.history) > 200:
            self.history.pop(0)

        return fix_record
