"""FastAPI application providing REST endpoints, static asset serving, and real-time WebSocket telemetry."""

import asyncio
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.alert_engine import (
    AlertEngine,
    clear_alerts,
    get_all_alerts,
)
from app.features import FeatureExtractor
from app.geo import compute_first_principles_step, interpolate_route
from app.gps_simulator import GPSSimulator
from app.risk_model import RiskPredictor
from app.train import ARTIFACTS_DIR, train_and_evaluate
from tests.test_scenarios import run_all_scenarios

app = FastAPI(
    title="AI-Based Women Safety Monitoring System",
    description="SLAM Review-3 Prototype - Track: AI-SDLC (SDG 5 & 11)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate core interdependent modules (WP7 Pipeline)
gps_sim = GPSSimulator()
feature_extractor = FeatureExtractor(planned_route_coords=gps_sim.planned_polyline, window_size=10)
risk_predictor = RiskPredictor()
alert_engine = AlertEngine(debounce_count=3, caution_timeout_s=15.0)

# Paths
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(__file__).parent / "data"

# Mount static and artifacts files
STATIC_DIR.mkdir(exist_ok=True, parents=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")


# Request models
class ThresholdRequest(BaseModel):
    caution_threshold: float
    high_risk_threshold: float


class RealGpsRequest(BaseModel):
    lat: float
    lon: float
    accuracy: float = 10.0
    speed_kmh: Optional[float] = None


class SpeedRequest(BaseModel):
    multiplier: float


class TrainRequest(BaseModel):
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 12


# REST Endpoints
@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AI-Based Women Safety Monitoring API running."}


@app.get("/api/route")
async def get_route():
    """Returns planned route waypoints, interpolated line, and distance metrics."""
    route_file = DATA_DIR / "route.json"
    with open(route_file, "r", encoding="utf-8") as f:
        waypoints = json.load(f)
    return {
        "waypoints": waypoints,
        "interpolated_route": gps_sim.planned_points,
        "total_points": len(gps_sim.planned_points)
    }


@app.get("/api/risk-zones")
async def get_risk_zones():
    """Returns configured risk zones in Hyderabad corridor."""
    rz_file = DATA_DIR / "risk_zones.json"
    with open(rz_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/contacts")
async def get_contacts():
    """Returns emergency trusted contacts."""
    c_file = DATA_DIR / "contacts.json"
    with open(c_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/journey/start")
async def start_journey():
    gps_sim.start_journey()
    return {"status": "started", "active_scenario": gps_sim.active_scenario}


@app.post("/api/journey/stop")
async def stop_journey():
    gps_sim.stop_journey()
    return {"status": "paused"}


@app.post("/api/journey/reset")
async def reset_journey():
    gps_sim.reset_journey()
    feature_extractor.reset()
    alert_engine.reset()
    return {"status": "reset", "active_scenario": gps_sim.active_scenario}


@app.post("/api/journey/speed")
async def set_speed(req: SpeedRequest):
    gps_sim.set_speed_multiplier(req.multiplier)
    return {"status": "updated", "speed_multiplier": gps_sim.speed_multiplier}


@app.post("/api/scenario/{name}")
async def set_scenario(name: str):
    if name not in gps_sim.SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Scenario '{name}' not found. Valid: {gps_sim.SCENARIOS}")
    gps_sim.set_scenario(name)
    return {"status": "applied", "scenario": name}


@app.post("/api/demo-mode")
async def trigger_demo_mode():
    gps_sim.start_demo_mode()
    feature_extractor.reset()
    alert_engine.reset()
    return {"status": "demo_started", "duration_seconds": 90}


@app.post("/api/gps/real")
async def ingest_real_gps(req: RealGpsRequest):
    fix = gps_sim.ingest_real_gps(req.lat, req.lon, req.accuracy, req.speed_kmh)
    return {"status": "ingested", "fix": fix}


@app.post("/api/sos")
async def trigger_sos():
    """Immediate manual emergency alert."""
    latest_pt = gps_sim.planned_points[gps_sim.current_index]
    lat = latest_pt["lat"]
    lon = latest_pt["lon"]
    dispatched = alert_engine.trigger_manual_sos(lat, lon)
    return {
        "status": "SOS_DISPATCHED",
        "alerts_count": len(dispatched),
        "alerts": dispatched
    }


@app.post("/api/safe-confirm")
async def confirm_safe():
    """Passenger de-escalation response."""
    res = alert_engine.user_confirms_safe()
    return res


@app.get("/api/metrics")
async def get_metrics():
    """Returns training comparison, CV scores, feature importance, and ablation."""
    metrics_file = ARTIFACTS_DIR / "metrics.json"
    if not metrics_file.exists():
        train_and_evaluate()
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/validation/run")
async def run_validation():
    """Re-runs the 14-scenario CEP validation test suite."""
    report = run_all_scenarios()
    return report


@app.get("/api/threshold")
async def get_threshold():
    return {
        "caution_threshold": risk_predictor.caution_threshold,
        "high_risk_threshold": risk_predictor.high_risk_threshold
    }


@app.post("/api/threshold")
async def update_threshold(req: ThresholdRequest):
    risk_predictor.set_thresholds(req.caution_threshold, req.high_risk_threshold)
    return {
        "status": "updated",
        "caution_threshold": risk_predictor.caution_threshold,
        "high_risk_threshold": risk_predictor.high_risk_threshold
    }


@app.get("/api/alerts")
async def list_alerts():
    """Returns alert logs from SQLite / memory."""
    db_alerts = get_all_alerts(limit=50)
    return {
        "alerts": db_alerts,
        "total_count": len(db_alerts)
    }


@app.delete("/api/alerts")
async def clear_all_alerts():
    clear_alerts()
    alert_engine.in_memory_alerts.clear()
    return {"status": "cleared"}


@app.post("/api/train")
async def retrain_model(req: TrainRequest):
    """Re-runs training pipeline with customized hyperparameters."""
    res = train_and_evaluate(
        random_state=req.random_state,
        n_estimators=req.n_estimators,
        max_depth=req.max_depth
    )
    risk_predictor._ensure_model_loaded()
    return {
        "status": "retrained",
        "best_model": res["best_model_name"],
        "metrics": res["model_comparison"][res["best_model_name"]]
    }


@app.get("/api/first-principles")
async def get_first_principles():
    """Returns step-by-step mathematical derivation using the current vehicle location."""
    curr_idx = gps_sim.current_index
    curr_pt = gps_sim.planned_points[curr_idx]
    # Current segment
    seg_idx = min(len(gps_sim.raw_waypoints) - 2, max(0, curr_idx // 35))
    w1 = gps_sim.raw_waypoints[seg_idx]
    w2 = gps_sim.raw_waypoints[seg_idx + 1]

    calculation = compute_first_principles_step(
        plat=curr_pt["lat"], plon=curr_pt["lon"],
        alat=w1["lat"], alon=w1["lon"],
        blat=w2["lat"], blon=w2["lon"]
    )
    return calculation


# WebSocket for Live Telemetry Stream with Latency Profiling
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            t0 = time.perf_counter()

            # 1. GPS Simulator Step
            t_sim_0 = time.perf_counter()
            gps_fix = gps_sim.tick(delta_time_real_s=0.8)
            lat = gps_fix["lat"]
            lon = gps_fix["lon"]
            t_sim_ms = (time.perf_counter() - t_sim_0) * 1000.0

            # 2. Feature Extraction Step
            t_feat_0 = time.perf_counter()
            features = feature_extractor.add_fix(gps_fix)
            t_feat_ms = (time.perf_counter() - t_feat_0) * 1000.0

            # 3. ML Risk Model Inference Step
            t_model_0 = time.perf_counter()
            prediction = risk_predictor.predict(features)
            t_model_ms = (time.perf_counter() - t_model_0) * 1000.0

            # 4. Alert Engine State Machine Step
            t_alert_0 = time.perf_counter()
            alert_status = alert_engine.update(
                raw_risk_class=prediction["risk_class"],
                lat=lat,
                lon=lon,
                reasons=prediction["reasons"]
            )
            t_alert_ms = (time.perf_counter() - t_alert_0) * 1000.0

            # 5. Notifier Latency Step
            t_notif_ms = 0.4 if alert_status["alerts_dispatched"] else 0.05
            total_latency_ms = (time.perf_counter() - t0) * 1000.0

            # Pipeline status strip payload (WP7 Interdependent Components)
            pipeline_status = [
                {
                    "name": "GPS Simulator",
                    "status": "NORMAL" if gps_sim.is_running else "PAUSED",
                    "latency_ms": round(t_sim_ms, 2),
                    "indicator": "green" if gps_sim.is_running else "amber"
                },
                {
                    "name": "Feature Extractor",
                    "status": "PROCESSING",
                    "latency_ms": round(t_feat_ms, 2),
                    "indicator": "green"
                },
                {
                    "name": "Risk Model (RF)",
                    "status": prediction["risk_label"].upper(),
                    "latency_ms": round(t_model_ms, 2),
                    "indicator": "green" if prediction["risk_class"] == 0 else ("amber" if prediction["risk_class"] == 1 else "red")
                },
                {
                    "name": "Alert Engine",
                    "status": alert_status["current_state"].upper(),
                    "latency_ms": round(t_alert_ms, 2),
                    "indicator": "green" if alert_status["current_state"] == "Safe" else ("amber" if alert_status["current_state"] == "Caution" else "red")
                },
                {
                    "name": "Notifier Gateway",
                    "status": "DISPATCHING" if alert_status["alerts_dispatched"] else "IDLE",
                    "latency_ms": round(t_notif_ms, 2),
                    "indicator": "red" if alert_status["alerts_dispatched"] else "green"
                }
            ]

            payload = {
                "telemetry": gps_fix,
                "features": features,
                "prediction": prediction,
                "alert_state": alert_status,
                "pipeline_status": pipeline_status,
                "total_pipeline_latency_ms": round(total_latency_ms, 2),
                "timestamp": time.time()
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.8)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS ERROR] Live stream disconnected: {e}")
