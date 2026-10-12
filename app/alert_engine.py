"""Alert Engine & State Machine with Debounce, Escalation, and Pluggable Notifiers.

State Machine transitions:
- SAFE: Normal state, no alerts sent.
- CAUTION: In-app prompt 'Are you safe?' triggered with 15-second countdown.
  If confirmed safe -> reset to SAFE.
  If 15s timer expires without confirmation -> escalate to HIGH_RISK.
- HIGH_RISK: Dispatches simulated SMS to emergency trusted contacts with
  Google Maps live location link and detected risk reasons.
- MANUAL SOS: Immediate escalation to HIGH_RISK regardless of debouncer.

Debounce/Hysteresis:
Requires N (default 3) consecutive raw classification windows before changing
system state to prevent false transient flickering.
"""

from abc import ABC, abstractmethod
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = Path(__file__).parent.parent / "artifacts" / "safety_alerts.db"


class BaseNotifier(ABC):
    """Abstract interface for dispatching emergency notifications."""

    @abstractmethod
    def send_alert(self, contact: Dict[str, Any], alert_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an alert to a single contact."""
        pass


class MockSmsNotifier(BaseNotifier):
    """Simulates carrier SMS gateway. Writes sent messages to database and in-memory log."""

    def send_alert(self, contact: Dict[str, Any], alert_payload: Dict[str, Any]) -> Dict[str, Any]:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        phone = contact.get("phone", "N/A")
        name = contact.get("name", "Unknown Contact")
        lat = alert_payload.get("lat", 0.0)
        lon = alert_payload.get("lon", 0.0)
        maps_link = f"https://www.google.com/maps?q={lat:.5f},{lon:.5f}"
        reasons_summary = "; ".join(alert_payload.get("reasons", ["Anomaly detected"]))

        sms_body = (
            f"[EMERGENCY ALERT - TG SAFETY] Attention {name}: "
            f"Passenger distress / high risk detected on route. "
            f"Reasons: {reasons_summary}. "
            f"Live GPS Location: {maps_link} at {timestamp_str}."
        )

        record = {
            "contact_id": contact.get("id"),
            "contact_name": name,
            "phone": phone,
            "relation": contact.get("relation", "Contact"),
            "body": sms_body,
            "maps_link": maps_link,
            "risk_level": alert_payload.get("risk_level", "High-Risk"),
            "timestamp": timestamp_str,
            "status": "DELIVERED (SIMULATED)"
        }
        return record


class TwilioSmsNotifierStub(BaseNotifier):
    """Production stub for Twilio Programmable SMS API integration."""

    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None):
        self.account_sid = account_sid
        self.auth_token = auth_token

    def send_alert(self, contact: Dict[str, Any], alert_payload: Dict[str, Any]) -> Dict[str, Any]:
        # Ready for production: twilio.rest.Client(account_sid, auth_token).messages.create(...)
        return {
            "status": "STUB_NOT_CONFIGURED",
            "message": "Twilio credentials not set in prototype mode. Dispatched via MockSmsNotifier."
        }


def init_db(db_path: Path = DB_PATH):
    """Initializes SQLite database for alerts and journey records."""
    db_path.parent.mkdir(exist_ok=True, parents=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                contact_name TEXT,
                phone TEXT,
                relation TEXT,
                risk_level TEXT,
                body TEXT,
                maps_link TEXT,
                lat REAL,
                lon REAL,
                status TEXT
            )
        """)
        conn.commit()


def log_alert_to_db(record: Dict[str, Any], lat: float, lon: float, db_path: Path = DB_PATH):
    """Persists an alert record to SQLite."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (timestamp, contact_name, phone, relation, risk_level, body, maps_link, lat, lon, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("timestamp"),
                record.get("contact_name"),
                record.get("phone"),
                record.get("relation"),
                record.get("risk_level"),
                record.get("body"),
                record.get("maps_link"),
                lat,
                lon,
                record.get("status")
            ))
            conn.commit()
    except Exception as e:
        print(f"[DB ERROR] Failed to log alert: {e}")


def get_all_alerts(limit: int = 50, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Fetches recent alert logs."""
    init_db(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"[DB ERROR] Could not read alerts: {e}")
        return []


def clear_alerts(db_path: Path = DB_PATH):
    """Clears alerts database for clean runs."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM alerts")
        conn.commit()


class AlertEngine:
    """State machine governing safety assessments, debounce, and escalation."""

    STATE_SAFE = "Safe"
    STATE_CAUTION = "Caution"
    STATE_HIGH_RISK = "High-Risk"

    def __init__(self, debounce_count: int = 3, caution_timeout_s: float = 15.0):
        self.debounce_count = debounce_count
        self.caution_timeout_s = caution_timeout_s

        self.current_state = self.STATE_SAFE
        self.raw_history: List[int] = []  # Recent raw model class predictions
        self.caution_entered_at: Optional[float] = None
        self.caution_acknowledged: bool = False
        self.alerts_sent_for_incident: bool = False

        self.notifier: BaseNotifier = MockSmsNotifier()
        self.contacts = self._load_contacts()
        self.in_memory_alerts: List[Dict[str, Any]] = []
        init_db()

    def _load_contacts(self) -> List[Dict[str, Any]]:
        c_path = DATA_DIR / "contacts.json"
        if c_path.exists():
            with open(c_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def set_notifier(self, notifier: BaseNotifier):
        """Allows swapping the notification backend (e.g., Twilio)."""
        self.notifier = notifier

    def user_confirms_safe(self) -> Dict[str, Any]:
        """Passenger taps 'I am Safe' in UI to de-escalate."""
        self.caution_acknowledged = True
        self.current_state = self.STATE_SAFE
        self.caution_entered_at = None
        self.alerts_sent_for_incident = False
        return {
            "status": "ACKNOWLEDGED",
            "message": "Passenger confirmed safety. Alert timer cancelled.",
            "current_state": self.current_state
        }

    def trigger_manual_sos(self, lat: float, lon: float, reasons: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Manual SOS bypasses debouncing and immediately dispatches alerts."""
        self.current_state = self.STATE_HIGH_RISK
        sos_reasons = ["EMERGENCY SOS BUTTON ACTIVATED BY PASSENGER"]
        if reasons:
            sos_reasons.extend(reasons)

        return self._dispatch_emergency_alerts(lat, lon, sos_reasons, is_sos=True)

    def update(
        self,
        raw_risk_class: int,
        lat: float,
        lon: float,
        reasons: List[str],
        now_ts: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs the state machine tick with debounce and countdown escalation.

        raw_risk_class: 0 = Safe, 1 = Caution, 2 = High-Risk
        """
        now = now_ts or time.time()

        # Update debounce window
        self.raw_history.append(raw_risk_class)
        if len(self.raw_history) > self.debounce_count:
            self.raw_history.pop(0)

        # Check if last N classes are consistent (debounce criteria)
        is_debounced = len(self.raw_history) >= self.debounce_count and len(set(self.raw_history)) == 1
        debounced_class = self.raw_history[-1] if is_debounced else None

        # State transitions
        new_alerts = []
        countdown_remaining = None

        # 1. High-Risk condition: debounced as class 2
        if debounced_class == 2:
            if self.current_state != self.STATE_HIGH_RISK:
                self.current_state = self.STATE_HIGH_RISK
                self.caution_entered_at = None
                if not self.alerts_sent_for_incident:
                    new_alerts = self._dispatch_emergency_alerts(lat, lon, reasons)
                    self.alerts_sent_for_incident = True

        # 2. Caution condition: debounced as class 1
        elif debounced_class == 1:
            if self.current_state == self.STATE_SAFE:
                self.current_state = self.STATE_CAUTION
                self.caution_entered_at = now
                self.caution_acknowledged = False
                self.alerts_sent_for_incident = False

        # 3. Safe condition: debounced as class 0
        elif debounced_class == 0:
            if self.current_state != self.STATE_SAFE and self.caution_entered_at is None:
                self.current_state = self.STATE_SAFE
                self.alerts_sent_for_incident = False

        # Check Caution Countdown timer if currently in Caution state
        if self.current_state == self.STATE_CAUTION and self.caution_entered_at is not None:
            elapsed = now - self.caution_entered_at
            countdown_remaining = max(0.0, round(self.caution_timeout_s - elapsed, 1))

            if countdown_remaining <= 0.0 and not self.caution_acknowledged:
                # Timer expired without response -> ESCALATE TO HIGH RISK
                self.current_state = self.STATE_HIGH_RISK
                self.caution_entered_at = None
                escalation_reasons = ["Caution timeout expired (15s): Passenger did not respond to safety check."] + reasons
                if not self.alerts_sent_for_incident:
                    new_alerts = self._dispatch_emergency_alerts(lat, lon, escalation_reasons)
                    self.alerts_sent_for_incident = True

        return {
            "current_state": self.current_state,
            "raw_class": raw_risk_class,
            "debounced_class": debounced_class,
            "is_debounced": is_debounced,
            "countdown_remaining_s": countdown_remaining,
            "alerts_dispatched": new_alerts,
            "total_alerts_count": len(self.in_memory_alerts)
        }

    def _dispatch_emergency_alerts(
        self,
        lat: float,
        lon: float,
        reasons: List[str],
        is_sos: bool = False
    ) -> List[Dict[str, Any]]:
        """Dispatches simulated messages to all configured contacts."""
        dispatched = []
        payload = {
            "lat": lat,
            "lon": lon,
            "reasons": reasons,
            "risk_level": "EMERGENCY SOS" if is_sos else "High-Risk"
        }

        for contact in self.contacts:
            record = self.notifier.send_alert(contact, payload)
            log_alert_to_db(record, lat, lon)
            self.in_memory_alerts.insert(0, record)
            dispatched.append(record)

        return dispatched

    def reset(self):
        """Resets engine state for fresh scenario run."""
        self.current_state = self.STATE_SAFE
        self.raw_history.clear()
        self.caution_entered_at = None
        self.caution_acknowledged = False
        self.alerts_sent_for_incident = False
