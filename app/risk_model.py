"""Risk prediction wrapper with explainability and sensitivity controls."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
import joblib
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from app.features import FEATURE_NAMES, explain_risk_factors
from app.train import ARTIFACTS_DIR, CLASS_LABELS, train_and_evaluate


class RiskPredictor:
    """Inference wrapper for trained Random Forest safety risk model."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (ARTIFACTS_DIR / "model.joblib")
        self.model = None
        self.feature_names = FEATURE_NAMES
        self.caution_threshold = 0.38
        self.high_risk_threshold = 0.32
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Loads model from disk, or triggers training if not yet built."""
        if not self.model_path.exists():
            print(f"[MODEL] Model artifact not found at {self.model_path}. Training initial baseline...")
            train_and_evaluate()

        try:
            self.model = joblib.load(self.model_path)
            # Force n_jobs=1 to avoid thread-spawn overhead on single-row inference
            if hasattr(self.model, 'n_jobs'):
                self.model.n_jobs = 1
            print(f"[MODEL] Successfully loaded Random Forest model from {self.model_path}")
            self._warmup()
        except Exception as e:
            print(f"[MODEL] Error loading model ({e}). Retraining...")
            train_and_evaluate()
            self.model = joblib.load(self.model_path)
            if hasattr(self.model, 'n_jobs'):
                self.model.n_jobs = 1
            self._warmup()

    def _warmup(self):
        """Run one prediction to pre-compile the prediction path (eliminates first-call JIT overhead)."""
        try:
            dummy = np.zeros((1, len(self.feature_names)), dtype=np.float64)
            self.model.predict_proba(dummy)
            print("[MODEL] Warm-up prediction complete. Inference path ready.")
        except Exception as e:
            print(f"[MODEL] Warm-up failed (non-fatal): {e}")

    def set_thresholds(self, caution_thresh: float, high_risk_thresh: float):
        """Adjusts alert sensitivity thresholds (WP2 Conflicting Requirements)."""
        self.caution_threshold = max(0.10, min(0.90, caution_thresh))
        self.high_risk_threshold = max(0.15, min(0.95, high_risk_thresh))

    def predict(
        self,
        features: Dict[str, Any],
        caution_override: Optional[float] = None,
        high_risk_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """Predicts risk class, probabilities, top contributing features, and human-readable reasons."""
        if self.model is None:
            self._ensure_model_loaded()

        c_thresh = caution_override if caution_override is not None else self.caution_threshold
        h_thresh = high_risk_override if high_risk_override is not None else self.high_risk_threshold

        # Build numpy array directly — avoids pandas import + DataFrame overhead per call
        vector = np.array(
            [[float(features.get(f, 0.0)) for f in self.feature_names]],
            dtype=np.float64
        )

        probs = self.model.predict_proba(vector)[0]
        prob_dict = {
            "Safe": round(float(probs[0]), 3),
            "Caution": round(float(probs[1]), 3),
            "High-Risk": round(float(probs[2]), 3)
        }

        # Apply safety-biased threshold logic
        # If high-risk probability exceeds threshold -> classify as High-Risk
        # Else if combined risk (Caution + High-Risk) exceeds caution threshold -> Caution
        # Else -> Safe
        p_safe, p_caution, p_high = probs[0], probs[1], probs[2]

        if p_high >= h_thresh:
            pred_class = 2
        elif (p_caution + p_high) >= c_thresh:
            pred_class = 1
        else:
            pred_class = 0

        pred_label = CLASS_LABELS[pred_class]

        # Calculate feature contributions (feature value * importance heuristic)
        importances = getattr(self.model, "feature_importances_", np.ones(len(self.feature_names)))
        flat_vector = vector[0]
        contributions = []
        for name, val, imp in zip(self.feature_names, flat_vector, importances):
            # Normalized impact heuristic
            contributions.append({
                "feature": name,
                "value": float(val),
                "importance_weight": round(float(imp), 4),
                "impact_score": round(float(abs(val) * imp), 4)
            })

        contributions.sort(key=lambda x: x["impact_score"], reverse=True)
        top_features = contributions[:3]

        reasons = explain_risk_factors(features, pred_class, prob_dict)

        return {
            "risk_class": pred_class,
            "risk_label": pred_label,
            "probabilities": prob_dict,
            "top_features": top_features,
            "reasons": reasons,
            "thresholds_used": {
                "caution_threshold": c_thresh,
                "high_risk_threshold": h_thresh
            }
        }
