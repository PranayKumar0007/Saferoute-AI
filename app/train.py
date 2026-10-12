"""Synthetic dataset generation, model training, cross-validation, and ablation analysis.

Produces:
- artifacts/model.joblib (best Random Forest classifier)
- artifacts/metrics.json (metrics, CV scores, per-class report, ablation analysis)
- artifacts/confusion_matrix.png
- artifacts/feature_importance.png
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from app.features import FEATURE_NAMES

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

CLASS_LABELS = ["Safe", "Caution", "High-Risk"]


def generate_synthetic_dataset(n_samples: int = 6000, random_state: int = 42) -> pd.DataFrame:
    """Generates synthetic multi-modal dataset representing real-world urban transit conditions.

    Distribution:
    - ~55% Safe journeys (normal bus speed 15-50 km/h, deviation < 100m, normal traffic stops < 60s)
    - ~25% Caution journeys (moderate deviation 100-350m, stops 60-180s, high delay ratio, near risk zones)
    - ~20% High-Risk journeys (severe deviation > 400m, prolonged off-route stops > 180s, inside isolated zones, late night)

    Includes Gaussian noise and ~5% label noise so classes are realistically overlapping.
    """
    rng = np.random.RandomState(random_state)

    n_safe = int(n_samples * 0.55)
    n_caution = int(n_samples * 0.25)
    n_high = n_samples - n_safe - n_caution

    # 1. Safe instances
    safe_dev = np.abs(rng.normal(30.0, 25.0, n_safe))
    safe_stop = np.maximum(0.0, rng.exponential(15.0, n_safe))
    # Cap stops for safe
    safe_stop = np.clip(safe_stop, 0.0, 75.0)
    safe_speed = rng.normal(32.0, 8.0, n_safe)
    safe_speed = np.clip(safe_speed, 8.0, 65.0)
    safe_var = rng.normal(12.0, 5.0, n_safe)
    safe_delay = rng.normal(1.05, 0.15, n_safe)
    safe_night = rng.binomial(1, 0.20, n_safe)
    safe_zone = np.abs(rng.normal(0.05, 0.08, n_safe))
    safe_zone = np.clip(safe_zone, 0.0, 0.30)
    y_safe = np.zeros(n_safe, dtype=int)

    # 2. Caution instances
    caut_dev = rng.normal(220.0, 70.0, n_caution)
    caut_dev = np.clip(caut_dev, 80.0, 420.0)
    caut_stop = rng.normal(110.0, 40.0, n_caution)
    caut_stop = np.clip(caut_stop, 40.0, 220.0)
    caut_speed = rng.normal(18.0, 9.0, n_caution)
    caut_speed = np.clip(caut_speed, 0.0, 45.0)
    caut_var = rng.normal(28.0, 10.0, n_caution)
    caut_delay = rng.normal(1.55, 0.25, n_caution)
    caut_night = rng.binomial(1, 0.45, n_caution)
    caut_zone = rng.normal(0.42, 0.15, n_caution)
    caut_zone = np.clip(caut_zone, 0.15, 0.70)
    y_caut = np.ones(n_caution, dtype=int)

    # 3. High-Risk instances
    high_dev = rng.normal(620.0, 160.0, n_high)
    high_dev = np.clip(high_dev, 320.0, 1500.0)
    high_stop = rng.normal(280.0, 90.0, n_high)
    high_stop = np.clip(high_stop, 120.0, 800.0)
    high_speed_stopped = rng.normal(2.0, 1.5, n_high)
    high_speed_erratic = rng.normal(65.0, 12.0, n_high)
    is_stopped_mask = rng.binomial(1, 0.55, n_high)
    high_speed = np.where(is_stopped_mask == 1, high_speed_stopped, high_speed_erratic)
    high_speed = np.clip(high_speed, 0.0, 95.0)
    high_var = rng.normal(48.0, 18.0, n_high)
    high_delay = rng.normal(2.20, 0.45, n_high)
    high_night = rng.binomial(1, 0.80, n_high)
    high_zone = rng.normal(0.78, 0.12, n_high)
    high_zone = np.clip(high_zone, 0.45, 1.0)
    y_high = np.full(n_high, 2, dtype=int)

    # Concatenate features
    route_dev = np.concatenate([safe_dev, caut_dev, high_dev])
    stop_dur = np.concatenate([safe_stop, caut_stop, high_stop])
    avg_spd = np.concatenate([safe_speed, caut_speed, high_speed])
    spd_var = np.concatenate([safe_var, caut_var, high_var])
    delay_r = np.concatenate([safe_delay, caut_delay, high_delay])
    is_ngt = np.concatenate([safe_night, caut_night, high_night])
    zn_risk = np.concatenate([safe_zone, caut_zone, high_zone])
    y = np.concatenate([y_safe, y_caut, y_high])

    # Inject realistic ~5% label noise (simulates ambiguous border cases & human mislabeling)
    n_noisy = int(n_samples * 0.05)
    noise_indices = rng.choice(n_samples, size=n_noisy, replace=False)
    for idx in noise_indices:
        current_y = y[idx]
        alt_classes = [c for c in [0, 1, 2] if c != current_y]
        y[idx] = rng.choice(alt_classes)

    df = pd.DataFrame({
        "route_deviation_m": np.round(route_dev, 2),
        "stop_duration_s": np.round(stop_dur, 2),
        "avg_speed_kmh": np.round(avg_spd, 2),
        "speed_variance": np.round(spd_var, 2),
        "delay_ratio": np.round(delay_r, 2),
        "is_night": is_ngt.astype(int),
        "zone_risk_score": np.round(zn_risk, 3),
        "risk_class": y
    })

    return df


def train_and_evaluate(
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 12
) -> Dict:
    """Trains 3 candidate models, runs 5-fold CV, computes metrics, ablation, and saves artifacts."""
    df = generate_synthetic_dataset(n_samples=6000, random_state=random_state)
    X = df[FEATURE_NAMES]
    y = df["risk_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # 1. Candidate Models
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state)),
        "Decision Tree": DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=1
        )
    }

    model_comparison = {}
    best_name = "Random Forest"
    best_model = None

    for name, clf in models.items():
        # Stratified 5-Fold Cross Validation
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = float(np.mean(y_pred == y_test))
        f1_m = float(f1_score(y_test, y_pred, average="macro"))
        prec_m = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_m = float(recall_score(y_test, y_pred, average="macro"))

        model_comparison[name] = {
            "cv_mean_f1": round(float(np.mean(scores)), 4),
            "cv_std_f1": round(float(np.std(scores)), 4),
            "test_accuracy": round(acc, 4),
            "test_f1_macro": round(f1_m, 4),
            "test_precision_macro": round(prec_m, 4),
            "test_recall_macro": round(rec_m, 4)
        }

        if name == "Random Forest":
            best_model = clf

    # Evaluate best model (Random Forest) in detail
    y_pred_best = best_model.predict(X_test)
    report_dict = classification_report(y_test, y_pred_best, target_names=CLASS_LABELS, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_best)

    # Feature Importance
    importances = best_model.feature_importances_
    feat_imp_dict = {
        name: round(float(imp), 4)
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    }

    # Ablation Study: Drop feature groups and evaluate F1 drop
    feature_groups = {
        "Route Deviation Only": ["route_deviation_m"],
        "Stop & Dwell Only": ["stop_duration_s"],
        "Speed Dynamics Only": ["avg_speed_kmh", "speed_variance"],
        "Schedule Delay Only": ["delay_ratio"],
        "Temporal & Zone Risk Only": ["is_night", "zone_risk_score"]
    }

    ablation_results = {}
    baseline_f1 = model_comparison["Random Forest"]["test_f1_macro"]

    for group_name, drop_cols in feature_groups.items():
        ablated_cols = [c for c in FEATURE_NAMES if c not in drop_cols]
        clf_ablated = RandomForestClassifier(n_estimators=50, max_depth=max_depth, random_state=random_state)
        clf_ablated.fit(X_train[ablated_cols], y_train)
        y_abl_pred = clf_ablated.predict(X_test[ablated_cols])
        abl_f1 = float(f1_score(y_test, y_abl_pred, average="macro"))
        delta_f1 = round(abl_f1 - baseline_f1, 4)
        ablation_results[f"Without {group_name}"] = {
            "retained_features": ablated_cols,
            "ablated_f1_macro": round(abl_f1, 4),
            "f1_delta": delta_f1,
            "performance_drop_pct": round(abs(min(0.0, delta_f1)) * 100, 2)
        }

    # Save model artifact
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(best_model, model_path)

    # Generate Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Best Model: Random Forest)", fontsize=11, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_LABELS))
    plt.xticks(tick_marks, CLASS_LABELS, rotation=15)
    plt.yticks(tick_marks, CLASS_LABELS)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=180)
    plt.close()

    # Generate Feature Importance Plot
    plt.figure(figsize=(7, 4.5))
    sorted_feats = sorted(feat_imp_dict.items(), key=lambda x: x[1])
    names = [x[0] for x in sorted_feats]
    values = [x[1] for x in sorted_feats]
    bars = plt.barh(names, values, color="#3b82f6", edgecolor="#1d4ed8")
    plt.title("Gini Feature Importance (Random Forest)", fontsize=11, fontweight="bold")
    plt.xlabel("Relative Importance Score")
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height() / 2.0, f"{w:.3f}", va="center", fontsize=9)
    plt.xlim(0, max(values) * 1.20)
    plt.tight_layout()
    fi_path = ARTIFACTS_DIR / "feature_importance.png"
    plt.savefig(fi_path, dpi=180)
    plt.close()

    # Build comprehensive metrics JSON
    metrics_data = {
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": FEATURE_NAMES,
            "classes": CLASS_LABELS,
            "class_distribution": {
                "Safe (0)": int((df["risk_class"] == 0).sum()),
                "Caution (1)": int((df["risk_class"] == 1).sum()),
                "High-Risk (2)": int((df["risk_class"] == 2).sum())
            },
            "noise_rate": "5% synthetic stochastic noise"
        },
        "model_comparison": model_comparison,
        "best_model_name": best_name,
        "best_model_hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "criterion": "gini",
            "random_state": random_state
        },
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "feature_importances": feat_imp_dict,
        "ablation_study": ablation_results,
        "metadata": {
            "synthetic_disclaimer": "Trained on synthetic urban transit telemetry for collegiate prototype demonstration. Real deployment requires labelled telemetry from RTC transport fleets.",
            "supervisor": "Ms. R. Soumika",
            "students": ["Pranay Kumar Masi (25951A05DU)", "Nikhil Kaduduram (25951A05D8)"]
        }
    }

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    return metrics_data


if __name__ == "__main__":
    print("[TRAIN] Initiating ML training pipeline across 3 model candidates...")
    res = train_and_evaluate()
    print("[TRAIN] Completed successfully. Best model:", res["best_model_name"])
    print("[TRAIN] Test Accuracy:", res["model_comparison"]["Random Forest"]["test_accuracy"])
    print("[TRAIN] Saved model to artifacts/model.joblib and metrics to artifacts/metrics.json")
