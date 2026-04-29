"""
ML Model Utilities
Loads the trained model + scaler and exposes a single predict function.
Auto-trains the model on first use if pkl files are missing.
"""

import numpy as np
import joblib
from pathlib import Path

ML_DIR = Path(__file__).parent

FEATURE_COLS = [
    "avg_glucose", "avg_bp_systolic", "avg_hemoglobin",
    "avg_temperature", "avg_cholesterol",
    "abnormal_ratio", "case_count", "avg_symptom_count",
]

_model  = None
_scaler = None


def _ensure_model():
    """Load (or train) model + scaler, caching in module-level globals."""
    global _model, _scaler

    if _model is not None:
        return True

    model_path  = ML_DIR / "model.pkl"
    scaler_path = ML_DIR / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        # Auto-train on first run
        try:
            from ml.train_model import train_and_save
            train_and_save()
        except Exception as e:
            print(f"[model_utils] Auto-training failed: {e}")
            return False

    try:
        _model  = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        return True
    except Exception as e:
        print(f"[model_utils] Load error: {e}")
        return False


def predict_outbreak_prob(record: dict, community_stats: dict = None) -> float:
    """
    Predict outbreak probability for a given health record.
    Falls back gracefully to 0.0 if model unavailable.

    Args:
        record:          Individual patient health record dict.
        community_stats: Optional per-region aggregated stats dict from
                         risk_scoring.aggregate_community_stats(). When provided,
                         community averages are used as features (more accurate).

    Returns:
        float in [0, 1] — outbreak probability.
    """
    if not _ensure_model():
        return 0.0

    region = record.get("region")

    if community_stats and region and region in community_stats:
        s = community_stats[region]
        
        def safe_get(k, default):
            val = s.get(k)
            return default if val is None or np.isnan(val) else val

        avg_glucose      = safe_get("avg_glucose", record.get("glucose") or 100)
        avg_bp           = safe_get("avg_bp_systolic", record.get("bp_systolic") or 120)
        avg_hgb          = safe_get("avg_hemoglobin", record.get("hemoglobin") or 13)
        avg_temp         = safe_get("avg_temperature", record.get("temperature") or 98.6)
        avg_chol         = safe_get("avg_cholesterol", record.get("cholesterol") or 180)
        abnormal_ratio   = safe_get("abnormal_ratio", 0.0)
        case_count       = safe_get("case_count", 1)
        avg_symptom_count = safe_get("avg_symptom_count", 0.0)
    else:
        # Individual record → derive community features from single record
        avg_glucose   = record.get("glucose") or 100
        avg_bp        = record.get("bp_systolic") or 120
        avg_hgb       = record.get("hemoglobin") or 13
        avg_temp      = record.get("temperature") or 98.6
        avg_chol      = record.get("cholesterol") or 180
        case_count    = 1
        avg_symptom_count = len([
            s for s in (record.get("symptoms") or "").split(",") if s.strip()
        ])

        # Compute abnormal ratio for this single record
        from modules.risk_scoring import is_abnormal
        flags = is_abnormal(record)
        abnormal_ratio = sum(flags.values()) / max(len(flags), 1)

    features = np.array([[
        avg_glucose, avg_bp, avg_hgb, avg_temp, avg_chol,
        abnormal_ratio, case_count, avg_symptom_count,
    ]])

    try:
        features_scaled = _scaler.transform(features)
        prob = _model.predict_proba(features_scaled)[0][1]
        return round(float(prob), 3)
    except Exception as e:
        print(f"[model_utils] Prediction error: {e}")
        return 0.0


def predict_all_regions(community_stats: dict) -> dict:
    """Return outbreak probability for every region in community_stats."""
    return {
        region: predict_outbreak_prob({"region": region}, community_stats)
        for region, stats in community_stats.items()
    }
