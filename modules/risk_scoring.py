"""
Module 5: Risk Scoring & Community Aggregation
Computes individual risk scores and aggregates community-level statistics.
"""

import pandas as pd
import numpy as np
from typing import Optional

# ── Thresholds ─────────────────────────────────────────────────────────────────

PARAM_THRESHOLDS = {
    "glucose":      {"high": 126, "low": 70,  "weight": 0.25},
    "bp_systolic":  {"high": 140, "low": 90,  "weight": 0.25},
    "hemoglobin":   {"high": 18,  "low": 11,  "weight": 0.20},
    "temperature":  {"high": 99.5,"low": 96.0,"weight": 0.15},
    "cholesterol":  {"high": 200, "low": 0,   "weight": 0.15},
}


# ── Individual Risk Score ─────────────────────────────────────────────────────

def compute_risk_score(record: dict) -> float:
    """
    Weighted risk score 0–100.
    Formula: 40% abnormal params + 30% symptom load + 30% deviation magnitude.
    """
    score = 0.0
    counted = 0

    for param, cfg in PARAM_THRESHOLDS.items():
        val = record.get(param)
        if val is None:
            continue
        counted += 1
        high, low, w = cfg["high"], cfg["low"], cfg["weight"]

        if val > high:
            deviation = min((val - high) / max(high, 1), 1.0)
            score += w * (50 + deviation * 50)
        elif low > 0 and val < low:
            deviation = min((low - val) / max(low, 1), 1.0)
            score += w * (50 + deviation * 50)
        # else within normal range → contributes 0

    # Symptom penalty (up to 20 extra points)
    symptoms = record.get("symptoms", "") or ""
    symptom_count = len([s for s in symptoms.split(",") if s.strip()])
    score += min(symptom_count * 5, 20)

    return round(min(score, 100), 2)


def classify_risk(score: float) -> str:
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    return "High"


def is_abnormal(record: dict) -> dict:
    """Return dict of which params are abnormal (True/False)."""
    flags = {}
    for param, cfg in PARAM_THRESHOLDS.items():
        val = record.get(param)
        if val is None:
            flags[param] = False
            continue
        flags[param] = val > cfg["high"] or (cfg["low"] > 0 and val < cfg["low"])
    return flags


# ── Community Aggregation ──────────────────────────────────────────────────────

def aggregate_community_stats(df: pd.DataFrame) -> dict:
    """
    Compute per-region aggregated statistics for dashboard + ML features.
    Returns dict keyed by region name.
    """
    if df.empty:
        return {}

    stats = {}
    numeric_cols = ["glucose", "bp_systolic", "bp_diastolic",
                    "hemoglobin", "temperature", "cholesterol"]

    for region in df["region"].unique():
        rdf = df[df["region"] == region].copy()

        # Mean values
        avgs = {f"avg_{col}": rdf[col].dropna().mean() for col in numeric_cols}

        # Abnormal ratios
        total = len(rdf)
        abnormal_count = 0
        for _, row in rdf.iterrows():
            flags = is_abnormal(row.to_dict())
            if any(flags.values()):
                abnormal_count += 1

        abnormal_ratio = abnormal_count / max(total, 1)

        # Risk distribution
        risk_counts = rdf["risk_flag"].value_counts().to_dict() if "risk_flag" in rdf.columns else {}

        # Symptom prevalence
        symptom_counts: dict = {}
        for syms in rdf["symptoms"].dropna():
            for s in syms.split(","):
                s = s.strip()
                if s:
                    symptom_counts[s] = symptom_counts.get(s, 0) + 1

        # 7-day trend: avg glucose trend slope
        trend_slope = _compute_trend_slope(rdf, "glucose")

        stats[region] = {
            **avgs,
            "case_count":      total,
            "abnormal_count":  abnormal_count,
            "abnormal_ratio":  round(abnormal_ratio, 3),
            "risk_counts":     risk_counts,
            "symptom_counts":  symptom_counts,
            "trend_slope":     trend_slope,
            "avg_symptom_count": (
                rdf["symptoms"].dropna()
                .apply(lambda x: len([s for s in x.split(",") if s.strip()]))
                .mean()
            ),
        }

    return stats


def _compute_trend_slope(df: pd.DataFrame, col: str) -> float:
    """Compute linear trend slope of a parameter over time."""
    try:
        df = df.dropna(subset=["date", col]).copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        if len(df) < 2:
            return 0.0
        x = (df["date"] - df["date"].min()).dt.days.values.astype(float)
        y = df[col].values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0])
        return round(slope, 4)
    except Exception:
        return 0.0
