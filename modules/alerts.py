"""
Module 7: Alert Generation
Produces three tiers of health alerts based on patient and community data.
"""

from datetime import datetime
from typing import Optional


RISK_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

# ── Thresholds ─────────────────────────────────────────────────────────────────
ABNORMAL_THRESHOLDS = {
    "glucose":     ("Glucose",     lambda v: v > 126 or v < 70),
    "bp_systolic": ("BP Systolic", lambda v: v > 140 or v < 90),
    "hemoglobin":  ("Hemoglobin",  lambda v: v < 11),
    "temperature": ("Temperature", lambda v: v > 99.5 or v < 96.0),
    "cholesterol": ("Cholesterol", lambda v: v > 200),
}

OUTBREAK_THRESHOLD       = 0.60
COMMUNITY_RISK_THRESHOLD = "Medium"   # or "High"


def generate_alerts(
    record: dict,
    community_stats: Optional[dict] = None,
    outbreak_prob: float = 0.0,
) -> list[dict]:
    """
    Returns a list of alert dicts:
      { type, severity, title, message, color, timestamp }
    """
    alerts = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 1. Individual Abnormal Alert ──────────────────────────────────────────
    abnormal_params = []
    for key, (label, check_fn) in ABNORMAL_THRESHOLDS.items():
        val = record.get(key)
        if val is not None and check_fn(val):
            abnormal_params.append(f"{label} ({val})")

    if len(abnormal_params) >= 1:
        severity = "Critical" if len(abnormal_params) >= 3 else "Warning"
        color    = "#ef4444"  if severity == "Critical" else "#f59e0b"
        alerts.append({
            "type":      "individual",
            "severity":  severity,
            "title":     f"Abnormal Parameters Detected — Patient {record.get('patient_id', 'N/A')}",
            "message":   f"Out-of-range values: {', '.join(abnormal_params)}.",
            "color":     color,
            "timestamp": ts,
        })

    # ── 2. Community Risk Alert ───────────────────────────────────────────────
    region = record.get("region")
    if community_stats and region and region in community_stats:
        stats = community_stats[region]
        risk_counts = stats.get("risk_counts", {})
        high_cases  = risk_counts.get("High", 0)
        total       = stats.get("case_count", 1)
        high_ratio  = high_cases / max(total, 1)

        if high_ratio >= 0.25 or stats.get("abnormal_ratio", 0) >= 0.4:
            alerts.append({
                "type":      "community",
                "severity":  "High",
                "title":     f"Community Risk Alert — {region}",
                "message":   (
                    f"{high_cases}/{total} high-risk cases "
                    f"({high_ratio:.0%} of population). "
                    f"Abnormal parameter ratio: {stats.get('abnormal_ratio', 0):.0%}."
                ),
                "color":     "#f97316",
                "timestamp": ts,
            })

    # ── 3. Early Warning / Outbreak Alert ────────────────────────────────────
    if outbreak_prob >= OUTBREAK_THRESHOLD:
        alerts.append({
            "type":      "outbreak",
            "severity":  "Critical",
            "title":     f"Early Outbreak Warning — {region}",
            "message":   (
                f"ML model predicts {outbreak_prob:.0%} outbreak probability "
                f"in {region}. Immediate health intervention recommended."
            ),
            "color":     "#dc2626",
            "timestamp": ts,
        })

    return alerts


def get_community_level_alerts(community_stats: dict, outbreak_probs: dict) -> list[dict]:
    """
    Generate community-wide alerts without a specific patient context.
    Used on the Alerts dashboard tab.
    """
    alerts = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    for region, stats in community_stats.items():
        prob = outbreak_probs.get(region, 0.0)

        # Outbreak warning
        if prob >= OUTBREAK_THRESHOLD:
            alerts.append({
                "type":      "outbreak",
                "severity":  "Critical",
                "title":     f"Outbreak Warning — {region}",
                "message":   f"Predicted outbreak probability: {prob:.0%}. Immediate action required.",
                "color":     "#dc2626",
                "timestamp": ts,
            })

        # High abnormal ratio
        if stats.get("abnormal_ratio", 0) >= 0.45:
            alerts.append({
                "type":      "community",
                "severity":  "High",
                "title":     f"Elevated Risk — {region}",
                "message":   (
                    f"{stats['abnormal_ratio']:.0%} of cases show abnormal parameters. "
                    f"Case count: {stats['case_count']}."
                ),
                "color":     "#f97316",
                "timestamp": ts,
            })

        # Symptom cluster
        syms = stats.get("symptom_counts", {})
        dominant = max(syms, key=syms.get) if syms else None
        if dominant and syms[dominant] >= max(3, stats.get("case_count", 10) * 0.3):
            alerts.append({
                "type":      "symptom_cluster",
                "severity":  "Medium",
                "title":     f"Symptom Cluster Detected — {region}",
                "message":   f"'{dominant}' reported in {syms[dominant]} cases. Monitor for spread.",
                "color":     "#eab308",
                "timestamp": ts,
            })

    return alerts
