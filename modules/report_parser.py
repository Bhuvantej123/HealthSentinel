"""
Module 2: Health Parameter Extraction
Parses OCR text and extracts structured health indicators using
keyword mapping and regex patterns.
"""

import re
import uuid
from datetime import date
from typing import Optional, Tuple


SYMPTOM_KEYWORDS = [
    "fever", "cough", "fatigue", "headache", "nausea",
    "vomiting", "diarrhea", "shortness of breath", "chest pain",
    "body ache", "loss of appetite", "weakness", "dizziness",
    "rash", "joint pain", "chills", "sweating",
]


def parse_health_parameters(raw_text: str, region: str) -> dict:
    """
    Parse raw OCR text and return a structured health record dict.
    """
    text = raw_text.lower()

    glucose      = _extract_glucose(text)
    bp_sys, bp_dia = _extract_blood_pressure(text)
    hemoglobin   = _extract_hemoglobin(text)
    temperature  = _extract_temperature(text)
    cholesterol  = _extract_cholesterol(text)
    symptoms     = _extract_symptoms(text)

    return {
        "patient_id":    f"P-{str(uuid.uuid4())[:8].upper()}",
        "region":        region,
        "glucose":       glucose,
        "bp_systolic":   bp_sys,
        "bp_diastolic":  bp_dia,
        "hemoglobin":    hemoglobin,
        "temperature":   temperature,
        "cholesterol":   cholesterol,
        "symptoms":      symptoms,
        "date":          date.today().isoformat(),
        "raw_text":      raw_text[:3000],
    }


# ── Extractors ────────────────────────────────────────────────────────────────

def _extract_glucose(text: str) -> Optional[float]:
    patterns = [
        r"(?:blood\s+)?g[a-z]+cose[:\s=]+(\d+\.?\d*)",
        r"(?:fasting\s+)?(?:blood\s+)?sugar[:\s=]+(\d+\.?\d*)",
        r"(?:fbs|rbs|bs)[:\s=]+(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*mg[/\\]?dl",
    ]
    return _first_match(text, patterns, float, 30, 700)


def _extract_blood_pressure(text: str) -> Tuple[Optional[float], Optional[float]]:
    # "BP: 130/85" or "blood pressure 130/85"
    for pattern in [
        r"(?:blood\s+pressure|bp)[:\s=]+(\d{2,3})[^\d]+(\d{2,3})",
        r"(\d{2,3})[^\d]+(\d{2,3})\s*mm[/\\]?hg",
        r"(\d{2,3})\s*[/\\]\s*(\d{2,3})",
    ]:
        m = re.search(pattern, text)
        if m:
            s, d = float(m.group(1)), float(m.group(2))
            if 60 <= s <= 250 and 40 <= d <= 150:
                return s, d
    return None, None


def _extract_hemoglobin(text: str) -> Optional[float]:
    patterns = [
        r"h[a-z]*mo?globin[:\s=]+(\d+\.?\d*)",
        r"\bhb\b[:\s=]+(\d+\.?\d*)",
        r"\bhgb\b[:\s=]+(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*g[/\\]?dl",
    ]
    return _first_match(text, patterns, float, 3, 25)


def _extract_temperature(text: str) -> Optional[float]:
    patterns = [
        r"temp(?:erature)?[:\s=]+(\d+\.?\d*)",
        r"(\d{2,3}\.?\d*)\s*°?\s*[fc]\b",
        r"(\d{2,3}\.?\d*)\s*degrees",
    ]
    val = _first_match(text, patterns, float, 90, 110)
    if val is None:
        return None
    # Convert Celsius → Fahrenheit if value looks like Celsius
    if val < 50:
        val = round(val * 9 / 5 + 32, 1)
    return val


def _extract_cholesterol(text: str) -> Optional[float]:
    patterns = [
        r"(?:total\s+)?ch[a-z]*ol[:\s=]+(\d+\.?\d*)",
        r"chol[:\s=]+(\d+\.?\d*)",
        r"(?:ldl|hdl)[:\s=]+(\d+\.?\d*)",
    ]
    return _first_match(text, patterns, float, 50, 600)


def _extract_symptoms(text: str) -> str:
    found = [s for s in SYMPTOM_KEYWORDS if s in text]
    return ", ".join(found)


# ── Helper ────────────────────────────────────────────────────────────────────

def _first_match(text, patterns, cast, min_val, max_val):
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                val = cast(m.group(1))
                if min_val <= val <= max_val:
                    return val
            except (ValueError, IndexError):
                continue
    return None
