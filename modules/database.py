"""
Module 3: Database — SQLite CRUD + Demo Data Seeder
All DB operations for health records.
"""

import sqlite3
import random
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "health_records.db"

REGIONS = ["North Zone", "South Zone", "East Zone", "West Zone", "Central Zone"]

SYMPTOM_POOL = [
    "fever", "cough", "fatigue", "headache", "nausea",
    "diarrhea", "body ache", "weakness", "dizziness", "chills",
]


# ── Connection ────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist, then seed demo data once."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS health_records (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id   TEXT    NOT NULL,
            region       TEXT    NOT NULL,
            glucose      REAL,
            bp_systolic  REAL,
            bp_diastolic REAL,
            hemoglobin   REAL,
            temperature  REAL,
            cholesterol  REAL,
            symptoms     TEXT    DEFAULT '',
            risk_score   REAL    DEFAULT 0,
            risk_flag    TEXT    DEFAULT 'Low',
            outbreak_prob REAL   DEFAULT 0,
            date         TEXT    NOT NULL,
            raw_text     TEXT    DEFAULT '',
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    _seed_demo_data()


# ── CRUD ──────────────────────────────────────────────────────────────────────

def insert_record(record: dict) -> int:
    """Insert a health record and return its new row id."""
    conn = _get_conn()
    cur = conn.execute("""
        INSERT INTO health_records
            (patient_id, region, glucose, bp_systolic, bp_diastolic,
             hemoglobin, temperature, cholesterol, symptoms,
             risk_score, risk_flag, outbreak_prob, date, raw_text)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record.get("patient_id"),    record.get("region"),
        record.get("glucose"),       record.get("bp_systolic"),
        record.get("bp_diastolic"),  record.get("hemoglobin"),
        record.get("temperature"),   record.get("cholesterol"),
        record.get("symptoms", ""),  record.get("risk_score", 0),
        record.get("risk_flag", "Low"), record.get("outbreak_prob", 0),
        record.get("date"),          record.get("raw_text", ""),
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_all_records():
    """Return all health records as a pandas DataFrame."""
    import pandas as pd
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM health_records ORDER BY created_at DESC", conn
    )
    conn.close()
    return df


def get_records_by_region(region: str):
    """Return records for a specific region as a pandas DataFrame."""
    import pandas as pd
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM health_records WHERE region = ? ORDER BY date",
        conn, params=(region,)
    )
    conn.close()
    return df


def get_record_count() -> int:
    conn = _get_conn()
    count = conn.execute("SELECT COUNT(*) FROM health_records").fetchone()[0]
    conn.close()
    return count


# ── Demo Seeder ───────────────────────────────────────────────────────────────

def _seed_demo_data():
    """
    Insert 40 synthetic records across 5 regions (only if table is empty).
    Risk weighting per region makes the data realistic and interesting.
    """
    conn = _get_conn()
    count = conn.execute("SELECT COUNT(*) FROM health_records").fetchone()[0]
    conn.close()
    if count > 0:
        return

    # Lazy import to avoid circular dependency at module load
    from modules.risk_scoring import compute_risk_score, classify_risk

    random.seed(42)

    # Higher multiplier → worse health outcomes for that region
    region_risk = {
        "North Zone":   0.25,
        "South Zone":   0.65,
        "East Zone":    0.20,
        "West Zone":    0.72,
        "Central Zone": 0.45,
    }

    records = []
    for i in range(40):
        region = random.choice(REGIONS)
        rm     = region_risk[region]
        days_ago = random.randint(0, 30)
        rec_date = (date.today() - timedelta(days=days_ago)).isoformat()

        glucose     = round(random.gauss(100 + rm * 80,  20), 1)
        bp_sys      = round(random.gauss(118 + rm * 40,  14), 1)
        bp_dia      = round(random.gauss(78  + rm * 20,  9),  1)
        hemoglobin  = round(random.gauss(14  - rm * 4,   1.5), 1)
        temperature = round(random.gauss(98.6 + rm * 2,  0.8), 1)
        cholesterol = round(random.gauss(178 + rm * 60,  25), 1)

        n_syms   = random.choices([0, 1, 2, 3], weights=[0.5, 0.25, 0.15, 0.10])[0]
        symptoms = ", ".join(random.sample(SYMPTOM_POOL, min(n_syms, len(SYMPTOM_POOL))))

        rec = {
            "glucose": glucose, "bp_systolic": bp_sys, "bp_diastolic": bp_dia,
            "hemoglobin": hemoglobin, "temperature": temperature,
            "cholesterol": cholesterol, "symptoms": symptoms,
        }
        risk_score = compute_risk_score(rec)
        risk_flag  = classify_risk(risk_score)

        records.append((
            f"DEMO-{i+1:03d}", region, glucose, bp_sys, bp_dia,
            hemoglobin, temperature, cholesterol, symptoms,
            risk_score, risk_flag, 0.0, rec_date, "Synthetic demo record",
        ))

    conn = _get_conn()
    conn.executemany("""
        INSERT INTO health_records
            (patient_id, region, glucose, bp_systolic, bp_diastolic,
             hemoglobin, temperature, cholesterol, symptoms,
             risk_score, risk_flag, outbreak_prob, date, raw_text)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, records)
    conn.commit()
    conn.close()
