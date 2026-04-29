"""
Module 6: ML Model Training
Trains a Random Forest classifier on synthetic community health data.
Run once:  python ml/train_model.py
Outputs:   ml/model.pkl  and  ml/scaler.pkl
"""

import sys
from pathlib import Path

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

ML_DIR      = Path(__file__).parent
RANDOM_SEED = 42
N_SAMPLES   = 1200

FEATURE_COLS = [
    "avg_glucose", "avg_bp_systolic", "avg_hemoglobin",
    "avg_temperature", "avg_cholesterol",
    "abnormal_ratio", "case_count", "avg_symptom_count",
]


# ── Synthetic Data Generation ─────────────────────────────────────────────────

def _generate_data(n: int = N_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    for _ in range(n):
        # Community risk level drives the distributions
        risk_level = rng.choice([0, 1, 2], p=[0.55, 0.28, 0.17])  # low/med/high

        base_glucose    = [95,  130, 165][risk_level]
        base_bp         = [115, 135, 158][risk_level]
        base_hgb        = [14,  12,  10][risk_level]
        base_temp       = [98.4, 99.0, 100.2][risk_level]
        base_chol       = [175, 210, 250][risk_level]
        base_abn_ratio  = [0.10, 0.35, 0.60][risk_level]
        base_cases      = rng.integers(5, 80)
        base_syms       = [0.5, 1.5, 3.0][risk_level]

        avg_glucose      = float(rng.normal(base_glucose,   18))
        avg_bp           = float(rng.normal(base_bp,        14))
        avg_hgb          = float(rng.normal(base_hgb,       1.2))
        avg_temp         = float(rng.normal(base_temp,       0.6))
        avg_chol         = float(rng.normal(base_chol,       22))
        abnormal_ratio   = float(np.clip(rng.normal(base_abn_ratio, 0.1), 0, 1))
        case_count       = int(base_cases)
        avg_syms         = float(max(0, rng.normal(base_syms, 0.7)))

        # Outbreak label: high probability only in severe community conditions
        outbreak = int(
            (avg_glucose > 145 and abnormal_ratio > 0.40) or
            (avg_bp      > 150 and abnormal_ratio > 0.45) or
            (abnormal_ratio > 0.60) or
            (avg_syms    > 3.5 and abnormal_ratio > 0.35)
        )

        rows.append([
            avg_glucose, avg_bp, avg_hgb, avg_temp, avg_chol,
            abnormal_ratio, case_count, avg_syms, outbreak,
        ])

    cols = FEATURE_COLS + ["outbreak"]
    return pd.DataFrame(rows, columns=cols)


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_save() -> None:
    print("-" * 50)
    print("  Smart Health Monitoring -- Model Training")
    print("-" * 50)

    print(f"\n[1/4] Generating {N_SAMPLES} synthetic community records...")
    df = _generate_data(N_SAMPLES)
    print(f"      Outbreak cases: {df['outbreak'].sum()} / {N_SAMPLES} "
          f"({df['outbreak'].mean():.1%})")

    X = df[FEATURE_COLS].values
    y = df["outbreak"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    print("\n[2/4] Scaling features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\n[3/4] Training Random Forest (100 trees)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print("\n[4/4] Evaluation:")
    print(classification_report(y_test, y_pred, target_names=["No Outbreak", "Outbreak"]))
    print(f"      ROC-AUC: {auc:.4f}")

    # Save
    ML_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  ML_DIR / "model.pkl")
    joblib.dump(scaler, ML_DIR / "scaler.pkl")
    print(f"\n[OK] model.pkl  -> {ML_DIR / 'model.pkl'}")
    print(f"[OK] scaler.pkl -> {ML_DIR / 'scaler.pkl'}")
    print("-" * 50)


if __name__ == "__main__":
    train_and_save()
