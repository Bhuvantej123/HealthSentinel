# Smart Community Health Monitoring & Early Warning System

A full-stack Python/Streamlit application that ingests medical report images/PDFs via OCR, extracts key health parameters, stores them in SQLite, aggregates community-level risk scores, predicts outbreak probabilities using a Random Forest model, and surfaces real-time alerts on a premium interactive dashboard.

---

## User Review Required

> [!IMPORTANT]
> **Tech choices to confirm before I build:**
> - EasyOCR downloads ~1 GB of model weights on first run. Acceptable for competition demo environment?
> - `pdf2image` requires **Poppler** installed on Windows. I'll include setup instructions. If Poppler can't be installed, I'll fall back to `PyMuPDF` (fitz) — which has no external dependency. **Recommend: PyMuPDF fallback.**
> - The ML model (Module 6) will be pre-trained on **synthetic representative data** I generate, then saved as `model.pkl`. On upload, the model uses the live extracted data for prediction.
> - Do you have a Gemini/OpenAI API key? If yes, I can optionally add AI-powered report narrative summarization as a premium feature.

> [!WARNING]
> **EasyOCR Installation**: First run will attempt to download ~1 GB of pytorch model weights. Make sure you have internet and disk space. The fallback to `pytesseract` requires Tesseract-OCR to be installed separately.

---

## Open Questions

> [!NOTE]
> 1. Should I add **patient name / patient ID** parsing from OCR text, or is the schema purely anonymous by region?
> 2. Should reports from the **same patient** update an existing record, or always create new records?
> 3. Any specific **regions/areas** to pre-populate in the dropdown (e.g., Area 1–5, North/South/East/West)?

I'll proceed with sensible defaults (5 predefined regions, always create new records, anonymous schema) unless you instruct otherwise.

---

## Proposed Changes

### Module 1 — OCR Extraction

#### [NEW] `ocr_extract.py`
- `extract_text(file_path, file_type)` — primary EasyOCR pipeline
- Falls back to `pytesseract` if EasyOCR fails
- PDF pages are converted to images using `PyMuPDF` (fitz), each page OCR'd and text concatenated
- Returns raw OCR string

---

### Module 2 — Health Parameter Extraction

#### [NEW] `report_parser.py`
- `parse_health_parameters(raw_text, region)` — uses regex + keyword mapping
- Extracts: `glucose`, `blood_pressure` (systolic/diastolic), `hemoglobin`, `temperature`, `cholesterol`, `symptoms`
- Returns structured dict with patient record
- Handles common OCR artifacts (e.g., "l72" → 172, "B|" → "BI")

---

### Module 3 — Database

#### [NEW] `database.py`
- SQLite-backed with `sqlite3`
- Table: `health_records` with columns: `patient_id`, `region`, `glucose`, `bp_systolic`, `bp_diastolic`, `hemoglobin`, `temperature`, `cholesterol`, `symptoms`, `risk_flag`, `risk_score`, `outbreak_prob`, `date`, `raw_text`
- Functions: `init_db()`, `insert_record(record)`, `get_all_records()`, `get_records_by_region(region)`, `get_aggregated_stats()`

---

### Module 4 — Community Aggregation

#### [NEW] `risk_scoring.py` (also handles aggregation)
- `aggregate_community_stats(df)` — computes per-region averages, abnormal case counts, trend data
- Thresholds for "abnormal": glucose > 126, BP systolic > 140, hemoglobin < 11, temperature > 99.5°F, cholesterol > 200

---

### Module 5 — Risk Scoring

Inside `risk_scoring.py`:
- `compute_risk_score(record)` — weighted formula:
  - 40% abnormal indicator count
  - 30% symptom prevalence score  
  - 30% trend growth (vs. 7-day rolling average)
- `classify_risk(score)` → `"Low"` / `"Medium"` / `"High"`

---

### Module 6 — ML Outbreak Prediction

#### [NEW] `train_model.py`
- Generates 500-record synthetic training dataset with realistic distributions
- Trains `RandomForestClassifier` (scikit-learn) with features: avg_glucose, avg_bp, abnormal_ratio, case_count, trend_slope
- Target: outbreak (0/1) — labeled based on domain thresholds
- Saves `model.pkl` + `scaler.pkl` via joblib
- Run once to generate model files; app loads them at startup

---

### Module 7 — Alerts

#### [NEW] `alerts.py`
- `generate_alerts(record, community_stats, outbreak_prob)` → list of alert dicts
- Three alert types:
  - 🔴 **Individual Abnormal Alert** — triggered when ≥2 parameters are out of range
  - 🟠 **Community Risk Alert** — triggered when region risk score ≥ Medium
  - 🚨 **Early Warning Alert** — triggered when outbreak probability ≥ 0.6

---

### Module 8 — Dashboard (Main App)

#### [NEW] `app.py`
- Streamlit multi-page layout with tabs: **Upload**, **Records**, **Community Analytics**, **Outbreak Prediction**, **Alerts**
- Premium dark-mode design with custom CSS injected via `st.markdown`
- Upload tab: drag-drop file uploader, live OCR preview, parameter extraction display, instant DB save
- Records tab: filterable data table with region/date filters
- Community Analytics tab: Plotly charts — bar chart (avg glucose by region), heat-map (risk scores), time-series trend lines
- Outbreak Prediction tab: gauge chart showing outbreak probability per region
- Alerts tab: color-coded alert cards (red/orange/yellow)

---

### Supporting Files

#### [NEW] `requirements.txt`
```
streamlit>=1.32
easyocr
pymupdf
opencv-python-headless
pandas
numpy
scikit-learn
plotly
joblib
Pillow
pytesseract
```

#### [NEW] `sample_report.txt`
A synthetic medical report text file for demonstration.

---

## File Structure After Build

```
Community Health Monitoring/
├── app.py                  ← Streamlit dashboard (main entry)
├── ocr_extract.py          ← OCR pipeline
├── report_parser.py        ← Health parameter extraction
├── database.py             ← SQLite operations
├── risk_scoring.py         ← Aggregation + risk scoring
├── train_model.py          ← Model training script
├── alerts.py               ← Alert generation
├── model.pkl               ← Trained RF model (generated)
├── scaler.pkl              ← Feature scaler (generated)
├── health_records.db       ← SQLite DB (auto-created)
├── requirements.txt
└── Instructions.md
```

---

## Verification Plan

### Automated Tests
1. Run `python train_model.py` — must produce `model.pkl` and `scaler.pkl`
2. Run `python database.py` — must create `health_records.db` with correct schema
3. Run `python report_parser.py` — test with sample text, must extract parameters
4. Run `streamlit run app.py` — dashboard must load without errors

### Manual Verification (Browser)
- Upload a sample medical report image (I'll create one)
- Verify OCR extracts text
- Verify parameters are extracted and saved
- Verify dashboard shows charts and outbreak prediction

### Demo-Ready Features
- **Pre-loaded synthetic data** (30 records across 5 regions) so dashboard is never empty
- Sample report PNG generated for judges to test upload
- Alerting system immediately triggers when high-risk parameters detected
