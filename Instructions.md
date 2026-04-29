# Project: Smart Community Health Monitoring and Early Warning System
## Primary Input via Medical Report Images/PDF

## Objective
Build a system where community health data is collected from uploaded medical reports (images or PDFs), extracted using OCR, analyzed for trends, and used for outbreak prediction and early warning alerts.

---

# Core Workflow

Upload medical report image/PDF

-> OCR text extraction

-> Extract health parameters

-> Store structured health records

-> Community aggregation

-> Risk scoring

-> ML outbreak prediction

-> Alert generation

-> Dashboard

---

# Input Types
Accept:
- JPG
- PNG
- PDF

---

# Module 1 OCR Data Collection

Build OCR pipeline using:

EasyOCR

Fallback:
pytesseract

Extract text from uploaded reports.

Function:
extract_text(file)

---

# Module 2 Health Parameter Extraction

Parse report text and extract:

glucose

blood_pressure

hemoglobin

temperature

cholesterol

symptoms (if present)

Use:
keyword mapping + regex.

Example:

Glucose: 172 mg/dL

Extract:
172

Return structured record:

{
region: "Area1",
glucose:172,
blood_pressure:140,
hemoglobin:12,
date:"2026-04-29"
}

---

# Module 3 Store Community Data

Save extracted records in SQLite.

Each uploaded report becomes one health record.

Schema:

patient_id

region

glucose

bp

hemoglobin

risk_flag

date

---

# Module 4 Community Aggregation

Compute:
average glucose by region

abnormal case counts

disease pattern frequency

risk score

---

# Module 5 Risk Score Formula

Risk Score =
weighted combination of:

abnormal case ratio

symptom prevalence

trend growth

Classify:
Low

Medium

High

---

# Module 6 ML Outbreak Prediction

Train Random Forest model.

Inputs:
aggregated community indicators.

Output:
outbreak probability.

Save:
model.pkl

---

# Module 7 Alerts

Generate:

Individual abnormal alert

Community risk alert

Early warning alert

---

# Module 8 Dashboard

Show:

Uploaded reports count

Extracted health records

Community risk score

Trend graphs

Outbreak probability

Alerts

---

# Files

app.py

ocr_extract.py

report_parser.py

database.py

risk_scoring.py

train_model.py

alerts.py

model.pkl

---

# Build Order

Step 1
OCR extraction

Step 2
Report parser

Step 3
Store extracted records

Step 4
Community aggregation

Step 5
Risk scoring

Step 6
ML outbreak prediction

Step 7
Dashboard

---

# Requirements
streamlit

easyocr

pdf2image

opencv-python

pandas

numpy

scikit-learn

sqlite3

plotly

joblib

---

# Deliverable
Working prototype that:

accepts medical reports

extracts health data automatically

monitors community health

predicts outbreak risk

generates early warning alerts