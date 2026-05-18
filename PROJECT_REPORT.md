# HealthSentinel: Intelligent Community Health Monitoring & Early Warning System

## 📌 Project Overview
**HealthSentinel** is a proactive healthcare command center designed to monitor community health in real-time. Unlike traditional reactive systems, HealthSentinel uses local **Machine Learning (ML)** and **Computer Vision (OCR)** to digitize medical reports, calculate individual risk scores, and predict regional outbreaks before they happen.

The system is built with a **Privacy-First (Edge AI)** philosophy, ensuring all sensitive medical data is processed locally without any dependency on external cloud services or APIs.

---

## 🎯 Project Objectives
1.  **Digitization of Health Records**: Automating the extraction of clinical data from physical lab reports using OCR.
2.  **Individual Risk Assessment**: Calculating a standardized risk score based on medical vitals (Glucose, BP, Hb, etc.).
3.  **Regional Outbreak Prediction**: Aggregating individual data to predict the probability of community-level health crises.
4.  **Real-Time Monitoring**: Providing a live, interactive dashboard for health administrators with instant audio-visual alerts.
5.  **Data Privacy**: Implementing all AI/ML pipelines locally to comply with healthcare data protection standards.

---

## 🛠️ Technology Stack
*   **Frontend/Dashboard**: [Streamlit](https://streamlit.io/) (Python-based interactive web framework)
*   **Data Visualization**: [Plotly](https://plotly.com/) (Interactive Radar, Bar, and Line charts)
*   **Database**: [SQLite](https://sqlite.org/) (Local, serverless relational database)
*   **Computer Vision (OCR)**: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) & [EasyOCR](https://github.com/JaidedAI/EasyOCR)
*   **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Random Forest Classifier for outbreak prediction)
*   **Data Processing**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)

---

## 🏗️ System Architecture
The system follows a modular pipeline architecture:
1.  **Ingestion Layer**: Users upload images/PDFs of medical reports.
2.  **Processing Layer (OCR)**: Text is extracted using a multi-stage OCR pipeline (direct PDF extraction → Tesseract → EasyOCR).
3.  **Parsing Layer**: Raw text is converted into structured clinical parameters using RegEx-based heuristics.
4.  **Intelligence Layer (AI/ML)**: 
    *   **Risk Engine**: Calculates individual risk based on clinical thresholds.
    *   **Outbreak Engine**: Uses a pre-trained ML model to predict community risk based on regional trends.
5.  **Persistence Layer**: All data is stored in a local SQLite database.
6.  **Visualization Layer**: A real-time dashboard renders trends, maps, and radar charts.

---

## 💻 Major Code Explanations

### 1. The Multi-Stage OCR Pipeline (`modules/ocr_extract.py`)
To ensure high accuracy and speed, the system uses a 3-tier fallback mechanism:
*   **Tier 1 (PyMuPDF)**: If the file is a digital PDF, it extracts text directly (sub-millisecond speed).
*   **Tier 2 (Tesseract)**: For scanned images, Tesseract provides high-speed local OCR.
*   **Tier 3 (EasyOCR)**: If Tesseract fails, a deep-learning-based OCR (EasyOCR) is used for better handwriting/complex layout recognition.

```python
def _ocr_image(img_array: np.ndarray) -> str:
    # Try fast Tesseract first
    try:
        text = pytesseract.image_to_string(pil_img)
        if len(text.strip()) > 20: return text
    except: pass

    # Fallback to Deep Learning EasyOCR
    return "\n".join(reader.readtext(img_array, detail=0))
```

### 2. The Heuristic Clinical Parser (`modules/report_parser.py`)
Extracting data from unstructured text is challenging. We developed a robust **Regex-based Heuristic Engine** that identifies medical parameters regardless of formatting.
*   **Contextual Awareness:** It recognizes multiple variations like "BP", "Blood Pressure", or "Systolic/Diastolic".
*   **Unit Conversion:** Automatically detects if a temperature is in Celsius and converts it to Fahrenheit for standardized processing.
*   **Validation:** Includes "Clinical Sanity Checks" (e.g., ignoring a Glucose reading of 5000 as an OCR error).

```python
# Example: Intelligent Blood Pressure Extraction
def _extract_blood_pressure(text):
    for pattern in [
        r"(?:bp)[:\s=]+(\d{2,3})[^\d]+(\d{2,3})", # Matches "BP: 120/80"
        r"(\d{2,3})\s*[/\\]\s*(\d{2,3})",          # Matches "120 / 80"
    ]:
        m = re.search(pattern, text)
        if m:
            return float(m.group(1)), float(m.group(2))
```

### 2. The Predictive Outbreak Engine (`ml/model_utils.py` & `ml/train_model.py`)
The system doesn't just look at individuals; it looks at the "bigger picture." We trained a **Random Forest Classifier** on over 1,200 synthetic community health records.
*   **Features:** Average regional glucose, blood pressure, abnormal ratio (percentage of sick people in a zone), and symptom prevalence.
*   **Intelligence:** The model identifies patterns that human monitors might miss, such as a subtle but consistent rise in regional temperatures and cough frequency, which could signal an emerging flu outbreak.

```python
# Predicting outbreak probability for a specific region
def predict_outbreak_prob(record, comm_stats):
    features = [
        comm_stats['avg_glucose'], 
        comm_stats['avg_bp_systolic'],
        comm_stats['abnormal_ratio'],
        # ... other features
    ]
    # Model returns probability [0.0 - 1.0]
    return model.predict_proba([features])[0][1]
```

### 3. The Weighted Risk Engine (`modules/risk_scoring.py`)
Individual risk isn't just about one bad reading. Our engine uses a weighted formula:
*   **40%**: Count of abnormal parameters.
*   **30%**: Magnitude of deviation (how far a value is from the healthy range).
*   **30%**: Symptom load (e.g., fever + cough increases risk).

```python
def compute_risk_score(record: dict) -> float:
    score = 0.0
    for param, cfg in PARAM_THRESHOLDS.items():
        val = record.get(param)
        if val > cfg["high"]:
            # Deviation magnitude
            deviation = (val - cfg["high"]) / cfg["high"]
            score += cfg["weight"] * (50 + deviation * 50)
    # Add symptom penalty
    score += min(symptom_count * 5, 20)
    return round(min(score, 100), 2)
```

### 3. Asynchronous Audio-Visual Alerts (`app.py`)
Standard browser notifications are often blocked. We engineered a custom **JavaScript AudioContext** bridge that synthesizes different frequencies based on alert severity:
*   **Medium Risk**: 300Hz (Low chime)
*   **Critical Risk**: 800Hz + Dual-tone (Aggressive alarm)

```javascript
const ctx = new AudioContext();
const osc = ctx.createOscillator();
osc.frequency.value = freq; // Dynamic frequency based on risk
osc.start();
osc.stop(ctx.currentTime + 0.6);
```

### 4. Real-Time Network Simulation (`app.py`)
To demonstrate the system's capability without manual data entry, we built a simulation engine. It asynchronously injects synthetic patient data into the database every few seconds, triggering the entire ML and alerting pipeline.

---

---

## 🗄️ Database & Data Flow
The system uses a robust **SQLite** backend designed for high-concurrency simulation.
*   **Schema**: Stores patient identifiers, clinical vitals, calculated risk flags, and geospatial data.
*   **Data Flow**:
    1.  `ocr_extract.py` → Raw Text
    2.  `report_parser.py` → Structured JSON
    3.  `risk_scoring.py` → Risk Metrics
    4.  `database.py` → Persistent Storage
    5.  `app.py` → Live Visualization

---

---

## 📋 Step-by-Step Execution Guide
To run HealthSentinel in a college lab or local environment:

1.  **Environment Setup**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Model Initialization**:
    Ensure the ML model is trained (only needs to be done once):
    ```bash
    python ml/train_model.py
    ```
3.  **Launch Dashboard**:
    ```bash
    streamlit run app.py
    ```

---

## 📊 Data Dictionary (SQLite Schema)
| Column | Type | Description |
| :--- | :--- | :--- |
| `patient_id` | TEXT | Unique identifier for each patient (Auto-generated). |
| `region` | TEXT | Zone where the patient is located (North, South, etc.). |
| `glucose` | REAL | Blood sugar levels in mg/dL. |
| `bp_systolic` | REAL | Systolic blood pressure in mmHg. |
| `temperature` | REAL | Body temperature in °F. |
| `risk_score` | REAL | Calculated risk metric (0 - 100). |
| `risk_flag` | TEXT | Categorical risk level (Low, Medium, High). |
| `outbreak_prob`| REAL | Predicted probability of a regional outbreak. |

---

## 🚀 Key Innovation Highlights
*   **Interactive Radar Charts**: Allows doctors to see a patient's "Health Signature" compared to a healthy baseline at a glance.
*   **Geospatial Intelligence**: Maps outbreak probability across different urban zones (North, South, East, West, Central).
*   **Zero-Cloud Privacy**: Works entirely offline, making it ideal for rural health centers or high-security hospitals.

---

## 🔮 Future Enhancements
*   **Voice-to-Text Clinical Notes**: Allowing doctors to dictate patient observations.
*   **Blockchain Integration**: For immutable and secure medical record sharing between hospitals.
*   **Federated Learning**: Improving the ML model across different hospitals without actually sharing private patient data.

---
**Presented by:** [Your Name]
**Project Name:** HealthSentinel
**Version:** 1.0.0
