"""
app.py — Smart Community Health Monitoring & Early Warning System
Main Streamlit dashboard entry point.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile, os

from modules.database     import init_db, insert_record, get_all_records, get_record_count
from modules.ocr_extract  import extract_text
from modules.report_parser import parse_health_parameters
from modules.risk_scoring  import compute_risk_score, classify_risk, aggregate_community_stats
from modules.alerts        import generate_alerts, get_community_level_alerts
from ml.model_utils        import predict_outbreak_prob, predict_all_regions

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthSentinel",
    page_icon="desktop_computer",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "play_alert" in st.session_state and st.session_state.play_alert:
    alert_tuple = st.session_state.play_alert
    if isinstance(alert_tuple, tuple):
        alert_type, alert_msg = alert_tuple
    else:
        alert_type, alert_msg = "High", f"High risk anomaly detected."
        
    # Visual Pop-up Notification
    icon_map = {"Critical": "🚨", "High": "⚠️", "Medium": "🔔", "Outbreak": "☣️"}
    st.toast(f"**{alert_type} ALERT**\\n\\n{alert_msg}", icon=icon_map.get(alert_type, "⚠️"))
    
    # Audio Alert via Javascript AudioContext
    if alert_type == "Medium": freq = 300
    elif alert_type == "High": freq = 500
    elif alert_type == "Critical": freq = 800
    else: freq = 1000
    
    import streamlit.components.v1 as components
    import time
    # Unique ID forces Streamlit to re-render the JS and play sound every time
    unique_ts = time.time()
    
    components.html(f"""
        <!-- {unique_ts} -->
        <script>
            setTimeout(() => {{
                try {{
                    const ctx = new (window.AudioContext || window.webkitAudioContext)();
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = 'sine';
                    osc.frequency.value = {freq};
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    gain.gain.setValueAtTime(0.5, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                    osc.start();
                    osc.stop(ctx.currentTime + 0.6);
                    
                    if ('{alert_type}' === 'Critical' || '{alert_type}' === 'Outbreak') {{
                        setTimeout(() => {{
                            const osc2 = ctx.createOscillator();
                            const gain2 = ctx.createGain();
                            osc2.type = 'sine';
                            osc2.frequency.value = {freq + 200};
                            osc2.connect(gain2);
                            gain2.connect(ctx.destination);
                            gain2.gain.setValueAtTime(0.5, ctx.currentTime);
                            gain2.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                            osc2.start();
                            osc2.stop(ctx.currentTime + 0.6);
                        }}, 200);
                    }}
                }} catch(e) {{}}
            }}, 100);
        </script>
    """, height=0, width=0)
    
    st.session_state.play_alert = None

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: #0d1117; color: #e6edf3; }

/* Bring content to the top */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* Animated Tab Headers */
@keyframes textGlow {
    0% { text-shadow: 0 0 8px rgba(88, 166, 255, 0.4), 0 0 15px rgba(88, 166, 255, 0.2); color: #e6edf3; }
    50% { text-shadow: 0 0 20px rgba(88, 166, 255, 0.9), 0 0 35px rgba(88, 166, 255, 0.6); color: #ffffff; }
    100% { text-shadow: 0 0 8px rgba(88, 166, 255, 0.4), 0 0 15px rgba(88, 166, 255, 0.2); color: #e6edf3; }
}
@keyframes slideInGlow {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: translateY(0); }
}
.tab-container-animated {
    margin-bottom: 24px;
    animation: slideInGlow 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
.tab-header-glow {
    margin-bottom: 4px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    animation: textGlow 2.5s infinite ease-in-out;
}
.tab-subtitle {
    color: #8b949e !important; 
    font-size: 0.95rem !important;
}
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(248, 81, 73, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(248, 81, 73, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(248, 81, 73, 0); }
}
.live-indicator {
    background: #f85149;
    border-radius: 50%;
    width: 10px;
    height: 10px;
    display: inline-block;
    margin-right: 8px;
    animation: pulse 1.5s infinite;
}

/* Hide Streamlit element toolbar */
/* [data-testid="stToolbar"] { display: none; } */

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: transform .2s, border-color .2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: #58a6ff; }
.metric-title { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-size: 2rem; font-weight: 700; color: #e6edf3; margin: 4px 0; }
.metric-sub   { font-size: 0.82rem; color: #8b949e; }

/* Alert cards */
.alert-card {
    border-left: 4px solid;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    background: #161b22;
}
.alert-title   { font-weight: 600; font-size: 0.95rem; }
.alert-message { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
.alert-time    { font-size: 0.75rem; color: #6e7681; margin-top: 6px; }

/* Risk badges */
.badge-low    { background:#1a3a2a; color:#3fb950; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
.badge-medium { background:#3a2e00; color:#d29922; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
.badge-high   { background:#3d1a1a; color:#f85149; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }

/* Upload zone */
.upload-hint { color:#8b949e; font-size:.85rem; text-align:center; margin-top:8px; }

/* Section header */
.section-header {
    font-size:1.3rem; font-weight:700; color:#e6edf3;
    border-bottom:1px solid #30363d; padding-bottom:8px; margin-bottom:16px;
}

/* Sidebar Custom Styling */
.sidebar-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff 0%, #2ea043 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
    padding-bottom: 0px;
    letter-spacing: -0.5px;
}
.sidebar-subtitle {
    font-size: 0.8rem;
    color: #8b949e;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 24px;
}
.sidebar-card {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.sidebar-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 10px;
}
.sidebar-card-text {
    font-size: 0.85rem;
    color: #c9d1d9;
    line-height: 1.6;
}
.instruction-step {
    font-size: 0.82rem;
    color: #c9d1d9;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
}
.instruction-num {
    background: #30363d;
    color: #e6edf3;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: bold;
    margin-right: 10px;
    flex-shrink: 0;
    margin-top: 1px;
    border: 1px solid #484f58;
}

/* Glowing Streamlit Tabs Overrides */
div[data-testid="stTabs"] button { 
    color: #8b949e !important; 
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    background-color: transparent !important;
    border-radius: 8px 8px 0 0 !important;
}
div[data-testid="stTabs"] button:hover {
    color: #e6edf3 !important;
    text-shadow: 0 0 8px rgba(230,237,243,0.4) !important;
    background-color: rgba(255,255,255,0.02) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] { 
    color: #58a6ff !important; 
    text-shadow: 0 0 15px rgba(88,166,255,0.8), 0 0 30px rgba(88,166,255,0.4) !important;
    background-color: rgba(88,166,255,0.05) !important;
    border-bottom: none !important;
}
/* Style the moving underline indicator */
div[data-baseweb="tab-highlight"] {
    background-color: #58a6ff !important;
    box-shadow: 0 0 10px #58a6ff, 0 0 20px #58a6ff !important;
    height: 4px !important;
    border-radius: 2px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white; border: none; border-radius: 8px;
    padding: 10px 24px; font-weight: 700; width: 100%;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(46,160,67,0.2);
    transition: all 0.2s ease;
}
.stButton > button:hover { 
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(46,160,67,0.4);
    color: white;
}
.stButton > button:active {
    transform: translateY(0px);
}
</style>
""", unsafe_allow_html=True)

# ── Init DB on startup ─────────────────────────────────────────────────────────
if "db_ready" not in st.session_state:
    with st.spinner("Initialising database…"):
        init_db()
    st.session_state["db_ready"] = True

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">HealthSentinel</div>
    <div class="sidebar-subtitle">Intelligent Outbreak Monitor</div>
    
    <div class="sidebar-card">
        <div class="sidebar-card-title">About System</div>
        <div class="sidebar-card-text">
            An advanced early-warning engine that analyzes individual medical records to predict and visualize regional health trends in real-time.
        </div>
    </div>
    
    <div class="sidebar-card">
        <div class="sidebar-card-title">Quick Guide</div>
        <div class="instruction-step"><span class="instruction-num">1</span> Upload patient medical report</div>
        <div class="instruction-step"><span class="instruction-num">2</span> AI extracts health parameters</div>
        <div class="instruction-step"><span class="instruction-num">3</span> Monitor regional health KPIs</div>
        <div class="instruction-step"><span class="instruction-num">4</span> Track outbreak probabilities</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    REGIONS = ["North Zone", "South Zone", "East Zone", "West Zone", "Central Zone"]
    total = get_record_count()
    
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; padding: 0 4px;">
        <span style="color:#8b949e; font-size:0.75rem; font-weight:700; letter-spacing:1px;">TOTAL RECORDS</span>
        <span style="background:#2ea043; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:700; box-shadow: 0 0 10px rgba(46,160,67,0.3);">{total}</span>
    </div>
    <div style="display:flex; justify-content:space-between; align-items:center; padding: 0 4px;">
        <span style="color:#8b949e; font-size:0.75rem; font-weight:700; letter-spacing:1px;">ACTIVE REGIONS</span>
        <span style="background:#58a6ff; color:white; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:700; box-shadow: 0 0 10px rgba(88,166,255,0.3);">{len(REGIONS)}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card-title">System Controls</div>', unsafe_allow_html=True)
    st.session_state.live_sim = st.toggle("📡 Simulate Live Network", help="Auto-injects real-time patient data every 3 seconds to simulate an active hospital network.")

    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
        from modules.database import _get_conn
        conn = _get_conn()
        conn.execute("DELETE FROM health_records")
        conn.commit()
        conn.close()
        st.rerun()

    status_html = ""
    if st.session_state.get('live_sim', False):
        status_html = '<span class="live-indicator"></span><span style="color: #f85149; font-weight: bold; letter-spacing: 1px;">SIMULATION LIVE</span>'
    else:
        status_html = 'SYSTEM STATUS: <span style="color: #3fb950; font-weight: bold;">ONLINE</span>'

    st.markdown(f"""
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #30363d; text-align: center;">
            <p style="color: #6e7681; font-size: 0.8rem; letter-spacing: 0.5px;">{status_html}</p>
        </div>
    """, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_records, tab_analytics, tab_prediction, tab_alerts = st.tabs([
    "Upload Report",
    "Health Records",
    "Community Analytics",
    "Outbreak Prediction",
    "Alerts",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ════════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("""
        <div class="tab-container-animated">
            <h2 class="tab-header-glow">Patient Intake & Analysis</h2>
        </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        col_up, col_preview = st.columns([1, 1], gap="large")

        with col_up:
            uploaded = st.file_uploader(
                "Upload Medical Document",
                type=["jpg", "jpeg", "png", "pdf"],
                help="Supports JPG, PNG, PDF formats (Max 200MB)",
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            region_for_upload = st.selectbox("Assign Patient to Region", REGIONS, key="up_region")

            if uploaded:
                st.markdown('<p class="upload-hint">File received — ready for extraction</p>', unsafe_allow_html=True)

        with col_preview:
            st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)
            if uploaded:
                if uploaded.type.startswith("image"):
                    st.image(uploaded, caption="Uploaded Report", use_container_width=True)
                else:
                    st.info("PDF uploaded — OCR will process all pages.")
            else:
                st.markdown("""
                    <div style="height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 2px dashed #30363d; border-radius: 8px; background: #0d1117;">
                        <span style="font-size: 2rem; margin-bottom: 8px; color: #30363d;">📄</span>
                        <span style="color: #484f58; font-size: 0.9rem; font-weight: 500;">Document preview will appear here</span>
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if uploaded and st.button("Extract & Analyse Report", use_container_width=True):
        with st.spinner("Running OCR extraction…"):
            # Save to temp file
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                raw_text = extract_text(tmp_path)
            except Exception as e:
                raw_text = f"[OCR failed: {str(e)}]"
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        if not raw_text.strip() or raw_text.startswith("[OCR failed"):
            st.error("OCR could not extract text. Try a clearer image.")
        else:
            with st.spinner("Parsing health parameters…"):
                record = parse_health_parameters(raw_text, region_for_upload)
                record["risk_score"]  = compute_risk_score(record)
                record["risk_flag"]   = classify_risk(record["risk_score"])

                # Predict outbreak probability
                df_all = get_all_records()
                comm_stats = aggregate_community_stats(df_all) if not df_all.empty else {}
                record["outbreak_prob"] = predict_outbreak_prob(record, comm_stats)

                # Generate alerts
                alerts_now = generate_alerts(record, comm_stats, record["outbreak_prob"])

                # Save to DB
                record_id = insert_record(record)

            # ── Results ──────────────────────────────────────────────────────
            st.success(f"Report processed! Record ID: **{record_id}**  |  Patient: **{record['patient_id']}**")

            st.markdown("#### OCR Extracted Text")
            with st.expander("View raw OCR text"):
                st.code(raw_text[:2000], language=None)

            st.markdown("#### Extracted Health Parameters")
            params = {
                "Glucose (mg/dL)":   record.get("glucose"),
                "BP Systolic (mmHg)":record.get("bp_systolic"),
                "BP Diastolic (mmHg)":record.get("bp_diastolic"),
                "Hemoglobin (g/dL)": record.get("hemoglobin"),
                "Temperature (°F)":  record.get("temperature"),
                "Cholesterol (mg/dL)":record.get("cholesterol"),
            }
            cols = st.columns(3)
            for i, (label, val) in enumerate(params.items()):
                display = f"{val:.1f}" if val else "—"
                cols[i % 3].markdown(
                    f'<div class="metric-card"><div class="metric-title">{label}</div>'
                    f'<div class="metric-value">{display}</div></div>',
                    unsafe_allow_html=True,
                )

            r = record["risk_flag"]
            badge = f'<span class="badge-{r.lower()}">{r} Risk</span>'
            st.markdown(
                f"**Risk Score:** `{record['risk_score']:.1f} / 100`  &nbsp; {badge}  &nbsp; "
                f"**Outbreak Prob:** `{record['outbreak_prob']:.0%}`",
                unsafe_allow_html=True,
            )

            if record.get("symptoms"):
                st.markdown(f"**Symptoms:** `{record['symptoms']}`")

            if alerts_now:
                st.markdown("#### Triggered Alerts")
                for a in alerts_now:
                    st.markdown(
                        f'<div class="alert-card" style="border-color:{a["color"]}">'
                        f'<div class="alert-title">{a["title"]}</div>'
                        f'<div class="alert-message">{a["message"]}</div>'
                        f'<div class="alert-time">{a["timestamp"]}</div></div>',
                        unsafe_allow_html=True,
                    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — HEALTH RECORDS
# ════════════════════════════════════════════════════════════════════════════════
with tab_records:
    st.markdown("""
        <div class="tab-container-animated">
            <h2 class="tab-header-glow">Patient Database</h2>
        </div>
    """, unsafe_allow_html=True)

    df = get_all_records()
    if df.empty:
        st.info("No records yet. Upload a medical report to get started.")
    else:
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 2, 1], gap="medium")
            with c1:
                region_filter = st.selectbox("Filter by Region", ["All"] + REGIONS, key="rec_region")
            with c2:
                risk_filter = st.selectbox("Filter by Risk Level", ["All", "Low", "Medium", "High"], key="rec_risk")
            with c3:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("Refresh", use_container_width=True):
                    st.rerun()

        fdf = df.copy()
        if region_filter != "All":
            fdf = fdf[fdf["region"] == region_filter]
        if risk_filter != "All":
            fdf = fdf[fdf["risk_flag"] == risk_filter]

        # Clean up column names for display
        rename_map = {
            "patient_id": "Patient ID", "region": "Region", "glucose": "Glucose",
            "bp_systolic": "Sys BP", "bp_diastolic": "Dia BP", "hemoglobin": "Hb",
            "temperature": "Temp", "cholesterol": "Cholesterol", "risk_score": "Risk Score",
            "risk_flag": "Risk Level", "outbreak_prob": "Outbreak Prob",
            "symptoms": "Symptoms", "date": "Date"
        }
        fdf_display = fdf.rename(columns=rename_map)
        
        if "Outbreak Prob" in fdf_display.columns:
            fdf_display["Outbreak Prob"] = fdf_display["Outbreak Prob"] * 100

        display_cols = ["Patient ID", "Region", "Glucose", "Sys BP", "Dia BP", "Hb", "Temp", "Cholesterol", "Risk Score", "Risk Level", "Symptoms", "Date"]
        available_cols = [c for c in display_cols if c in fdf_display.columns]

        # Export Button
        csv = fdf_display[available_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Database to CSV",
            data=csv,
            file_name="health_records_export.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

        st.dataframe(
            fdf_display[available_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score",
                    help="Calculated patient risk score",
                    format="%.0f",
                    min_value=0,
                    max_value=100,
                )
            }
        )
        st.markdown(f"""
            <div style="text-align: right; color: #8b949e; font-size: 0.85rem; margin-top: 8px;">
                Showing <b style="color: #e6edf3;">{len(fdf)}</b> of <b style="color: #e6edf3;">{len(df)}</b> total records
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 32px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size:1rem; color:#e6edf3; margin-bottom:12px;'>🕸️ Patient Vital Radar Analysis</h4>", unsafe_allow_html=True)
        
        with st.container(border=True):
            p_cols = st.columns([1, 3])
            with p_cols[0]:
                selected_pid = st.selectbox("Select Patient ID", fdf["patient_id"].tolist())
                
                if selected_pid:
                    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                    st.markdown("<p style='color:#8b949e; font-size:0.85rem;'>The Radar Chart compares the selected patient's vitals against a clinically healthy baseline.<br><br>• <b style='color:#2ea043;'>Green Ring:</b> Normal Baseline (100%)<br>• <b style='color:#e6edf3;'>Spikes outside:</b> Elevated Danger Levels</p>", unsafe_allow_html=True)

            with p_cols[1]:
                if selected_pid:
                    patient_data = fdf[fdf["patient_id"] == selected_pid].iloc[0]
                    
                    # Healthy Baselines
                    baselines = {"Glucose": 100, "Sys BP": 120, "Dia BP": 80, "Temp": 98.6, "Cholesterol": 160}
                    categories = ["Glucose", "Sys BP", "Dia BP", "Temp", "Cholesterol"]
                    
                    patient_vals = [
                        patient_data["glucose"], patient_data["bp_systolic"],
                        patient_data["bp_diastolic"], patient_data["temperature"],
                        patient_data["cholesterol"]
                    ]
                    
                    # Normalize to % of healthy baseline
                    normalized_vals = [(val / baselines[cat]) * 100 for val, cat in zip(patient_vals, categories)]
                    
                    # Close the loop for radar
                    categories.append(categories[0])
                    normalized_vals.append(normalized_vals[0])
                    healthy_line = [100, 100, 100, 100, 100, 100]
                    
                    fig_radar = go.Figure()
                    
                    # Patient Trace
                    p_color = "#f85149" if patient_data["risk_flag"] in ["High", "Critical"] else ("#d29922" if patient_data["risk_flag"] == "Medium" else "#58a6ff")
                    r_col, g_col, b_col = int(p_color[1:3],16), int(p_color[3:5],16), int(p_color[5:7],16)
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=normalized_vals, theta=categories,
                        fill='toself', name=f'Patient {selected_pid}',
                        line_color=p_color, fillcolor=f"rgba({r_col},{g_col},{b_col}, 0.3)"
                    ))
                    
                    # Baseline Trace
                    fig_radar.add_trace(go.Scatterpolar(
                        r=healthy_line, theta=categories,
                        mode='lines', name='Healthy Baseline (100%)',
                        line=dict(color="#2ea043", dash="dash", width=2)
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(visible=True, range=[0, max(200, max(normalized_vals) + 10)], gridcolor="#30363d", tickfont=dict(color="#8b949e")),
                            angularaxis=dict(gridcolor="#30363d", tickfont=dict(color="#e6edf3", size=13))
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=30, b=30, l=30, r=30),
                        height=400,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(color="#c9d1d9"))
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMMUNITY ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("""
        <div class="tab-container-animated">
            <h2 class="tab-header-glow">Macro Analytics Dashboard</h2>
        </div>
    """, unsafe_allow_html=True)

    df = get_all_records()
    if df.empty:
        st.info("No data yet.")
    else:
        comm_stats = aggregate_community_stats(df)

        # ── KPI Rows ───────────────────────────────────────────────────────────
        total_cases   = len(df)
        high_risk     = len(df[df["risk_flag"] == "High"])
        avg_glucose   = df["glucose"].dropna().mean()
        avg_bp        = df["bp_systolic"].dropna().mean()

        # Calculate global anomaly count
        abnormal_count = sum([s.get("abnormal_count", 0) for s in comm_stats.values()])
        anomaly_rate = abnormal_count / max(total_cases, 1)

        # Calculate top symptom
        global_syms = {}
        for s in comm_stats.values():
            for sym, cnt in s.get("symptom_counts", {}).items():
                global_syms[sym] = global_syms.get(sym, 0) + cnt
        top_symptom = max(global_syms, key=global_syms.get) if global_syms else "None"


        avg_glucose_disp = f"{avg_glucose:.0f} <span style='font-size:1rem;color:#8b949e'>mg/dL</span>" if not pd.isna(avg_glucose) else "—"
        avg_bp_disp      = f"{avg_bp:.0f} <span style='font-size:1rem;color:#8b949e'>mmHg</span>" if not pd.isna(avg_bp) else "—"
        anomaly_disp     = f"{anomaly_rate:.0%}"

        st.markdown("<h4 style='font-size:0.8rem; color:#8b949e; margin-bottom:12px; text-transform:uppercase; letter-spacing:1px;'>Core Metrics</h4>", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        
        # Simulated trend calculations
        case_trend = f"<span style='color: #3fb950; font-weight:bold;'>⭡ +{max(1, int(total_cases * 0.05))}</span> in last hr" if total_cases > 10 else "across all regions"
        hr_trend = f"<span style='color: #f85149; font-weight:bold;'>⭡ +{max(1, int(high_risk * 0.1))}</span> detected" if high_risk > 0 else f"{high_risk/max(total_cases,1):.0%} of population"
        anomaly_trend = f"<span style='color: #d29922; font-weight:bold;'>⭡ +2.1%</span> vs yesterday" if total_cases > 5 else "records with ≥1 warning"

        for col, title, val, sub, color in [
            (k1, "Total Screened",   total_cases,             case_trend, "#58a6ff"),
            (k2, "Critical Risk",    high_risk,               hr_trend, "#f85149"),
            (k3, "Anomaly Rate",     anomaly_disp,            anomaly_trend, "#d29922"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="metric-title">{title}</div>'
                f'<div class="metric-value" style="color:{color}">{val}</div><div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

        k4, k5, k6 = st.columns(3)
        for col, title, val, sub, color in [
            (k4, "Mean Glucose",     avg_glucose_disp,        "community average", "#e6edf3"),
            (k5, "Mean Sys BP",      avg_bp_disp,             "community average", "#e6edf3"),
            (k6, "Top Symptom",      top_symptom.title(),     "most prevalent complaint", "#e6edf3"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="metric-title">{title}</div>'
                f'<div class="metric-value" style="color:{color}">{val}</div><div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size:0.8rem; color:#8b949e; margin-bottom:12px; text-transform:uppercase; letter-spacing:1px;'>Visualization Engine</h4>", unsafe_allow_html=True)
        
        ch1, ch2 = st.columns(2, gap="medium")

        # Chart 1 — Avg glucose by region
        with ch1:
            with st.container(border=True):
                st.markdown("<h4 style='font-size:1rem; color:#e6edf3; margin-bottom:0;'>Regional Glucose Averages</h4>", unsafe_allow_html=True)
                reg_df = df.groupby("region")["glucose"].mean().reset_index()
                fig = px.bar(
                    reg_df, x="region", y="glucose",
                    color="glucose",
                    color_continuous_scale=["#2ea043","#d29922","#f85149"],
                    labels={"glucose":"Avg Glucose (mg/dL)","region":""},
                    template="plotly_dark",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False, margin=dict(t=20,b=10,l=0,r=0),
                    height=300
                )
                fig.update_traces(marker_line_width=0, opacity=0.9)
                fig.add_hline(y=126, line_dash="dash", line_color="#f85149", annotation_text="Diabetic threshold", annotation_font_size=10)
                st.plotly_chart(fig, use_container_width=True)

        # Chart 2 — Risk distribution pie
        with ch2:
            with st.container(border=True):
                st.markdown("<h4 style='font-size:1rem; color:#e6edf3; margin-bottom:0;'>Population Risk Distribution</h4>", unsafe_allow_html=True)
                risk_counts = df["risk_flag"].value_counts().reset_index()
                risk_counts.columns = ["Risk","Count"]
                fig2 = px.pie(
                    risk_counts, names="Risk", values="Count",
                    color="Risk",
                    color_discrete_map={"Low":"#2ea043","Medium":"#d29922","High":"#f85149"},
                    template="plotly_dark",
                    hole=0.65,
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20,b=10,l=0,r=0),
                    height=300,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
                )
                fig2.update_traces(textposition='inside', textinfo='percent', hoverinfo='label+value+percent', marker=dict(line=dict(color='#0d1117', width=2)))
                st.plotly_chart(fig2, use_container_width=True)

        ch3, ch4 = st.columns(2, gap="medium")

        # Chart 3 — Trend over time
        with ch3:
            with st.container(border=True):
                st.markdown("<h4 style='font-size:1rem; color:#e6edf3; margin-bottom:0;'>Longitudinal Glucose Trends</h4>", unsafe_allow_html=True)
                tdf = df.dropna(subset=["glucose","date"]).copy()
                tdf["date"] = pd.to_datetime(tdf["date"])
                trend = tdf.groupby(["date","region"])["glucose"].mean().reset_index()
                fig3 = px.line(
                    trend, x="date", y="glucose", color="region",
                    labels={"glucose":"Avg Glucose","date":""},
                    template="plotly_dark",
                )
                fig3.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                    margin=dict(t=20,b=10,l=0,r=0), height=300,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                fig3.update_traces(line=dict(width=3))
                st.plotly_chart(fig3, use_container_width=True)

        # Chart 4 — Abnormal ratio heatmap
        with ch4:
            with st.container(border=True):
                st.markdown("<h4 style='font-size:1rem; color:#e6edf3; margin-bottom:0;'>Regional Parameter Heatmap</h4>", unsafe_allow_html=True)
                params = ["glucose","bp_systolic","hemoglobin","temperature","cholesterol"]
                thresholds = {"glucose":126,"bp_systolic":140,"hemoglobin":11,"temperature":99.5,"cholesterol":200}
                heat_data = {}
                for p in params:
                    heat_data[p] = {
                        r: int((df[df["region"]==r][p].dropna() > thresholds[p]).sum())
                        for r in REGIONS
                    }
                heat_df = pd.DataFrame(heat_data, index=REGIONS)
                fig4 = px.imshow(
                    heat_df.T,
                    color_continuous_scale=["#0d1117","#d29922","#f85149"],
                    labels=dict(x="", y="", color="Abnormal Cases"),
                    template="plotly_dark",
                    aspect="auto",
                )
                fig4.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20,b=10,l=0,r=0), 
                    height=300,
                    coloraxis_colorbar=dict(title="", thicknessmode="pixels", thickness=10, lenmode="pixels", len=150)
                )
                st.plotly_chart(fig4, use_container_width=True)
                
        ch5, ch6 = st.columns(2, gap="medium")

        # Chart 5 — Top Symptoms Bar Chart
        with ch5:
            with st.container(border=True):
                st.markdown("<h4 style='font-size:1rem; color:#e6edf3; margin-bottom:0;'>Symptom Frequency Analysis</h4>", unsafe_allow_html=True)
                if global_syms:
                    sym_df = pd.DataFrame(list(global_syms.items()), columns=["Symptom", "Count"]).sort_values("Count", ascending=True).tail(5)
                    sym_df["Symptom"] = sym_df["Symptom"].str.title()
                    fig5 = px.bar(
                        sym_df, x="Count", y="Symptom", orientation='h',
                        color="Count", color_continuous_scale=["#1f6feb", "#58a6ff"],
                        template="plotly_dark",
                        labels={"Count": "Frequency", "Symptom": ""}
                    )
                    fig5.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False, margin=dict(t=20,b=10,l=0,r=0),
                        height=300
                    )
                    fig5.update_traces(marker_line_width=0, opacity=0.9)
                    st.plotly_chart(fig5, use_container_width=True)
                else:
                    st.info("No symptoms recorded.")

        # Chart 6 — Glucose vs BP Scatter
        with ch6:
            with st.container(border=True):
                st.markdown("<h4 style='font-size:1rem; color:#e6edf3; margin-bottom:0;'>Glucose vs. Systolic BP Correlation</h4>", unsafe_allow_html=True)
                scatter_df = df.dropna(subset=["glucose", "bp_systolic", "risk_flag"]).copy()
                if not scatter_df.empty:
                    fig6 = px.scatter(
                        scatter_df,
                        x="glucose", y="bp_systolic", color="risk_flag",
                        color_discrete_map={"Low":"#2ea043","Medium":"#d29922","High":"#f85149"},
                        labels={"glucose":"Glucose (mg/dL)","bp_systolic":"Systolic BP (mmHg)"},
                        template="plotly_dark", opacity=0.8
                    )
                    fig6.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=20,b=10,l=0,r=0), height=300,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, title="")
                    )
                    # Add threshold lines
                    fig6.add_vline(x=126, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    fig6.add_hline(y=140, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                    st.plotly_chart(fig6, use_container_width=True)
                else:
                    st.info("Not enough data.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — OUTBREAK PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab_prediction:
    st.markdown("""
        <div class="tab-container-animated">
            <h2 class="tab-header-glow">Geospatial Threat Matrix</h2>
        </div>
    """, unsafe_allow_html=True)

    df = get_all_records()
    if df.empty:
        st.info("No data yet.")
    else:
        comm_stats = aggregate_community_stats(df)
        probs      = predict_all_regions(comm_stats)

        # Global threat level
        max_prob = max(probs.values()) if probs else 0.0
        threat_level = "CRITICAL" if max_prob >= 0.6 else ("ELEVATED" if max_prob >= 0.35 else "NOMINAL")
        threat_color = "#f85149" if max_prob >= 0.6 else ("#d29922" if max_prob >= 0.35 else "#2ea043")
        
        st.markdown(f"""
            <div style="background: linear-gradient(90deg, rgba(22,27,34,1) 0%, rgba(13,17,23,0) 100%); border-left: 4px solid {threat_color}; padding: 16px 20px; border-radius: 4px; margin-bottom: 24px;">
                <div style="font-size: 0.75rem; color: #8b949e; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;">System Threat Level</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: {threat_color}; margin-top: 4px;">{threat_level}</div>
                <div style="font-size: 0.85rem; color: #c9d1d9; margin-top: 4px;">Peak regional outbreak probability detected at <b>{max_prob:.1%}</b></div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<h4 style='font-size:1rem; color:#e6edf3; margin-bottom:12px;'>Regional Risk Projections</h4>", unsafe_allow_html=True)
        
        # Gauge charts — one per region
        gauge_cols = st.columns(len(REGIONS), gap="small")
        for i, region in enumerate(REGIONS):
            with gauge_cols[i]:
                with st.container(border=True):
                    prob = probs.get(region, 0.0)
                    color = "#2ea043" if prob < 0.35 else ("#d29922" if prob < 0.60 else "#f85149")

                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(prob * 100, 1),
                        number={"suffix": "%", "font": {"color": color, "size": 24, "family": "Inter"}},
                        title={"text": region.replace(" Zone",""), "font": {"size": 13, "color": "#c9d1d9", "family": "Inter"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#30363d", "tickwidth": 1},
                            "bar":  {"color": color, "thickness": 0.85},
                            "bgcolor": "rgba(0,0,0,0)",
                            "bordercolor": "rgba(0,0,0,0)",
                            "steps": [
                                {"range": [0,  35], "color": "rgba(46,160,67,0.15)"},
                                {"range": [35, 60], "color": "rgba(210,153,34,0.15)"},
                                {"range": [60,100], "color": "rgba(248,81,73,0.15)"},
                            ],
                            "threshold": {
                                "line": {"color": "#f85149", "width": 2},
                                "thickness": 0.85, "value": 60,
                            },
                        },
                    ))
                    fig_g.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=160,
                        margin=dict(t=30, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size:1rem; color:#e6edf3; margin-bottom:12px;'>Geospatial Risk Distribution</h4>", unsafe_allow_html=True)

        coords = {
            "Central Zone": {"lat": 12.9716, "lon": 77.5946}, # Majestic/MG Road
            "North Zone":   {"lat": 13.0285, "lon": 77.5896}, # Hebbal
            "South Zone":   {"lat": 12.9141, "lon": 77.5806}, # Jayanagar
            "East Zone":    {"lat": 12.9784, "lon": 77.6408}, # Indiranagar
            "West Zone":    {"lat": 12.9851, "lon": 77.5451}, # Rajajinagar
        }
        
        map_data = []
        for r in REGIONS:
            p = probs.get(r, 0.0) * 100
            map_data.append({
                "Region": r,
                "Lat": coords[r]["lat"],
                "Lon": coords[r]["lon"],
                "Outbreak Risk (%)": max(p, 5.0), # Minimum size for visibility
                "Actual Risk": p,
                "Cases": comm_stats.get(r, {}).get("case_count", 0),
            })
        
        map_df = pd.DataFrame(map_data)
        
        with st.container(border=True):
            fig_map = px.scatter_mapbox(
                map_df, lat="Lat", lon="Lon",
                size="Outbreak Risk (%)", color="Actual Risk",
                color_continuous_scale=["#2ea043", "#d29922", "#f85149"],
                range_color=[0, 100], hover_name="Region",
                hover_data={"Lat":False, "Lon":False, "Outbreak Risk (%)":False, "Actual Risk":':.1f', "Cases":True},
                zoom=10.5, center={"lat": 12.9716, "lon": 77.5946},
                mapbox_style="carto-darkmatter", height=400
            )
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)
        st.markdown("<h4 style='font-size:1rem; color:#e6edf3; margin-bottom:8px;'>Prediction Details Matrix</h4>", unsafe_allow_html=True)

        # Table summary
        summary = []
        for region in REGIONS:
            prob = probs.get(region, 0.0)
            s    = comm_stats.get(region, {})
            summary.append({
                "Region":           region,
                "Outbreak Risk":    prob * 100,
                "Case Volume":      s.get("case_count", 0),
                "Abnormal Ratio":   s.get("abnormal_ratio", 0) * 100,
                "Avg Glucose":      s.get("avg_glucose", 0),
                "Status":           "CRITICAL" if prob >= 0.60 else ("WATCH" if prob >= 0.35 else "NORMAL"),
            })
            
        sum_df = pd.DataFrame(summary)
        st.dataframe(
            sum_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Outbreak Risk": st.column_config.ProgressColumn(
                    "Outbreak Risk",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Abnormal Ratio": st.column_config.NumberColumn(
                    "Abnormal Ratio",
                    format="%.1f%%",
                ),
                "Avg Glucose": st.column_config.NumberColumn(
                    "Avg Glucose",
                    format="%.0f mg/dL",
                )
            }
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — ALERTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_alerts:
    st.markdown("""
        <div class="tab-container-animated">
            <h2 class="tab-header-glow">Active Threat Monitoring</h2>
        </div>
    """, unsafe_allow_html=True)

    df = get_all_records()
    if df.empty:
        st.info("No data to generate alerts from.")
    else:
        comm_stats = aggregate_community_stats(df)
        probs      = predict_all_regions(comm_stats)
        all_alerts = get_community_level_alerts(comm_stats, probs)

        if not all_alerts:
            st.markdown("""
                <div style="display:flex; align-items:center; padding: 24px; background: rgba(46,160,67,0.1); border: 1px solid #2ea043; border-radius: 8px;">
                    <div style="font-size: 1.5rem; font-weight: 800; color: #3fb950; margin-right: 16px; border: 2px solid #3fb950; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center;">✓</div>
                    <div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #3fb950;">System Secure</div>
                        <div style="color: #c9d1d9; font-size: 0.9rem;">No active alerts or outbreaks detected. Community health looks stable.</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            sev_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
            all_alerts.sort(key=lambda a: sev_order.get(a.get("severity","Low"), 3))

            outbreak_alerts  = [a for a in all_alerts if a["type"] == "outbreak"]
            community_alerts = [a for a in all_alerts if a["type"] == "community"]
            symptom_alerts   = [a for a in all_alerts if a["type"] == "symptom_cluster"]

            # Summary metrics
            st.markdown(f"""
            <div style="display:flex; gap: 12px; margin-bottom: 24px;">
                <div style="flex:1; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center;">
                    <div style="color: #f85149; font-size: 2rem; font-weight: 800;">{len(outbreak_alerts)}</div>
                    <div style="color: #8b949e; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Outbreak Warnings</div>
                </div>
                <div style="flex:1; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center;">
                    <div style="color: #d29922; font-size: 2rem; font-weight: 800;">{len(community_alerts)}</div>
                    <div style="color: #8b949e; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Community Risks</div>
                </div>
                <div style="flex:1; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center;">
                    <div style="color: #58a6ff; font-size: 2rem; font-weight: 800;">{len(symptom_alerts)}</div>
                    <div style="color: #8b949e; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Symptom Clusters</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for section, items, s_color in [
                ("CRITICAL OUTBREAK WARNINGS", outbreak_alerts, "#f85149"),
                ("COMMUNITY RISK ALERTS", community_alerts, "#d29922"),
                ("SYMPTOM CLUSTERS", symptom_alerts, "#58a6ff"),
            ]:
                if items:
                    st.markdown(f"""
                        <h4 style='font-size:0.85rem; color:{s_color}; margin-bottom:12px; margin-top:24px; border-bottom: 1px solid #30363d; padding-bottom: 4px; letter-spacing: 1px;'>{section}</h4>
                    """, unsafe_allow_html=True)
                    
                    for a in items:
                        bg_color = "rgba(255,255,255,0.05)"
                        if a["color"].startswith("#"):
                            h = a["color"].lstrip('#')
                            if len(h) == 6:
                                rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                                bg_color = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08)"
                        
                        st.markdown(
                            f"""
                            <div style="border-left: 4px solid {a['color']}; border-radius: 6px; padding: 16px; margin-bottom: 12px; background: {bg_color}; border-top: 1px solid #30363d; border-right: 1px solid #30363d; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: flex-start;">
                                <div>
                                    <div style="font-weight: 700; font-size: 1.05rem; color: #e6edf3; margin-bottom: 4px;">{a["title"]}</div>
                                    <div style="font-size: 0.9rem; color: #c9d1d9;">{a["message"]}</div>
                                    <div style="font-size: 0.75rem; color: #6e7681; margin-top: 8px;">Generated: {a["timestamp"]}</div>
                                </div>
                                <div style="background: {a['color']}; color: #0d1117; padding: 2px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;">
                                    {a["severity"]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ════════════════════════════════════════════════════════════════════════════════
# SIMULATE LIVE NETWORK
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state.get("live_sim", False):
    import time
    import random
    from datetime import date
    from modules.database import _get_conn, SYMPTOM_POOL

    # Generate 1 synthetic record
    region = random.choice(REGIONS)
    rm = {"North Zone": 0.25, "South Zone": 0.65, "East Zone": 0.20, "West Zone": 0.72, "Central Zone": 0.45}[region]
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
    
    # We must compute risk right here
    from modules.risk_scoring import compute_risk_score, classify_risk
    risk_score = compute_risk_score(rec)
    risk_flag  = classify_risk(risk_score)

    conn = _get_conn()
    conn.execute("""
        INSERT INTO health_records
            (patient_id, region, glucose, bp_systolic, bp_diastolic,
             hemoglobin, temperature, cholesterol, symptoms,
             risk_score, risk_flag, outbreak_prob, date, raw_text)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (f"LIVE-{random.randint(10000,99999)}", region, glucose, bp_sys, bp_dia, hemoglobin, temperature, cholesterol, symptoms, risk_score, risk_flag, 0.0, date.today().isoformat(), "Auto-generated live simulation record"))
    conn.commit()
    conn.close()

    if risk_flag == "Critical":
        st.session_state.play_alert = ("Critical", f"CRITICAL PATIENT: {region} just reported a severe health anomaly.")
    elif risk_flag == "High":
        st.session_state.play_alert = ("High", f"HIGH RISK: Elevated vitals detected in {region}.")
    elif risk_flag == "Medium":
        st.session_state.play_alert = ("Medium", f"WATCH: Moderate anomaly detected in {region}.")
        
    df_live = get_all_records()
    from modules.risk_scoring import aggregate_community_stats
    from ml.model_utils import predict_all_regions
    c_stats = aggregate_community_stats(df_live)
    c_probs = predict_all_regions(c_stats)
    for r_name, p_val in c_probs.items():
        if p_val > 0.6:
            st.session_state.play_alert = ("Outbreak", f"OUTBREAK WARNING: {r_name} has exceeded critical threshold ({p_val*100:.1f}%)")
            break

    time.sleep(3)
    st.rerun()
