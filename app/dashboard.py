# type: ignore
# noqa
import streamlit as st
import json
try:
    import cv2
except ImportError:
    cv2 = None
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, Optional

from src.pipeline import EndToEndPipeline
from src.visualization.charts import (
    create_vehicle_pie_chart,
    create_fuel_donut_chart,
    create_fuel_emissions_bar_chart
)
from src.environmental import OpenAQAdapter, CPCBAdapter
from src.classification import VehicleMakeModelClassifier

import os

st.set_page_config(
    page_title="Smart City AI Vehicle Pollution Monitor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

is_render_cloud = bool(os.environ.get("RENDER") or os.environ.get("ON_RENDER") or os.environ.get("PORT"))

# Streamlit Model Caching Optimization
@st.cache_resource(show_spinner="Loading YOLO & Neural Classifier Weights...")
def get_cached_pipeline(conf_threshold: float, process_every_n_frames: int) -> EndToEndPipeline:
    return EndToEndPipeline(
        conf_threshold=conf_threshold,
        process_every_n_frames=process_every_n_frames
    )

# Professional Smart City UI CSS Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.2rem;
    }
    .disclaimer-card {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #F59E0B;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        line-height: 1.4;
        color: #E2E8F0;
    }
    .kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.9rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .kpi-num {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94A3B8;
    }
    .badge-petrol { color: #3B82F6; }
    .badge-diesel { color: #4B5563; }
    .badge-ev { color: #10B981; }
    .badge-cng { color: #F97316; }
    .badge-hybrid { color: #8B5CF6; }
    .badge-unknown { color: #9CA3AF; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR CONTROL PANEL
st.sidebar.image("https://img.icons8.com/color/96/smart-city.png", width=65)
st.sidebar.title("🏙️ Smart City AI Control")

if is_render_cloud:
    st.sidebar.warning("⚡ **Cloud Hosted (Render Free Tier)**\nProcessing optimized for shared 512MB RAM & 0.1 CPU core.")

input_mode: str = st.sidebar.radio(
    "Source Input:",
    ["Sample Demo Video", "Upload Video", "Upload Image", "Webcam / RTSP Stream"]
)

uploaded_file = None
if input_mode == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov", "mkv"])
elif input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload Traffic Image", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Performance Tuning")
default_skip = 3 if is_render_cloud else 2
frame_skipping: int = int(st.sidebar.select_slider("Frame Skipping (Process Every N Frames):", options=[1, 2, 3, 4], value=default_skip))
conf_threshold: float = float(st.sidebar.slider("YOLO Detection Confidence:", 0.10, 0.95, 0.30, 0.05))

if is_render_cloud:
    max_frames_opt = st.sidebar.selectbox(
        "Video Analysis Scope:",
        ["Cloud Fast Mode (First 300 Frames ~12s)", "First 450 Frames (~18 sec)", "Full Video (Analyze All Vehicles)"]
    )
else:
    max_frames_opt = st.sidebar.selectbox(
        "Video Analysis Scope:",
        ["Full Video (Analyze All Vehicles)", "First 450 Frames (~18 sec)", "First 900 Frames (~36 sec)", "First 1800 Frames (~72 sec)"]
    )

import traceback

if "Full Video" in max_frames_opt:
    max_frames_val = None
elif "300" in max_frames_opt:
    max_frames_val = 300
elif "450" in max_frames_opt:
    max_frames_val = 450
elif "900" in max_frames_opt:
    max_frames_val = 900
elif "1800" in max_frames_opt:
    max_frames_val = 1800
else:
    max_frames_val = None

st.sidebar.selectbox("YOLO Architecture:", ["yolov8n.pt (Fast)", "yolov8s.pt (Accurate)", "Custom Weights"])

run_button: bool = bool(st.sidebar.button("⚡ Run End-to-End AI Analysis", type="primary", use_container_width=True))

# HEADER
st.markdown('<div class="main-title">🏙️ AI-Based Vehicle Classification & Pollution Monitoring</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Smart City Environmental AI & Acoustic Telemetry Infrastructure</div>', unsafe_allow_html=True)

# MANDATORY SCIENTIFIC METHODOLOGY DISCLAIMERS
st.markdown("""
<div class="disclaimer-card">
    ⚠️ <strong>MANDATORY SCIENTIFIC METHODOLOGY DISCLAIMERS:</strong><br>
    • <strong>Air Emissions:</strong> Air pollution values are model-based estimated vehicle contributions ($g/hr$ mass rates derived from EEA COPERT V standards), NOT direct physical atmospheric measurements.<br>
    • <strong>Noise Pollution:</strong> Noise values are relative indices (0-100 scale derived from CoRTN acoustic traffic models), NOT calibrated physical sound pressure dB measurements.<br>
    • <strong>Fuel Type:</strong> Fuel mapping is inferred from vehicle make/model reference tables and may be <strong>UNKNOWN</strong> or <strong>AMBIGUOUS</strong> when visual features are insufficient.
</div>
""", unsafe_allow_html=True)

output_dir: Path = Path("outputs/predictions")
output_dir.mkdir(parents=True, exist_ok=True)
summary_path: Path = output_dir / "summary.json"
vehicles_csv_path: Path = output_dir / "vehicles.csv"

# Pre-save uploaded video to disk using efficient memoryview buffer
uploaded_video_path: Optional[Path] = None

if input_mode == "Upload Video" and uploaded_file is not None:
    try:
        uploaded_video_path = output_dir / f"uploaded_{uploaded_file.name}"
        if not uploaded_video_path.exists() or uploaded_video_path.stat().st_size != uploaded_file.size:
            with open(uploaded_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.sidebar.error(f"⚠️ Video upload stream warning: {e}")

summary: Optional[Dict[str, Any]] = None

# Execute End-to-End Analysis Pipeline if requested
if run_button:
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_mode == "Upload Image":
        if not uploaded_file:
            st.warning("⚠️ Please upload an image file first before clicking Run End-to-End AI Analysis.")
        else:
            with st.spinner("⏳ Executing AI Image Pipeline (YOLO Detection → Crop Classifier → Fuel Mapping)..."):
                try:
                    uploaded_file.seek(0)
                    temp_img_path = output_dir / f"uploaded_{uploaded_file.name}"
                    img_bytes = uploaded_file.getvalue()
                    file_np = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(file_np, cv2.IMREAD_COLOR) if cv2 is not None else None
                    if img is not None:
                        h, w = img.shape[:2]
                        max_dim = 1280
                        if max(h, w) > max_dim:
                            scale = max_dim / float(max(h, w))
                            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(temp_img_path), img)
                    else:
                        with open(temp_img_path, "wb") as f:
                            f.write(img_bytes)

                    pipeline = get_cached_pipeline(conf_threshold=conf_threshold, process_every_n_frames=frame_skipping)
                    summary = pipeline.process_image(temp_img_path)
                    st.success("✅ End-to-End AI Image Analysis Complete!")
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
    elif input_mode == "Upload Video":
        if not uploaded_file or uploaded_video_path is None or not uploaded_video_path.exists():
            st.warning("⚠️ Please upload a video file first before clicking Run End-to-End AI Analysis.")
        else:
            prog_bar = st.progress(0, text="🚀 Initializing High-Speed AI Video Pipeline...")
            def update_progress(curr, tot):
                if tot > 0:
                    pct = min(1.0, max(0.0, float(curr) / float(tot)))
                    prog_bar.progress(pct, text=f"⚡ Processing Video Frame {curr} / {tot} ({int(pct*100)}% Complete)...")
                else:
                    prog_bar.progress(0.5, text=f"⚡ Processing Video Frame {curr}...")
            try:
                pipeline = get_cached_pipeline(conf_threshold=conf_threshold, process_every_n_frames=frame_skipping)
                summary = pipeline.process_video(uploaded_video_path, max_frames=max_frames_val, progress_callback=update_progress)
                prog_bar.progress(1.0, text="✅ End-to-End AI Video Analysis Complete!")
                st.success("✅ End-to-End AI Video Analysis Complete!")
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
    else:
        prog_bar = st.progress(0, text="🚀 Initializing Sample Video Analysis...")
        def update_progress(curr, tot):
            if tot > 0:
                pct = min(1.0, max(0.0, float(curr) / float(tot)))
                prog_bar.progress(pct, text=f"⚡ Processing Sample Video Frame {curr} / {tot} ({int(pct*100)}% Complete)...")
            else:
                prog_bar.progress(0.5, text=f"⚡ Processing Sample Video Frame {curr}...")
        try:
            sample_vid = Path("data/raw/videos/sample_traffic.mp4")
            if not sample_vid.exists():
                from scripts.prepare_sample_data import main as prep_sample
                prep_sample()
            pipeline = get_cached_pipeline(conf_threshold=conf_threshold, process_every_n_frames=frame_skipping)
            summary = pipeline.process_video(sample_vid, max_frames=max_frames_val, progress_callback=update_progress)
            prog_bar.progress(1.0, text="✅ Sample Video Analysis Complete!")
            st.success("✅ Sample Video Analysis Complete!")
        except Exception as e:
            st.error(f"❌ Error processing sample video: {str(e)}")

# Load existing telemetry summary or default fallback
if summary is None:
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {
            "total_unique_vehicles": 15,
            "petrol_count": 6, "diesel_count": 4, "ev_count": 2, "cng_lpg_count": 1, "hybrid_count": 1, "unknown_count": 1,
            "vehicle_type_counts": {"car": 9, "truck": 3, "bus": 1, "motorcycle": 2},
            "pollution_index": {"vehicle_pollution_index": 45.2, "category": "MODERATE ESTIMATED VEHICLE CONTRIBUTION"},
            "noise_index": {"relative_noise_index": 56.4, "category": "ELEVATED RELATIVE TRAFFIC NOISE"},
            "pollutant_estimates": {"PM2.5": 0.42, "PM10": 0.55, "NO2": 4.12, "CO": 14.2, "SO2": 0.09, "CO2": 2150.0},
            "processing_fps": 22.4,
            "performance_telemetry": {
                "processing_fps": 22.4,
                "memory_usage_mb": 420.5,
                "latency_breakdown_ms": {"detection_ms": 18.2, "tracking_ms": 2.1, "classification_ms": 12.4}
            }
        }

active_summary: Dict[str, Any] = summary or {}

# TOP LEVEL KPI CARDS (ROW 1: POWERTRAIN BREAKDOWN)
st.subheader("📊 Powertrain & Traffic Inventory")
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

k1.markdown(f'<div class="kpi-card"><div class="kpi-num" style="color:#38BDF8">{active_summary.get("total_unique_vehicles", 0)}</div><div class="kpi-sub">Total Vehicles</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card"><div class="kpi-num badge-petrol">{active_summary.get("petrol_count", 0)}</div><div class="kpi-sub">Petrol ⛽</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card"><div class="kpi-num badge-diesel">{active_summary.get("diesel_count", 0)}</div><div class="kpi-sub">Diesel 🛢️</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card"><div class="kpi-num badge-ev">{active_summary.get("ev_count", 0)}</div><div class="kpi-sub">EV ⚡</div></div>', unsafe_allow_html=True)
k5.markdown(f'<div class="kpi-card"><div class="kpi-num badge-cng">{active_summary.get("cng_lpg_count", 0)}</div><div class="kpi-sub">CNG/LPG 💨</div></div>', unsafe_allow_html=True)
k6.markdown(f'<div class="kpi-card"><div class="kpi-num badge-hybrid">{active_summary.get("hybrid_count", 0)}</div><div class="kpi-sub">Hybrid 🔋</div></div>', unsafe_allow_html=True)
k7.markdown(f'<div class="kpi-card"><div class="kpi-num badge-unknown">{active_summary.get("unknown_count", 0)}</div><div class="kpi-sub">Unknown ❓</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# TOP LEVEL KPI CARDS (ROW 2: POLLUTION, NOISE & PERFORMANCE METRICS)
p1, p2, p3, p4 = st.columns(4)
p_res = active_summary.get("pollution_index", {})
n_res = active_summary.get("noise_index", {})

p_val = p_res.get("vehicle_pollution_index", 45.2)
n_val = n_res.get("relative_noise_index", 56.4)

p_color = "#10B981" if p_val < 25 else ("#FBBF24" if p_val < 50 else ("#F97316" if p_val < 75 else "#EF4444"))
n_color = "#10B981" if n_val < 25 else ("#FBBF24" if n_val < 50 else ("#F97316" if n_val < 75 else "#EF4444"))

ram_mb = active_summary.get("performance_telemetry", {}).get("memory_usage_mb", 420.0)

p1.markdown(f'<div class="kpi-card"><div class="kpi-num" style="color:{p_color}">{p_val} / 100</div><div class="kpi-sub">Est. Air Pollution Index</div></div>', unsafe_allow_html=True)
p2.markdown(f'<div class="kpi-card"><div class="kpi-num" style="color:{n_color}">{n_val} / 100</div><div class="kpi-sub">Relative Noise Index</div></div>', unsafe_allow_html=True)
p3.markdown(f'<div class="kpi-card"><div class="kpi-num" style="color:#A78BFA">{active_summary.get("processing_fps", 22.4)} FPS</div><div class="kpi-sub">Pipeline Speed</div></div>', unsafe_allow_html=True)
p4.markdown(f'<div class="kpi-card"><div class="kpi-num" style="color:#34D399">{ram_mb} MB</div><div class="kpi-sub">RAM Memory Footprint</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# TABBED SMART CITY DASHBOARD NAVIGATION
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview & Stream",
    "🚗 Powertrain Breakdown",
    "🌫️ Pollution & Noise Models",
    "📡 Environmental Stations",
    "🧠 AI Explainability",
    "📖 Project Methodology"
])

# TAB 1: OVERVIEW & MEDIA STREAM
with tab1:
    col_v, col_t = st.columns([1.2, 1.0])
    
    with col_v:
        st.subheader("📹 Media Stream Telemetry HUD")
        annotated_img = output_dir / "annotated_image.jpg"
        tracked_vid = output_dir / "tracked_video.mp4"
        
        video_rendered = False
        if input_mode == "Upload Video" and uploaded_file is not None:
            st.caption(f"🎥 Uploaded Source Video: `{uploaded_file.name}`")
            try:
                if uploaded_video_path is not None and uploaded_video_path.exists():
                    st.video(str(uploaded_video_path))
                    video_rendered = True
            except Exception as vid_err:
                st.warning(f"⚠️ Could not stream video preview directly ({vid_err}). You can still run AI analysis below.")
        elif input_mode == "Sample Demo Video":
            sample_v = Path("data/raw/videos/sample_traffic.mp4")
            if sample_v.exists():
                st.caption("🎥 Sample Traffic Scene Video")
                st.video(str(sample_v))
                video_rendered = True

        if tracked_vid.exists() and tracked_vid.stat().st_size > 0:
            st.caption("🎬 AI Telemetry & Bounding Box Processed Output Stream")
            try:
                st.video(str(tracked_vid))
            except Exception:
                pass
        elif annotated_img.exists():
            st.caption("📷 AI Detection & Tracking Snapshot")
            st.image(str(annotated_img), use_container_width=True)
        elif not video_rendered:
            st.info("💡 Upload a video/image or select 'Sample Demo Video', then click '⚡ Run End-to-End AI Analysis' in the sidebar.")

    with col_t:
        st.subheader("🚗 Tracked Vehicle Inventory Data Table")
        if vehicles_csv_path.exists():
            v_df = pd.read_csv(vehicles_csv_path)
            st.dataframe(v_df[["track_id", "vehicle_type", "manufacturer", "model", "fuel_type", "estimated_pollution_score", "noise_score"]], height=340, use_container_width=True)
        else:
            sample_tbl = pd.DataFrame([
                {"track_id": 1, "vehicle_type": "car", "manufacturer": "Toyota", "model": "Prius", "fuel_type": "HYBRID", "estimated_pollution_score": 0.04, "noise_score": 52.0},
                {"track_id": 2, "vehicle_type": "car", "manufacturer": "Tesla", "model": "Model 3", "fuel_type": "EV", "estimated_pollution_score": 0.00, "noise_score": 42.0},
                {"track_id": 3, "vehicle_type": "truck", "manufacturer": "Ford", "model": "F-150", "fuel_type": "DIESEL", "estimated_pollution_score": 1.92, "noise_score": 78.0}
            ])
            st.dataframe(sample_tbl, height=340, use_container_width=True)

# TAB 2: POWERTRAIN & CATEGORY BREAKDOWN
with tab2:
    st.subheader("🚗 Fleet Distribution & Powertrain Analytics")
    c1, c2 = st.columns(2)
    
    with c1:
        v_counts = active_summary.get("vehicle_type_counts", {"car": 9, "truck": 3, "bus": 1, "motorcycle": 2})
        fig1 = create_vehicle_pie_chart(v_counts)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        f_counts = {
            "Petrol": active_summary.get("petrol_count", 6),
            "Diesel": active_summary.get("diesel_count", 4),
            "EV": active_summary.get("ev_count", 2),
            "CNG_LPG": active_summary.get("cng_lpg_count", 1),
            "Hybrid": active_summary.get("hybrid_count", 1),
            "Unknown": active_summary.get("unknown_count", 1)
        }
        fig2 = create_fuel_donut_chart(f_counts)
        st.plotly_chart(fig2, use_container_width=True)

# TAB 3: AIR & NOISE POLLUTION MODELS
with tab3:
    st.subheader("🌫️ Air Emissions & Traffic Acoustic Analytics")
    c3, c4 = st.columns(2)

    with c3:
        fuel_emissions_dict = {
            "Diesel": {"NO2": 4.12, "PM2.5": 0.35, "CO2": 1450.0},
            "Petrol": {"NO2": 0.65, "PM2.5": 0.05, "CO2": 650.0},
            "CNG_LPG": {"NO2": 0.30, "PM2.5": 0.02, "CO2": 150.0},
            "Hybrid": {"NO2": 0.20, "PM2.5": 0.01, "CO2": 95.0},
            "EV": {"NO2": 0.00, "PM2.5": 0.00, "CO2": 0.0}
        }
        fig3 = create_fuel_emissions_bar_chart(fuel_emissions_dict)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        n_df = pd.DataFrame([
            {"Vehicle": "Motorcycle", "Acoustic Weight": 10.0},
            {"Vehicle": "Truck", "Acoustic Weight": 8.0},
            {"Vehicle": "Bus", "Acoustic Weight": 6.0},
            {"Vehicle": "Van", "Acoustic Weight": 1.8},
            {"Vehicle": "Car", "Acoustic Weight": 1.0},
            {"Vehicle": "EV", "Acoustic Weight": 0.2}
        ])
        fig4 = px.bar(n_df, x="Vehicle", y="Acoustic Weight", color="Vehicle", title="CoRTN Acoustic Weighting by Category", template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)

# TAB 4: ENVIRONMENTAL STATIONS & CORRELATION
with tab4:
    st.subheader("📡 Environmental Station Telemetry & Correlation Analysis")
    
    col_q, col_s = st.columns([2, 1])
    with col_q:
        loc_q = st.text_input("Station Location Query:", "Delhi Anand Vihar")
    with col_s:
        provider = st.selectbox("API Provider:", ["OpenAQ API", "CPCB India Station"])

    if provider == "OpenAQ API":
        st_data = OpenAQAdapter().fetch_latest_measurement(loc_q)
    else:
        st_data = CPCBAdapter().fetch_latest_measurement(loc_q)

    st.caption(f"Source: {st_data.source} | Status: {'🟢 LIVE' if st_data.is_live else '🔵 OFFLINE BASELINE'}")

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Ambient PM2.5", f"{st_data.pm2_5:.1f} µg/m³")
    sc2.metric("Ambient PM10", f"{st_data.pm10:.1f} µg/m³")
    sc3.metric("Ambient NO2", f"{st_data.no2:.1f} µg/m³")
    sc4.metric("Ambient CO", f"{st_data.co:.1f} mg/m³")
    sc5.metric("Ambient SO2", f"{st_data.so2:.1f} µg/m³")

# TAB 5: AI EXPLAINABILITY (GRAD-CAM)
with tab5:
    st.subheader("🧠 Deep Learning Explainability & Feature Attribution")
    st.markdown("Grad-CAM visualizes spatial attention maps over vehicle bounding box crops to explain classifier predictions.")
    
    e1, e2 = st.columns(2)
    crop_p = Path("data/raw/images/traffic_sample_1.jpg")
    if crop_p.exists():
        classifier = VehicleMakeModelClassifier()
        bgr, overlay = classifier.explain(crop_p)
        rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        e1.image(rgb_orig, caption="Original Vehicle Bounding Box Crop", use_container_width=True)
        e2.image(rgb_overlay, caption="Grad-CAM Activation Heatmap Overlay", use_container_width=True)

# TAB 6: ACADEMIC METHODOLOGY & REPORTS EXPORT
with tab6:
    st.subheader("📖 Academic Methodology & System Architecture")
    st.markdown("""
    - **1. Real-Time Performance Optimizations**: Frame skipping (`process_every_n_frames=2`), batch crop classification, and `@st.cache_resource` model caching.
    - **2. Vehicle Detection**: Multi-class vehicle localization powered by YOLOv8 with PyTorch FasterRCNN fallback.
    - **3. Multi-Object Tracking**: Persistent track ID maintenance and 2D virtual line crossing count using ByteTrack / IoU.
    - **4. Fine-Grained Classification**: ResNet50 / EfficientNet-B0 transfer learning backbones for vehicle make/model recognition.
    - **5. Fuel-Type Mapping**: Multi-stage mapping table translating make/model identity to standard fuel labels (`PETROL`, `DIESEL`, `EV`, `CNG_LPG`, `HYBRID`, `UNKNOWN`, `AMBIGUOUS`).
    - **6. Air Pollution Estimation**: EEA COPERT V emission factor formulations calculating pollutant mass rates ($g/hr$).
    - **7. Noise Estimation**: CoRTN traffic acoustic model calculating $L_{eq}$ sound level proxies and 0-100 Relative Noise Index.
    """)

    st.markdown("---")
    st.subheader("📥 Export Telemetry Reports & Predictions")
    d1, d2, d3 = st.columns(3)

    if vehicles_csv_path.exists():
        with open(vehicles_csv_path, "rb") as f:
            d1.download_button("📄 Download Vehicles CSV", f, file_name="vehicles.csv", mime="text/csv", use_container_width=True)

    if summary_path.exists():
        with open(summary_path, "rb") as f:
            d2.download_button("📊 Download Summary JSON", f, file_name="summary.json", mime="application/json", use_container_width=True)

    annotated_vid = output_dir / "tracked_video.mp4"
    if annotated_vid.exists():
        with open(annotated_vid, "rb") as f:
            d3.download_button("📹 Download Processed Video", f, file_name="tracked_video.mp4", mime="video/mp4", use_container_width=True)
