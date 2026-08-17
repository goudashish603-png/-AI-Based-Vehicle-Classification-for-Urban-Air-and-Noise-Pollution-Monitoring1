import streamlit as st
import cv2
import tempfile
import numpy as np
from pathlib import Path

from app.components.sidebar import render_sidebar
from app.components.metrics_cards import render_metrics_cards
from src.detection import VehicleDetector
from src.tracking import VehicleTracker
from src.classification import VehicleClassifier
from src.fuel_mapping import FuelTypeMapper
from src.pollution import AirPollutionEstimator
from src.noise import NoisePollutionEstimator
from src.visualization import (
    VisualAnnotator,
    create_fuel_distribution_chart,
    create_emissions_bar_chart,
    create_noise_meter_chart
)

st.set_page_config(page_title="Real-time Traffic AI", page_icon="🎥", layout="wide")

settings = render_sidebar()

st.title("🎥 Real-time Traffic Video AI Processing")
st.caption("Vehicle Detection, Tracking, Fine-Grained Fuel Mapping, & Pollution HUD")

# Instantiate Pipelines
@st.cache_resource
def load_pipeline():
    detector = VehicleDetector()
    classifier = VehicleClassifier()
    mapper = FuelTypeMapper()
    pollution_est = AirPollutionEstimator()
    noise_est = NoisePollutionEstimator()
    return detector, classifier, mapper, pollution_est, noise_est

detector, classifier, mapper, pollution_est, noise_est = load_pipeline()
detector.conf_threshold = settings["conf_thresh"]

# Data Selection Options
sample_img_path = Path("data/raw/images/traffic_sample_1.jpg")
sample_vid_path = Path("data/raw/videos/sample_traffic.mp4")

input_source = st.radio(
    "Select Input Media Source:",
    ["Sample Image", "Sample Video", "Upload Image", "Upload Video"],
    horizontal=True
)

image_to_process = None
video_path_to_process = None

if input_source == "Sample Image":
    if sample_img_path.exists():
        image_to_process = cv2.imread(str(sample_img_path))
    else:
        st.warning("Sample image not found. Run `python scripts/prepare_sample_data.py` first.")

elif input_source == "Sample Video":
    if sample_vid_path.exists():
        video_path_to_process = str(sample_vid_path)
    else:
        st.warning("Sample video not found. Run `python scripts/prepare_sample_data.py` first.")

elif input_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_file.seek(0)
        img_bytes = uploaded_file.getvalue()
        file_bytes = np.frombuffer(img_bytes, dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, 1)

elif input_source == "Upload Video":
    uploaded_vid = st.file_uploader("Upload Traffic Video (MP4/AVI)", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        uploaded_vid.seek(0)
        vid_bytes = uploaded_vid.getvalue()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid_bytes)
        tfile.close()
        video_path_to_process = tfile.name

st.markdown("---")

# Image Processing Flow
if image_to_process is not None:
    st.subheader("🖼️ Image Detection & Classification Analysis")

    # Run Detector
    detections = detector.detect(image_to_process)
    
    # Run Classifier & Fuel Mapper
    vehicle_records = []
    for det in detections:
        if det.vehicle_crop is not None:
            raw_fuel, raw_conf, probs = classifier.predict_crop(det.vehicle_crop, det.class_name)
            final_fuel, final_conf = mapper.map_fuel_type(det.class_name, raw_fuel, raw_conf, probs)
            det.fuel_type = final_fuel
            det.fuel_confidence = final_conf
        else:
            det.fuel_type = "Unknown"
            det.fuel_confidence = 0.50

        vehicle_records.append({
            "class_name": det.class_name,
            "fuel_type": det.fuel_type,
            "speed_kmh": 45.0
        })

    # Estimate Pollution & Noise
    pollution_res = pollution_est.estimate_emissions(vehicle_records, duration_seconds=1.0)
    noise_res = noise_est.estimate_from_traffic(vehicle_records)

    # Render Cards
    render_metrics_cards(
        vehicle_count=len(detections),
        pm25_g_hr=pollution_res.pm2_5_g_per_hr,
        nox_g_hr=pollution_res.nox_g_per_hr,
        co2_g_hr=pollution_res.co2_g_per_hr,
        noise_dba=noise_res.equivalent_sound_level_dba
    )

    # Draw Frame
    annotated = VisualAnnotator.draw_detections(image_to_process, detections)
    annotated = VisualAnnotator.draw_hud_overlay(
        annotated,
        len(detections),
        pollution_res.pm2_5_g_per_hr,
        pollution_res.nox_g_per_hr,
        noise_res.equivalent_sound_level_dba
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Annotated Traffic Scene")

    with col2:
        st.plotly_chart(create_fuel_distribution_chart(pollution_res.fuel_counts), use_container_width=True)
        st.plotly_chart(create_emissions_bar_chart({
            "PM2.5": pollution_res.pm2_5_g_per_hr,
            "PM10": pollution_res.pm10_g_per_hr,
            "NOx": pollution_res.nox_g_per_hr,
            "CO": pollution_res.co_g_per_hr
        }), use_container_width=True)

# Video Processing Flow
elif video_path_to_process is not None:
    st.subheader("📹 Video Inference & Telemetry Dashboard")

    tracker = VehicleTracker(fps=25.0)
    st_video = st.empty()
    st_metrics = st.empty()
    st_charts = st.empty()

    cap = cv2.VideoCapture(video_path_to_process)
    run_video = st.checkbox("Start Video Inference Loop", value=True)

    frame_idx = 0
    last_dets = []

    while cap.isOpened() and run_video and frame_idx < 400:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize large video frames to max 960px width for fast real-time inference
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960.0 / w
            frame = cv2.resize(frame, (960, int(h * scale)), interpolation=cv2.INTER_AREA)

        # Detect every 2nd frame for real-time speedup
        if frame_idx % 2 == 0:
            dets = detector.detect(frame)
            last_dets = dets
        else:
            dets = last_dets

        # Multi-object Tracking
        tracked_dets = tracker.update(dets, frame_idx=frame_idx)

        # Classify & Map
        vehicle_records = []
        for det in tracked_dets:
            if det.vehicle_crop is not None and det.fuel_type is None:
                raw_fuel, raw_conf, probs = classifier.predict_crop(det.vehicle_crop, det.class_name)
                final_fuel, final_conf = mapper.map_fuel_type(det.class_name, raw_fuel, raw_conf, probs)
                det.fuel_type = final_fuel
                det.fuel_confidence = final_conf
            
            vehicle_records.append({
                "class_name": det.class_name,
                "fuel_type": det.fuel_type or "Unknown",
                "speed_kmh": 40.0
            })

        pollution_res = pollution_est.estimate_emissions(vehicle_records, duration_seconds=1.0 / 25.0)
        noise_res = noise_est.estimate_from_traffic(vehicle_records)

        annotated = VisualAnnotator.draw_detections(frame, tracked_dets)
        annotated = VisualAnnotator.draw_hud_overlay(
            annotated,
            len(tracked_dets),
            pollution_res.pm2_5_g_per_hr,
            pollution_res.nox_g_per_hr,
            noise_res.equivalent_sound_level_dba
        )

        st_video.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        frame_idx += 1

    cap.release()
