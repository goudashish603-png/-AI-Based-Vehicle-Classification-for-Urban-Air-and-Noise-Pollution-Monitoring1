import streamlit as st
import numpy as np
import plotly.express as px
from pathlib import Path

from app.components.sidebar import render_sidebar
from src.noise import NoisePollutionEstimator
from src.visualization import create_noise_meter_chart

st.set_page_config(page_title="Noise Monitoring", page_icon="🔊", layout="wide")

settings = render_sidebar()

st.title("🔊 Relative Noise Pollution Index & Acoustic Monitor")
st.caption("CoRTN Road Traffic Noise Propagation Modeling & Waveform Signal Processing")

estimator = NoisePollutionEstimator()

tab1, tab2 = st.tabs(["🛣️ Road Traffic Acoustic Propagation (CoRTN)", "🎙️ Audio File Signal Processing"])

with tab1:
    st.subheader("Calculation of Road Traffic Noise (CoRTN Standard)")

    col_inputs, col_gauge = st.columns([1, 1])

    with col_inputs:
        num_cars = st.slider("Number of Passenger Cars", 0, 50, 15)
        num_buses = st.slider("Number of Transit Buses", 0, 20, 3)
        num_trucks = st.slider("Number of Heavy Trucks", 0, 20, 2)
        num_bikes = st.slider("Number of Motorcycles", 0, 30, 5)

        speed_kmh = st.slider("Average Traffic Speed (km/h)", 10, 120, 50)
        distance_m = st.slider("Observer Distance to Roadway (Meters)", 5, 100, 15)

    vehicle_records = []
    vehicle_records.extend([{"class_name": "car", "speed_kmh": speed_kmh}] * num_cars)
    vehicle_records.extend([{"class_name": "bus", "speed_kmh": speed_kmh}] * num_buses)
    vehicle_records.extend([{"class_name": "truck", "speed_kmh": speed_kmh}] * num_trucks)
    vehicle_records.extend([{"class_name": "motorcycle", "speed_kmh": speed_kmh}] * num_bikes)

    est_noise = estimator.estimate_from_traffic(vehicle_records, distance_meters=distance_m)

    with col_gauge:
        fig_meter = create_noise_meter_chart(
            est_noise.equivalent_sound_level_dba,
            est_noise.noise_category
        )
        st.plotly_chart(fig_meter, use_container_width=True)

        st.metric("Estimated Peak Sound Level (Lmax)", f"{est_noise.peak_sound_level_dba:.1f} dBA")
        st.metric("Relative Noise Index (0-100 Scale)", f"{est_noise.relative_noise_index:.1f}")

with tab2:
    st.subheader("🎙️ Acoustic WAV Waveform Analysis")

    sample_audio_path = Path("data/raw/audio/traffic_noise.wav")
    
    uploaded_audio = st.file_uploader("Upload Acoustic Audio File (WAV format)", type=["wav"])

    target_audio = None
    if uploaded_audio is not None:
        target_audio = uploaded_audio
    elif sample_audio_path.exists():
        st.info("Using sample synthetic traffic noise WAV file.")
        target_audio = str(sample_audio_path)

    if target_audio is not None:
        if isinstance(target_audio, str):
            st.audio(target_audio, format="audio/wav")
            audio_est = estimator.analyze_audio_file(target_audio)
        else:
            # Temporary file save for upload
            import tempfile
            target_audio.seek(0)
            aud_bytes = target_audio.getvalue()
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tfile.write(aud_bytes)
            tfile.close()
            st.audio(tfile.name, format="audio/wav")
            audio_est = estimator.analyze_audio_file(tfile.name)

        c1, c2, c3 = st.columns(3)
        c1.metric("Equivalent Continuous Sound (Leq)", f"{audio_est.equivalent_sound_level_dba:.1f} dBA")
        c2.metric("Peak Sound Level (Lmax)", f"{audio_est.peak_sound_level_dba:.1f} dBA")
        c3.metric("Acoustic Category", audio_est.noise_category)

        if audio_est.acoustic_features:
            st.json(audio_est.acoustic_features)
