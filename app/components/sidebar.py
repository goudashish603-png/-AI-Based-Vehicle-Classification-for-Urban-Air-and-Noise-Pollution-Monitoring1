import streamlit as st
import torch
from pathlib import Path
from src.utils.config import load_config
from src.utils.device import get_device

def render_sidebar():
    """Renders the custom Streamlit sidebar with system status and controls."""
    config = load_config()
    device = get_device(config.get("system", {}).get("device", "auto"))

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/traffic-jam.png", width=64)
        st.title("Vehicle AI Pollution Monitor")
        st.caption("Final Year University Engineering Project")
        st.markdown("---")

        st.subheader("⚙️ System Status")
        dev_type = "GPU (CUDA)" if device.type == "cuda" else ("Apple MPS" if device.type == "mps" else "CPU Fallback")
        dev_color = "🟢" if device.type in ["cuda", "mps"] else "🔵"
        st.markdown(f"**Compute Device:** {dev_color} `{dev_type}`")
        st.markdown(f"**Detector Model:** `YOLOv8n`")
        st.markdown(f"**Classifier:** `ResNet50 Fine-Grained`")
        st.markdown(f"**Emissions Std:** `COPERT V / EEA`")
        
        st.markdown("---")
        st.subheader("🎛️ Pipeline Settings")
        conf_thresh = st.slider("Detection Confidence", 0.1, 0.9, 0.45, 0.05)
        iou_thresh = st.slider("Tracking IoU Threshold", 0.1, 0.8, 0.30, 0.05)
        
        st.markdown("---")
        st.info(
            "💡 **Scientific Limitation Notice:**\n"
            "Camera system estimates **Vehicle Pollution Contribution** ($g/hr$) "
            "and **Relative Noise Index** ($dB_A$). Ambient air quality is fetched "
            "from OpenAQ / CPCB stations."
        )

        return {
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
            "device": device
        }
