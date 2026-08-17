import streamlit as st
import cv2
import numpy as np
import torch
import plotly.express as px
from pathlib import Path

from app.components.sidebar import render_sidebar
from src.classification import VehicleClassifier, GradCAM, FUEL_CLASSES
from src.fuel_mapping import FuelTypeMapper

st.set_page_config(page_title="Grad-CAM Explainability", page_icon="🧠", layout="wide")

settings = render_sidebar()

st.title("🧠 Fine-Grained Classification & Grad-CAM Visual Explainability")
st.caption("Inspect deep neural network activation heatmaps driving vehicle fuel classification decisions")

@st.cache_resource
def load_classifier_and_cam():
    classifier = VehicleClassifier()
    cam_explainer = GradCAM(classifier.model)
    mapper = FuelTypeMapper()
    return classifier, cam_explainer, mapper

classifier, cam_explainer, mapper = load_classifier_and_cam()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📷 Vehicle Image Input")
    
    upload_crop = st.file_uploader("Upload Vehicle Crop (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    crop_bgr = None
    if upload_crop is not None:
        file_bytes = np.asarray(bytearray(upload_crop.read()), dtype=np.uint8)
        crop_bgr = cv2.imdecode(file_bytes, 1)
    else:
        # Load sample cropped vehicle
        sample_path = Path("data/raw/images/traffic_sample_1.jpg")
        if sample_path.exists():
            full_img = cv2.imread(str(sample_path))
            # Crop a vehicle region
            crop_bgr = full_img[280:400, 150:370].copy()
            st.info("Using sample vehicle crop for demonstration.")

    if crop_bgr is not None:
        st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), caption="Cropped Vehicle Region", width=320)

with col_right:
    if crop_bgr is not None:
        st.subheader("📊 PyTorch Model Prediction")
        
        vehicle_class = st.selectbox("Vehicle Category Context", ["car", "bus", "truck", "motorcycle"])
        
        # Inference
        raw_fuel, raw_conf, probs = classifier.predict_crop(crop_bgr, vehicle_class)
        final_fuel, final_conf = mapper.map_fuel_type(vehicle_class, raw_fuel, raw_conf, probs)

        st.success(f"**Predicted Fuel Type:** `{final_fuel}` (Confidence: `{int(final_conf*100)}%`)")
        st.info(f"**Raw Fine-Grained Vision Label:** `{raw_fuel}` (`{int(raw_conf*100)}%`)")

        # Plot Probability Bar Chart
        prob_df = {"Fuel Class": list(probs.keys()), "Probability": list(probs.values())}
        fig = px.bar(
            prob_df,
            x="Probability",
            y="Fuel Class",
            orientation="h",
            text_auto=".2f",
            title="Classification Probabilities",
            color="Probability",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

if crop_bgr is not None:
    st.subheader("🔥 Grad-CAM Heatmap Analysis")
    
    alpha = st.slider("Heatmap Overlay Transparency (Alpha)", 0.1, 0.9, 0.5, 0.05)

    # Convert to Tensor for Grad-CAM
    rgb_img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    tensor = classifier.transform(pil_img).unsqueeze(0).to(classifier.device)

    # Generate CAM
    cam_heatmap, target_idx = cam_explainer.generate_cam(tensor)
    overlay = cam_explainer.overlay_heatmap(crop_bgr, cam_heatmap, alpha=alpha)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(rgb_img, caption="1. Original Crop", use_container_width=True)
    with c2:
        cam_vis = np.uint8(255 * cam_heatmap)
        cam_jet = cv2.applyColorMap(cv2.resize(cam_vis, (crop_bgr.shape[1], crop_bgr.shape[0])), cv2.COLORMAP_JET)
        st.image(cv2.cvtColor(cam_jet, cv2.COLOR_BGR2RGB), caption="2. Activation Heatmap", use_container_width=True)
    with c3:
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="3. Superimposed Grad-CAM", use_container_width=True)

    st.markdown("""
    > **Grad-CAM Interpretation:**
    > Red & Yellow regions highlight key visual features (front grille, headlight contours, body height ratio, badges) 
    > that contributed highest positive gradients to the ResNet50 neural network's fuel classification output.
    """)
