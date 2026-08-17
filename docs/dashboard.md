# Streamlit Dashboard & UI Reference

## User Interface Design

The Streamlit dashboard (`app/dashboard.py`) is structured into responsive tabs:

1. **📊 Overview & Stream Tab**:
   - Media player with HUD telemetry overlays (bounding boxes, track IDs, vehicle class, fuel type, confidence).
   - Tracked vehicle inventory table.
2. **🚗 Powertrain Breakdown Tab**:
   - Vehicle category pie chart & fuel distribution donut chart using explicit powertrain color mapping (`Petrol`=Blue, `Diesel`=Gray, `EV`=Green, `CNG`=Orange, `Hybrid`=Purple, `Unknown`=Light Gray).
3. **🌫️ Pollution & Noise Models Tab**:
   - Mass emission rate bar charts ($g/hr$) and CoRTN acoustic weighting chart.
   - Diurnal pollution and noise index time series.
4. **📡 Environmental Stations Tab**:
   - OpenAQ and CPCB ambient air station queries.
   - Pearson ($r$) and Spearman ($\rho$) correlation scatter plots.
5. **🧠 AI Explainability Tab**:
   - Grad-CAM heatmap overlays highlighting visual features.
6. **📖 Project Methodology Tab**:
   - Academic methodology summary and report download manager (`vehicles.csv`, `summary.json`, `tracked_video.mp4`).
