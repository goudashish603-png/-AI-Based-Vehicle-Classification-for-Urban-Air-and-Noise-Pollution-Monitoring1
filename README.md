# AI-Based Vehicle Classification for Urban Air and Noise Pollution Monitoring

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20Vision-green.svg)](https://ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Build Status](https://img.shields.io/badge/Tests-44%20PASSED%20%28100%25%29-brightgreen.svg)]()

---

## 📌 Abstract

Rapid urbanization and expanding vehicular traffic are major drivers of urban air degradation and environmental noise pollution. Traditional air quality and noise monitoring rely on sparse physical sensing stations, which lack fine-grained spatial coverage and real-time vehicular source attribution. This project presents an end-to-end Computer Vision and Machine Learning framework titled **"AI-Based Vehicle Classification for Urban Air and Noise Pollution Monitoring"**. The system processes RGB traffic camera streams to detect, track, classify fine-grained vehicle make/models, infer powertrain fuel types, and calculate model-based estimates of vehicular air emissions ($PM_{2.5}, PM_{10}, NO_2, CO, SO_2, CO_2$) using European Environment Agency (EEA) COPERT V standards, alongside relative acoustic noise indices using Calculation of Road Traffic Noise (CoRTN) acoustic models.

---

## 🚨 Problem Statement

Urban municipal authorities face critical challenges in monitoring and mitigating transportation emissions:
1. **Lack of Source Attribution:** Physical ambient air sensors measure cumulative pollutant concentrations but cannot attribute pollution spikes to specific vehicle categories or fuel types (e.g. diesel trucks vs. petrol cars).
2. **High Sensing Infrastructure Costs:** Installing high-density physical gas sensors and calibrated decibel meters across city intersections is cost-prohibitive.
3. **Coarse Temporal Telemetry:** Manual traffic surveys fail to capture real-time diurnal traffic fluctuations, congestion dynamics, and vehicle fleet composition.

---

## 🎯 Project Objectives

1. **Multi-Class Vehicle Detection & Tracking:** Implement real-time vehicle localization and persistent multi-object tracking across video streams.
2. **Fine-Grained Make/Model Classification:** Train PyTorch deep convolutional networks (ResNet50 / EfficientNet) to classify vehicle make/models from cropped bounding boxes.
3. **Powertrain Fuel-Type Mapping:** Construct a hierarchical database mapper translating vehicle identity to standard fuel types (`PETROL`, `DIESEL`, `EV`, `CNG_LPG`, `HYBRID`, `UNKNOWN`).
4. **Air Emission Modeling:** Compute estimated mass emission rates ($g/hr$) using EEA COPERT V emission factors and calculate a normalized 0-100 **Vehicle Pollution Contribution Index**.
5. **Acoustic Noise Index Estimation:** Formulate traffic noise proxies using CoRTN acoustic models to produce a 0-100 **Relative Noise Pollution Index**.
6. **Ambient Data & Correlation Analysis:** Integrate live OpenAQ and CPCB station APIs to compute Pearson ($r$) and Spearman ($\rho$) correlation against traffic activity.
7. **Interactive Dashboard:** Build a production Streamlit web application featuring real-time HUD overlays, Plotly analytics charts, Grad-CAM explainability, and report downloads.

---

## 🏗️ System Architecture & Complete Flow

```
                      INPUT TRAFFIC VIDEO / IMAGE
                                   │
                           Frame Extraction
                                   │
                         YOLO Vehicle Detection
                       (car, truck, bus, moto, van)
                                   │
                   Multi-Object Tracking & Line Counter
                     (ByteTrack / IoU + Dwell Time)
                                   │
                        Vehicle Crop Extraction
                                   │
                   Make/Model Transfer Classifier
                  (ResNet50 / EfficientNet-B0 + Grad-CAM)
                                   │
                       Fuel-Type Inference Engine
             (Exact, Normalized, Body Ambiguity, Fuzzy, UNKNOWN)
                                   │
                                   ├──────────────────────────┐
                                   ▼                          ▼
                       Air Emission Estimator      Traffic Noise Estimator
                       (EEA COPERT V Standards)     (CoRTN Acoustic Model)
                                   │                          │
                                   ▼                          ▼
                       0-100 Vehicle Pollution        0-100 Relative Noise
                          Contribution Index             Pollution Index
                                   │                          │
                                   └────────────┬─────────────┘
                                                │
                                      Results Aggregation
                                                │
                                  ┌─────────────┴─────────────┐
                                  ▼                           ▼
                         outputs/predictions/        outputs/predictions/
                             vehicles.csv                summary.json
```

---

## 🛠️ Technology Stack

- **Core & Logic:** Python 3.8+
- **Deep Learning Framework:** PyTorch, Torchvision
- **Object Detection:** Ultralytics YOLOv8 (FasterRCNN fallback)
- **Computer Vision & Image Processing:** OpenCV, PIL
- **Data Manipulation & Analytics:** pandas, NumPy, scikit-learn, SciPy
- **Visualization & Web App:** Streamlit, Plotly Express
- **Environmental APIs:** OpenAQ REST API, CPCB India API

---

## 📊 Dataset Description

Supported academic and benchmark traffic datasets:
1. **Stanford Cars Dataset:** 16,185 images of 196 car classes.
2. **CompCars Dataset:** Fine-grained car models with web and surveillance views.
3. **UA-DETRAC Benchmark:** Multi-object vehicle tracking and detection dataset.
4. **AI City Challenge Dataset:** City-scale multi-camera vehicle tracking.
5. **UrbanSound8K & ESC-50:** Environmental audio datasets for traffic acoustic events.

Dataset acquisition instructions, structure validation, and preprocessing scripts are documented in [docs/datasets.md](file:///c:/Users/gouda/OneDrive/AI-Based%20Vehicle%20Classification%20for%20Urban%20Air%20and%20Noise%20Pollution%20Monitoring/docs/datasets.md).

---

## 🧠 Machine Learning Methodology

### 1. Vehicle Detection & Tracking
- **Detection:** YOLOv8 bounding box regression and multi-class classification (`car`, `truck`, `bus`, `motorcycle`, `van`).
- **Tracking:** ByteTrack / IoU tracker maintains persistent track IDs across frames and calculates speed proxies and dwell time.

### 2. Fine-Grained Make/Model Classification
- **Backbone:** ResNet50 / EfficientNet-B0 transfer learning backbones trained with Cross-Entropy Loss and Adam optimizer.
- **Explainability:** Grad-CAM activation maps highlight spatial features (grille, headlights) influencing network predictions.

### 3. Powertrain Fuel-Type Mapping
- Maps manufacturer and model names against reference database (`data/external/fuel_mapping.csv`) using hierarchical matching (exact, normalized, body ambiguity, fuzzy, `UNKNOWN` fallback).

### 4. Air Emissions & Noise Modeling
- **Air Emissions:** Calculates mass rates ($g/hr$) for $PM_{2.5}, PM_{10}, NO_2, CO, SO_2, CO_2$ based on EEA COPERT V factors.
- **Noise Index:** Uses CoRTN acoustic model weighting (motorcycle=10.0, truck=8.0, bus=6.0, car=1.0, EV=0.2).

---

## ⚡ Quick Start & Installation

### 1. Setup Environment
```bash
git clone <repository_url>
cd "AI-Based Vehicle Classification for Urban Air and Noise Pollution Monitoring"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run System Test Suite
```bash
python run.py --test
```

### 3. Run Inference on Video / Image
```bash
# Video Inference
python run.py --input data/raw/videos/sample_traffic.mp4 --output outputs/predictions

# Image Inference
python run.py --input data/raw/images/traffic_sample_1.jpg --output outputs/predictions
```

### 4. Launch Streamlit Web Dashboard
```bash
python run.py --dashboard
```

---

## ⚠️ Scientific Limitations & Disclaimers

1. **Air Pollution Estimation:** Values represent **model-based estimated vehicle emission mass rates** ($g/hr$), NOT direct physical atmospheric concentration ($µg/m³$) or AQI measurements from camera images. Ambient concentration depends on meteorological dispersion (wind speed, humidity, boundary layer height).
2. **Noise Pollution Estimation:** Values represent a **relative noise index** (0-100 scale derived from CoRTN acoustic models), NOT calibrated physical microphone sound pressure levels (dB SPL).
3. **Fuel Type Inference:** Visually identical vehicle body styles (e.g. Ford Focus petrol vs. diesel) cannot be distinguished visually with 100% certainty; `UNKNOWN` or `AMBIGUOUS` is returned when visual features are insufficient.
4. **Correlation vs. Causation:** Statistical correlation between camera traffic activity and ambient air quality stations does **NOT** prove direct physical causation due to background industrial emissions and atmospheric transport.

---

## 👥 Team Contribution Placeholders

- **Lead ML / Computer Vision Engineer:** Model training, YOLO detection, PyTorch ResNet classifier, Grad-CAM.
- **Backend & Data Pipeline Engineer:** Dataset preparation, tracking engine, EEA emission modeling, fuel mapping database.
- **Frontend & UI Engineer:** Streamlit multi-page dashboard, Plotly interactive charts, HUD overlay player.
- **QA & Testing Lead:** Unit test suite, integration tests, performance profiling, documentation.

---

## 📚 References

1. European Environment Agency (EEA), *COPERT V Road Transport Emission Inventory Guidebook*.
2. Department of Transport UK, *Calculation of Road Traffic Noise (CoRTN)*, HMSO.
3. Ultralytics YOLOv8 Documentation, `https://docs.ultralytics.com/`.
4. OpenAQ Global Air Quality API, `https://openaq.org/`.
5. Central Pollution Control Board (CPCB) India, `https://cpcb.gov.in/`.
