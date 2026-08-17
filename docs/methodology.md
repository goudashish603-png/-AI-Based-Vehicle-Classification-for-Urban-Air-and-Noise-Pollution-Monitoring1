# Machine Learning Methodology & Operational Pipeline

## Pipeline Execution Stages

1. **Video Frame Ingestion & Preprocessing**:
   - Frames are extracted at native video frame rates.
   - Letterbox resizing with padding preserves aspect ratio for YOLO detection.

2. **Vehicle Detection (YOLOv8)**:
   - Locates vehicle bounding boxes and assigns initial COCO vehicle class labels (`car`, `truck`, `bus`, `motorcycle`, `van`).
   - Confidence thresholding filters weak detections.

3. **Multi-Object Tracking & Line Counting**:
   - Tracks maintain persistent IDs across consecutive frames using Kalman filter / IoU association.
   - Virtual line counters detect 2D segment crossings to prevent duplicate vehicle counts.

4. **Crop Classification & Fuel Inference**:
   - Localized vehicle bounding box crops are fed into PyTorch ResNet50 classifier to predict make/model.
   - The predicted make/model is queried against `data/external/fuel_mapping.csv` to infer fuel type.

5. **Environmental Emission & Acoustic Estimation**:
   - EEA COPERT V emission factors compute mass rates ($g/hr$) for $PM_{2.5}, PM_{10}, NO_2, CO, SO_2, CO_2$.
   - CoRTN acoustic model computes $L_{eq}$ sound level proxies and 0-100 Relative Noise Index.

6. **Results Aggregation & Dashboard Presentation**:
   - Per-vehicle data is saved to `outputs/predictions/vehicles.csv`.
   - Fleet metrics are saved to `outputs/predictions/summary.json`.
