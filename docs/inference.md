# End-to-End Inference Guide

## CLI Inference Commands

### 1. Process Input Video File
```bash
python run.py --input data/raw/videos/sample_traffic.mp4 --output outputs/predictions --confidence 0.40
```

### 2. Process Input Image File
```bash
python run.py --input data/raw/images/traffic_sample_1.jpg --output outputs/predictions
```

### 3. Launch Streamlit Web Dashboard
```bash
python run.py --dashboard
```

### 4. Run System Unit Tests
```bash
python run.py --test
```

### 5. Output Telemetry Schema (`outputs/predictions/vehicles.csv`)
- `track_id`: Unique vehicle tracking identifier.
- `vehicle_type`: COCO category (`car`, `truck`, `bus`, `motorcycle`, `van`).
- `manufacturer`: Vehicle make (e.g. `Toyota`, `Tesla`).
- `model`: Vehicle model (e.g. `Prius`, `Model 3`).
- `fuel_type`: Inferred fuel label (`PETROL`, `DIESEL`, `EV`, `CNG_LPG`, `HYBRID`, `UNKNOWN`, `AMBIGUOUS`).
- `estimated_pollution_score`: Sum of $PM_{2.5} + NO_2$ estimated mass rate.
- `noise_score`: CoRTN $L_{eq}$ sound level proxy.
