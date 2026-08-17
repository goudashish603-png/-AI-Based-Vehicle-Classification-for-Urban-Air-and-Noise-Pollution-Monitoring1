# Automated Test Suite & Quality Assurance

## PyTest Architecture

The system includes a 44-case automated test suite covering all pipeline modules:

```bash
python run.py --test
```

### Test Suite Map (`tests/`)

- **`test_detection.py`**: YOLO object detector initialization, confidence filtering, letterbox resizing, CSV/JSON output generation.
- **`test_tracking.py`**: ByteTrack / IoU multi-object tracking, persistent ID assignment, 2D line crossing intersection, dwell time calculation.
- **`test_classification.py`**: PyTorch ResNet50 classifier forward pass, batch inference (`predict_batch`), evaluation metrics, Grad-CAM explainability overlay generation.
- **`test_fuel_mapping.py`**: Multi-stage fuel inference engine (exact match, normalized match, model ambiguity, fuzzy match, `UNKNOWN` fallback).
- **`test_pollution.py`**: EEA COPERT V emission factor database, single vehicle emission formula, fleet aggregation, 0-100 Vehicle Pollution Contribution Index.
- **`test_noise.py`**: CoRTN acoustic weighting model, EV acoustic footprint, $20\log_{10}$ distance attenuation, 0-100 Relative Noise Index, PyTorch 1D CNN waveform classifier.
- **`test_environmental.py`**: OpenAQ and CPCB station adapters, response caching, offline station baseline fallback, Pearson ($r$) and Spearman ($\rho$) correlation math.
- **`test_pipeline.py`**: Unified end-to-end inference pipeline on image and video inputs.
- **`test_dataset_pipeline.py`**: Image corruption validation, MD5 duplicate check, train/test leakage detection, YOLO annotation format converter.
