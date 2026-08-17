# Academic Dataset Acquisition & Preparation Guide

This guide details step-by-step instructions for acquiring, structuring, and preparing datasets for vehicle detection, fine-grained vehicle classification, fuel-type mapping, and urban noise monitoring.

---

## 1. Stanford Cars Dataset

- **Description:** Fine-grained car dataset containing 16,185 images of 196 car classes.
- **Official URL:** [https://ai.stanford.edu/~jkrause/cars/car_dataset.html](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- **Expected Local Path:** `data/raw/stanford_cars/`

### Directory Structure
```
data/raw/stanford_cars/
├── car_ims/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── cars_annos.mat
└── cars_meta.mat
```

---

## 2. CompCars (Comprehensive Cars Dataset)

- **Description:** Large-scale fine-grained vehicle dataset containing 136,727 images of 1,687 car models.
- **Official URL:** [https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/)
- **Expected Local Path:** `data/raw/comp_cars/`

### Directory Structure
```
data/raw/comp_cars/
├── image/
│   ├── 1/           # Make ID
│   │   ├── 1/       # Model ID
│   │   │   ├── 2011/
│   │   │   └── 2012/
└── label/
```

---

## 3. UA-DETRAC Multi-Object Vehicle Detection & Tracking

- **Description:** Multi-object vehicle detection and tracking benchmark containing 10 hours of video captured at 24 fps (140,000 frames).
- **Official URL:** [https://detrac-db.rit.albany.edu/](https://detrac-db.rit.albany.edu/)
- **Expected Local Path:** `data/raw/ua_detrac/`

### Directory Structure
```
data/raw/ua_detrac/
├── DETRAC-images/
│   ├── MVI_20011/
│   ├── MVI_20012/
│   └── ...
└── DETRAC-Train-Annotations-XML/
```

---

## 4. AI City Challenge

- **Description:** NVIDIA AI City Challenge dataset for vehicle re-identification, detection, and tracking.
- **Official URL:** [https://www.aicitychallenge.org/](https://www.aicitychallenge.org/)
- **Expected Local Path:** `data/raw/aicity/`

---

## 5. UrbanSound8K & ESC-50 Audio Datasets

- **Description:** 8,732 labeled sound excerpts of urban acoustic noise (engine idle, car horn, siren, traffic).
- **Official URL:** [https://urbansounddataset.org/urbansound8k/](https://urbansounddataset.org/urbansound8k/)
- **Expected Local Path:** `data/raw/urbansound8k/`

---

## 6. Fuel Type Mapping Engine (`data/external/fuel_mapping.csv`)

Fuel type is mapped using the reference CSV table:

```csv
manufacturer,model,variant,fuel_type,source,confidence,notes
Tesla,Model 3,Standard Range,EV,Official,1.00,Fully electric sedan
BMW,3 Series,320d,DIESEL,Official,0.95,2.0L Diesel Inline-4
BMW,3 Series,Generic,AMBIGUOUS,Official,0.50,Offered in Petrol Diesel and Hybrid
```

### Allowed Fuel Labels
- `PETROL`
- `DIESEL`
- `EV`
- `CNG_LPG`
- `HYBRID`
- `UNKNOWN`
- `AMBIGUOUS` (used when a single car model variant supports multiple powertrain options)

### Updating Mapping manually
Run CLI command:
```bash
python scripts/prepare_datasets.py --update-mapping Tesla "Model Y" "Performance" EV
```

---

## 7. Running Dataset Preparation Pipeline

Execute:
```bash
python scripts/prepare_datasets.py
```
Outputs:
- Metadata CSV: `data/processed/unified_dataset_metadata.csv`
- Report: `outputs/reports/dataset_summary.md`
- Figures: `outputs/figures/dataset_distribution.png`
