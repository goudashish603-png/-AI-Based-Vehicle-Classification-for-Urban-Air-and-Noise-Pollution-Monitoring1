"""
Dataset Acquisition & Documentation Utility

Provides dataset download pointers, download scripts, and directory structures 
for academic vehicle & noise pollution datasets (UA-DETRAC, Stanford Cars, UrbanSound8K, OpenAQ).
"""
import os
import sys
from pathlib import Path

DATASETS_INFO = {
    "UA-DETRAC": {
        "description": "Multi-object vehicle detection and tracking benchmark dataset",
        "url": "https://detrac-db.rit.albany.edu/",
        "target_dir": "data/raw/ua_detrac"
    },
    "Stanford Cars": {
        "description": "Fine-grained vehicle dataset containing 16,185 images of 196 car classes",
        "url": "https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
        "target_dir": "data/raw/stanford_cars"
    },
    "CompCars": {
        "description": "Comprehensive Car Dataset with fine-grained vehicle types and view points",
        "url": "https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/",
        "target_dir": "data/raw/comp_cars"
    },
    "UrbanSound8K": {
        "description": "Dataset containing 8,732 labeled sound excerpts of urban sounds (engine, horn, traffic)",
        "url": "https://urbansounddataset.org/urbansound8k/",
        "target_dir": "data/raw/urbansound8k"
    }
}

def print_dataset_instructions():
    print("=" * 70)
    print("PUBLIC ACADEMIC DATASET ACQUISITION GUIDE")
    print("=" * 70)
    for name, info in DATASETS_INFO.items():
        print(f"\n[+] {name}")
        print(f"    Description : {info['description']}")
        print(f"    Official URL: {info['url']}")
        print(f"    Target Path : {info['target_dir']}")
    print("\nNote: Please download the desired datasets from their official sources")
    print("and place them into the respective target paths for model training.")
    print("For zero-setup testing, run `python scripts/prepare_sample_data.py`.")
    print("=" * 70)

if __name__ == "__main__":
    print_dataset_instructions()
