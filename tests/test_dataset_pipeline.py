import os
import pytest
import pandas as pd
from pathlib import Path

from src.data.fuel_lookup import FuelLookupManager, ALLOWED_FUEL_TYPES
from src.data.validator import DatasetValidator
from src.data.yolo_converter import YOLOConverter
from src.data.pipeline import DatasetPipeline

@pytest.fixture
def temp_dataset_dir(tmp_path):
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    return raw, processed

def test_fuel_lookup_allowed_types():
    manager = FuelLookupManager()
    fuel, conf = manager.lookup("Tesla", "Model 3")
    assert fuel in ALLOWED_FUEL_TYPES
    assert conf > 0.0

def test_invalid_fuel_label_handling(tmp_path):
    csv_path = tmp_path / "fuel_mapping.csv"
    with open(csv_path, "w") as f:
        f.write("manufacturer,model,variant,fuel_type,source,confidence,notes\n")
        f.write("FakeMake,FakeModel,v1,SOLAR_POWER,User,1.0,Invalid fuel\n")

    manager = FuelLookupManager(mapping_csv=csv_path)
    fuel, conf = manager.lookup("FakeMake", "FakeModel")
    assert fuel in ALLOWED_FUEL_TYPES
    assert fuel == "UNKNOWN"

def test_missing_file_validation(tmp_path):
    missing_img = tmp_path / "non_existent.jpg"
    assert not DatasetValidator.is_valid_image(missing_img)

def test_malformed_yolo_annotation(tmp_path):
    annot_file = tmp_path / "bad_annot.txt"
    with open(annot_file, "w") as f:
        f.write("0 0.5 0.5 0.2\n") # Only 4 numbers instead of 5
        f.write("999 1.5 0.5 0.2 0.3\n") # Out of bound class and coordinate

    is_valid, errors = DatasetValidator.validate_yolo_annotation(annot_file, 640, 480)
    assert not is_valid
    assert len(errors) >= 2

def test_duplicate_detection(tmp_path):
    # Create two identical dummy files
    f1 = tmp_path / "img1.txt"
    f2 = tmp_path / "img2.txt"
    content = b"header_data_123456"
    f1.write_bytes(content)
    f2.write_bytes(content)

    h1 = DatasetValidator.compute_image_hash(f1)
    h2 = DatasetValidator.compute_image_hash(f2)
    assert h1 == h2

def test_train_test_leakage(tmp_path):
    import cv2
    import numpy as np
    
    f1 = tmp_path / "train_img.png"
    f2 = tmp_path / "test_img.png"
    dummy_img = np.ones((10, 10, 3), dtype=np.uint8) * 150
    cv2.imwrite(str(f1), dummy_img)
    cv2.imwrite(str(f2), dummy_img)

    has_leakage, leaked = DatasetValidator.check_split_leakage([f1], [f2])
    assert has_leakage
    assert len(leaked) == 1

def test_yolo_converter_math():
    # Box 100, 100 to 300, 300 in 1000x1000 image
    xc, yc, w, h = YOLOConverter.bbox_to_yolo(100, 100, 300, 300, 1000, 1000)
    assert pytest.approx(xc, 0.001) == 0.20
    assert pytest.approx(yc, 0.001) == 0.20
    assert pytest.approx(w, 0.001) == 0.20
    assert pytest.approx(h, 0.001) == 0.20
