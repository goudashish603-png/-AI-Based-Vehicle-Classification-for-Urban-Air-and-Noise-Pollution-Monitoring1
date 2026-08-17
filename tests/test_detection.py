import numpy as np
import pytest
from pathlib import Path

from src.detection.types import BoundingBox, DetectionResult
from src.detection.detector import VehicleDetector, DetectionBatchResult
from src.detection.preprocessing import letterbox_resize, load_image
from src.detection.postprocessing import map_coco_class_name, export_detections_to_csv, export_detections_to_json

def test_bounding_box_properties():
    box = BoundingBox(10.0, 20.0, 110.0, 120.0)
    assert box.width == 100.0
    assert box.height == 100.0
    assert box.area == 10000.0
    assert box.center == (60.0, 70.0)
    assert box.to_int_tuple() == (10, 20, 110, 120)

def test_coco_class_mapping():
    assert map_coco_class_name(2, "car") == "car"
    assert map_coco_class_name(5, "bus") == "bus"
    assert map_coco_class_name(7, "truck") == "truck"
    assert map_coco_class_name(3, "motorcycle") == "motorcycle"
    assert map_coco_class_name(2, "Minivan") == "van"
    assert map_coco_class_name(1, "bicycle") == "other_vehicle"

def test_letterbox_resize():
    dummy = np.ones((480, 640, 3), dtype=np.uint8) * 100
    padded, scale, (pad_w, pad_h) = letterbox_resize(dummy, (640, 640))
    assert padded.shape == (640, 640, 3)
    assert scale <= 1.0

def test_detector_api():
    detector = VehicleDetector()
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Test predict_frame
    batch_res = detector.predict_frame(dummy_frame, frame_idx=1)
    assert isinstance(batch_res, DetectionBatchResult)
    assert batch_res.frame_idx == 1

    # Test predict_image
    annotated, img_batch = detector.predict_image(dummy_frame)
    assert annotated.shape == dummy_frame.shape
    assert isinstance(img_batch, DetectionBatchResult)

def test_detection_exporters(tmp_path):
    records = [
        {
            "frame_idx": 0,
            "timestamp": "2026-08-12 12:00:00.000",
            "track_id": 1,
            "class_name": "car",
            "class_id": 2,
            "confidence": 0.92,
            "bbox_x1": 10,
            "bbox_y1": 20,
            "bbox_x2": 100,
            "bbox_y2": 100,
            "fuel_type": "PETROL"
        }
    ]
    csv_file = tmp_path / "test_out.csv"
    json_file = tmp_path / "test_out.json"

    export_detections_to_csv(records, csv_file)
    export_detections_to_json(records, json_file)

    assert csv_file.exists()
    assert json_file.exists()
